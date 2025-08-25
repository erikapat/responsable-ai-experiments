#!/usr/bin/env python3
# tunning_v3_fixed.py
import os, math, time, json
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # macOS MPS friendlier

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from diffusers import (
    StableDiffusionPipeline,            # SD 1.5
    StableDiffusionXLPipeline,         # SDXL
    DDPMScheduler
)

# -------------------------
# Config (adjust as needed)
# -------------------------
class Cfg:
    train_dir         = "./my_faces/train"   # folder-of-folders: <prompt_folder>/*.jpg|*.png
    out_dir           = "./sdxl-finetuned"
    model_id          = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")  # or "stabilityai/stable-diffusion-xl-base-1.0"
    image_size        = 256 #512
    epochs            = 100
    steps_per_epoch   = 1000     # None = use all samples once; or set an int
    learning_rate     = 2e-5
    batch_size        = 4        # keep 1 for low VRAM/CPU
    vae_scale         = 0.18215  # SD/SDXL latent scaling
    log_every         = 25
    save_every        = 200      # save every N steps
    save_unet_only   = True      # set False to save the full pipeline (huge)
    seed              = 1234

cfg = Cfg()

# -------------------------
# Device & dtype
# -------------------------
def best_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = best_device()
dtype  = torch.float16 if device == "cuda" else torch.float32
print(f"Training on {device} | dtype={dtype}")

# -------------------------
# Dataset: folder-per-prompt
# root/
#   A_productive_person/
#       01.jpg ...
#   a_person_at_social_services/
#       a.png ...
# -------------------------
class FolderPromptDataset(Dataset):
    def __init__(self, root, size):
        self.samples = []
        self.size    = size
        for prompt_folder in sorted(os.listdir(root)):
            pf = os.path.join(root, prompt_folder)
            if not os.path.isdir(pf):
                continue
            # convert folder name to prompt text
            prompt_text = prompt_folder.replace("_", " ").strip()
            for fn in sorted(os.listdir(pf)):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    self.samples.append((os.path.join(pf, fn), prompt_text))
        if not self.samples:
            raise RuntimeError(f"No images found under {root}")
        print(f"Found {len(self.samples)} images across prompts.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, prompt = self.samples[idx]
        img = Image.open(img_path).convert("RGB").resize((self.size, self.size))
        # to tensor [-1,1] like diffusers preprocess (C,H,W)
        px = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).view(img.size[1], img.size[0], 3).numpy().astype("float32") / 255.0))
        px = px.permute(2, 0, 1) * 2 - 1
        return {"pixel_values": px, "prompt": prompt}

def collate_fn(batch):
    px = torch.stack([b["pixel_values"] for b in batch], dim=0)  # (B,C,H,W)
    prompts = [b["prompt"] for b in batch]
    return {"pixel_values": px, "prompt": prompts}

dataset = FolderPromptDataset(cfg.train_dir, cfg.image_size)
loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)

# -------------------------
# Load pipeline (SD 1.5 or SDXL)
# We disable the safety checker to avoid downloading ~1.2GB on CPU boxes.
# -------------------------
is_sdxl = "xl" in cfg.model_id.lower()
print("Loading pipeline...")
if is_sdxl:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype if device == "cuda" else torch.float32,
        safety_checker=None, requires_safety_checker=False,
        use_safetensors=True,
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype if device == "cuda" else torch.float32,
        safety_checker=None, requires_safety_checker=False,
        use_safetensors=True,
    )

# training scheduler (DDPM) — standard for UNet noise prediction objective
pipe.scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", prediction_type="epsilon")

# Place modules
pipe.unet.to(device, dtype=dtype)
pipe.unet.train()
pipe.unet.enable_gradient_checkpointing()

pipe.vae.to("cpu", dtype=torch.float32)  # keep VAE on CPU to save VRAM
if is_sdxl:
    pipe.text_encoder.to("cpu", dtype=torch.float32)
    pipe.text_encoder_2.to("cpu", dtype=torch.float32)
else:
    pipe.text_encoder.to("cpu", dtype=torch.float32)

for p in pipe.vae.parameters(): p.requires_grad_(False)
if is_sdxl:
    for p in pipe.text_encoder.parameters():   p.requires_grad_(False)
    for p in pipe.text_encoder_2.parameters(): p.requires_grad_(False)
else:
    for p in pipe.text_encoder.parameters():   p.requires_grad_(False)

# Memory helpers (safe if unavailable)
for fn in ("enable_vae_slicing", "enable_vae_tiling", "enable_attention_slicing"):
    try: getattr(pipe, fn)()
    except Exception: pass

optim = torch.optim.AdamW(pipe.unet.parameters(), lr=cfg.learning_rate)

# Reproducibility
g_cpu = torch.Generator("cpu").manual_seed(cfg.seed)

# -------------------------
# Helpers
# -------------------------
def compute_sdxl_time_ids(h, w, b, dtype_, dev_):
    # [orig_h, orig_w, crop_y, crop_x, target_h, target_w]
    t = torch.tensor([h, w, 0, 0, h, w], dtype=dtype_, device=dev_)
    return t.unsqueeze(0).repeat(b, 1)

os.makedirs(cfg.out_dir, exist_ok=True)
log_path = os.path.join(cfg.out_dir, "train_log.jsonl")
log_f = open(log_path, "a", encoding="utf-8")

global_step = 0
max_steps = cfg.steps_per_epoch if cfg.steps_per_epoch is not None else math.ceil(len(loader)/cfg.batch_size)

print(f"Epochs={cfg.epochs} | steps/epoch≈{max_steps} | saving to {cfg.out_dir}")

# -------------------------
# Training loop
# -------------------------
for epoch in range(cfg.epochs):
    pbar = tqdm(loader, total=max_steps, desc=f"Epoch {epoch+1}/{cfg.epochs}")
    step_in_epoch = 0
    for batch in pbar:
        if cfg.steps_per_epoch is not None and step_in_epoch >= cfg.steps_per_epoch:
            break

        # ----- Prepare data -----
        pixel_values = batch["pixel_values"]        # (B,C,H,W) on CPU
        B, C, H, W = pixel_values.shape

        with torch.no_grad():
            # VAE encode on CPU -> latents
            pv = pixel_values.to("cpu", dtype=torch.float32)
            latents_dist = pipe.vae.encode(pv).latent_dist
            latents = (latents_dist.sample() * cfg.vae_scale).to(device, dtype=dtype)  # (B,4,h/8,w/8)

            # noise & timesteps
            noise = torch.randn_like(latents, device=device, dtype=dtype)
            t = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (B,), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

            # Prompt embeddings
            if is_sdxl:
                # SDXL expects (prompt_embeds, pooled), plus time_ids
                pe, _, pooled, _ = pipe.encode_prompt(
                    prompt=batch["prompt"], device="cpu",
                    num_images_per_prompt=1, do_classifier_free_guidance=False,
                )
                prompt_embeds      = pe.to(device, dtype=dtype)
                pooled_prompt_embs = pooled.to(device, dtype=dtype)
                add_time_ids       = compute_sdxl_time_ids(H, W, B, prompt_embeds.dtype, device)
            else:
                # SD 1.5 (CLIP text): tokenize -> text_encoder
                tok = pipe.tokenizer(
                    batch["prompt"], padding=True, truncation=True, return_tensors="pt"
                )
                input_ids = tok.input_ids.to("cpu")
                attn_mask = tok.attention_mask.to("cpu") if hasattr(tok, "attention_mask") else None
                te_out = pipe.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
                prompt_embeds = te_out[0].to(device, dtype=dtype)  # (B,77,768)

        # ----- Forward UNet -----
        optim.zero_grad(set_to_none=True)
        if is_sdxl:
            noise_pred = pipe.unet(
                noisy_latents,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embs, "time_ids": add_time_ids},
            ).sample
        else:
            noise_pred = pipe.unet(
                noisy_latents,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample

        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        loss.backward()

        # light grad clip to be safe on CPU
        torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0)
        optim.step()

        global_step += 1
        step_in_epoch += 1

        if global_step % cfg.log_every == 0:
            rec = {
                "ts": int(time.time()),
                "epoch": epoch+1,
                "step": global_step,
                "loss": float(loss.detach().cpu().item()),
                "lr": float(optim.param_groups[0]["lr"]),
            }
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            pbar.set_postfix(loss=f"{rec['loss']:.4f}")

        if global_step % cfg.save_every == 0:
            ckpt_dir = os.path.join(cfg.out_dir, f"step_{global_step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            if cfg.save_unet_only:
                pipe.unet.save_pretrained(os.path.join(ckpt_dir, "unet"))
            else:
                pipe.save_pretrained(ckpt_dir)
            print(f"Saved checkpoint to {ckpt_dir}")

# -------------------------
# Final save
# -------------------------
final_dir = os.path.join(cfg.out_dir, "final")
os.makedirs(final_dir, exist_ok=True)
if cfg.save_unet_only:
    pipe.unet.save_pretrained(os.path.join(final_dir, "unet"))
else:
    pipe.save_pretrained(final_dir)
log_f.close()
print(f"Done. Weights saved to {final_dir}\nLogs → {log_path}")