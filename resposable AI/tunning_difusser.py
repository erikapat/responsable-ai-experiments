#!/usr/bin/env python3
# tuning_lora_peft_prompt_conditioned.py
# LoRA prompt-condicionado (SD 1.5 / SDXL) usando PEFT en Py 3.12

import os, math, time, json, random
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DDPMScheduler,
)

from peft import LoraConfig, get_peft_model, PeftModel

# -------------------------
# Config
# -------------------------
class Cfg:
    train_dir         = "./my_faces/train"  # carpeta con subcarpetas por prompt (+ opcional .txt por imagen)
    out_dir           = "./finetuned"
    model_id          = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")  # o "stabilityai/stable-diffusion-xl-base-1.0"
    image_size        = 512     # SD1.5 → 512. SDXL ideal 1024 si tienes VRAM
    epochs            = 5
    steps_per_epoch   = 500
    learning_rate     = 1e-4    # LoRA tolera LR relativamente alta
    batch_size       = 4
    vae_scale         = 0.18215
    log_every         = 25
    save_every        = 200
    lora_rank         = 16
    lora_alpha        = 32      # suele ir bien ≈ 2*r
    lora_dropout      = 0.0
    seed              = 1234

    # text encoders congelados; entrenamos UNet-LoRA (prompt-condicionado igualmente)
    prompt_dropout_p  = 0.1

cfg = Cfg()

base = "sdxl" if "xl" in cfg.model_id.lower() else "sd15"
cfg.out_dir = f"./{base}-finetuned"
# Forzar nombre de salida
cfg.out_dir = "./sdxl-finetuned"

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
# Dataset
# -------------------------
class FolderPromptDataset(Dataset):
    def __init__(self, root, size):
        self.samples = []
        self.size    = size
        if not os.path.isdir(root):
            raise RuntimeError(f"train_dir not found: {root}")
        for prompt_folder in sorted(os.listdir(root)):
            pf = os.path.join(root, prompt_folder)
            if not os.path.isdir(pf):
                continue
            default_prompt = prompt_folder.replace("_", " ").strip()
            for fn in sorted(os.listdir(pf)):
                lo = fn.lower()
                if lo.endswith((".png", ".jpg", ".jpeg", ".webp")):
                    img_path = os.path.join(pf, fn)
                    cap_path = os.path.splitext(img_path)[0] + ".txt"
                    prompt_text = default_prompt
                    if os.path.isfile(cap_path):
                        try:
                            with open(cap_path, "r", encoding="utf-8") as f:
                                cap = f.read().strip()
                            if cap:
                                prompt_text = cap
                        except Exception:
                            pass
                    self.samples.append((img_path, prompt_text))
        if not self.samples:
            raise RuntimeError(f"No images found under {root}")
        print(f"Found {len(self.samples)} images across prompts.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, prompt = self.samples[idx]
        img = Image.open(img_path).convert("RGB").resize((self.size, self.size))
        import numpy as np
        arr = torch.from_numpy(np.array(img).astype("float32") / 255.0)
        px = arr.permute(2, 0, 1) * 2 - 1
        return {"pixel_values": px, "prompt": prompt}

def collate_fn(batch):
    px = torch.stack([b["pixel_values"] for b in batch], dim=0)
    prompts = [b["prompt"] for b in batch]
    return {"pixel_values": px, "prompt": prompts}

dataset = FolderPromptDataset(cfg.train_dir, cfg.image_size)
loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)

# -------------------------
# Pipeline (SD 1.5 o SDXL)
# -------------------------
is_sdxl = "xl" in cfg.model_id.lower()
pipe_cls = StableDiffusionXLPipeline if is_sdxl else StableDiffusionPipeline
print("Loading pipeline...")
pipe = pipe_cls.from_pretrained(
    cfg.model_id,
    torch_dtype=dtype if device == "cuda" else torch.float32,
    safety_checker=None, requires_safety_checker=False,
    use_safetensors=True,
)

pipe.scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", prediction_type="epsilon")

pipe.unet.to(device, dtype=dtype).train()
pipe.unet.enable_gradient_checkpointing()

pipe.vae.to("cpu", dtype=torch.float32)
for p in pipe.vae.parameters(): p.requires_grad_(False)

if is_sdxl:
    pipe.text_encoder.to("cpu", dtype=torch.float32)
    pipe.text_encoder_2.to("cpu", dtype=torch.float32)
    for p in pipe.text_encoder.parameters():   p.requires_grad_(False)
    for p in pipe.text_encoder_2.parameters(): p.requires_grad_(False)
else:
    pipe.text_encoder.to("cpu", dtype=torch.float32)
    for p in pipe.text_encoder.parameters():   p.requires_grad_(False)

# -------------------------
# LoRA (PEFT) en el UNet
# -------------------------
# Objetivo: capas lineales de atención: to_q, to_k, to_v, to_out.0
# Añadimos también proj_in/proj_out para SDXL (beneficia algo la adaptación).
target_modules = ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"]

lora_cfg = LoraConfig(
    r=cfg.lora_rank,
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
    bias="none",
    target_modules=target_modules,
    task_type="UNSPECIFIED",   # genérico (no transformers)
)

pipe.unet = get_peft_model(pipe.unet, lora_cfg)
pipe.unet.print_trainable_parameters()

# Solo los params LoRA tienen requires_grad=True tras get_peft_model
trainable_params = [p for p in pipe.unet.parameters() if p.requires_grad]
assert trainable_params, "No LoRA trainable params found via PEFT"
optim = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)

# -------------------------
# Utils
# -------------------------
def compute_sdxl_time_ids(h, w, b, dtype_, dev_):
    t = torch.tensor([h, w, 0, 0, h, w], dtype=dtype_, device=dev_)
    return t.unsqueeze(0).repeat(b, 1)

random.seed(cfg.seed); torch.manual_seed(cfg.seed)
g_cpu = torch.Generator("cpu").manual_seed(cfg.seed)

os.makedirs(cfg.out_dir, exist_ok=True)
log_path = os.path.join(cfg.out_dir, "train_log.jsonl")
log_f = open(log_path, "a", encoding="utf-8")

global_step = 0
max_steps = cfg.steps_per_epoch if cfg.steps_per_epoch is not None else math.ceil(len(loader)/cfg.batch_size)
print(f"Epochs={cfg.epochs} | steps/epoch≈{max_steps} | saving to {cfg.out_dir}")

# -------------------------
# Entrenamiento
# -------------------------
for epoch in range(cfg.epochs):
    pbar = tqdm(loader, total=max_steps, desc=f"Epoch {epoch+1}/{cfg.epochs}")
    step_in_epoch = 0
    for batch in pbar:
        if cfg.steps_per_epoch is not None and step_in_epoch >= cfg.steps_per_epoch:
            break

        pixel_values = batch["pixel_values"]
        prompts_in   = batch["prompt"]
        prompts = [(p if random.random() > cfg.prompt_dropout_p else "") for p in prompts_in]
        B, C, H, W = pixel_values.shape

        with torch.no_grad():
            pv = pixel_values.to("cpu", dtype=torch.float32)
            latents_dist = pipe.vae.encode(pv).latent_dist
            latents = (latents_dist.sample() * cfg.vae_scale).to(device, dtype=dtype)

            noise = torch.randn_like(latents, device=device, dtype=dtype)
            t = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (B,), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

        if is_sdxl:
            pe, _, pooled, _ = pipe.encode_prompt(
                prompt=prompts, device="cpu",
                num_images_per_prompt=1, do_classifier_free_guidance=False,
            )
            prompt_embeds      = pe.to(device, dtype=dtype)
            pooled_prompt_embs = pooled.to(device, dtype=dtype)
            add_time_ids       = compute_sdxl_time_ids(H, W, B, prompt_embeds.dtype, device)
        else:
            tok = pipe.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
            input_ids = tok.input_ids.to("cpu")
            attn_mask = tok.attention_mask.to("cpu") if hasattr(tok, "attention_mask") else None
            with torch.no_grad():
                te_out = pipe.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
            prompt_embeds = te_out[0].to(device, dtype=dtype)

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
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
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
            # Guarda SOLO los adapters LoRA (PEFT)
            pipe.unet.save_pretrained(os.path.join(ckpt_dir, "lora_unet_peft"))
            print(f"Saved LoRA adapters (PEFT) to {ckpt_dir}")

# -------------------------
# Guardado final (adapters PEFT)
# -------------------------
final_dir = os.path.join(cfg.out_dir, "final")
os.makedirs(final_dir, exist_ok=True)
pipe.unet.save_pretrained(os.path.join(final_dir, "lora_unet_peft"))
log_f.close()
print(f"Done. LoRA adapters (PEFT) saved to {final_dir}\nLogs → {log_path}")
