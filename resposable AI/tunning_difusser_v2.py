import os
# ---- Reduce MPS OOM early (must be set before Torch loads MPS tensors) ----
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from PIL import Image
from torch import optim
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
train_dir = "./my_faces/train"
resolution = 512
batch_size = 1
epochs = 1
lr = 1e-5
vae_scale = 0.18215

use_mps = torch.backends.mps.is_available()
use_cuda = torch.cuda.is_available()
unet_device = "mps" if use_mps else ("cuda" if use_cuda else "cpu")
unet_dtype = torch.float32 if unet_device == "mps" else (torch.float16 if unet_device == "cuda" else torch.float32)

print(f"UNet device: {unet_device} | UNet dtype: {unet_dtype}")

# -------------------------
# Dataset (multi-prompt)
# -------------------------
class FolderPromptDataset(Dataset):
    def __init__(self, root_dir, size, processor):
        self.samples = []
        self.size = size
        self.processor = processor

        for prompt_folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, prompt_folder)
            if not os.path.isdir(folder_path):
                continue

            prompt_text = prompt_folder.replace("_", " ")

            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    img_path = os.path.join(folder_path, fname)
                    self.samples.append((img_path, prompt_text))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, prompt = self.samples[idx]
        img = Image.open(img_path).convert("RGB").resize((self.size, self.size))
        out = self.processor.preprocess(img)

        # handle dict vs tensor
        if isinstance(out, dict):
            pixel_values = out["pixel_values"]
        else:
            pixel_values = out

        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.as_tensor(pixel_values)

        # squeeze extra batch dim if needed
        if pixel_values.ndim == 4 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0)  # (C, H, W)

        return {"pixel_values": pixel_values, "prompt": prompt}

# --- custom collate ---
def collate_batch(samples):
    pixel_values = torch.stack([s["pixel_values"] for s in samples], dim=0)  # (B,C,H,W)
    prompts = [s["prompt"] for s in samples]
    return {"pixel_values": pixel_values, "prompt": prompts}

# -------------------------
# Pipeline + placements
# -------------------------
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
print("Loading pipeline components...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id, torch_dtype=unet_dtype if unet_device == "cuda" else torch.float32, use_safetensors=True
)

pipe.scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="scaled_linear",
    prediction_type="epsilon",
)

pipe.unet.to(unet_device, dtype=unet_dtype)
pipe.unet.train()
pipe.unet.enable_gradient_checkpointing()

pipe.vae.to("cpu", dtype=torch.float32)
pipe.text_encoder.to("cpu", dtype=torch.float32)
pipe.text_encoder_2.to("cpu", dtype=torch.float32)

try:
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
except Exception:
    pass

for p in pipe.vae.parameters(): p.requires_grad_(False)
for p in pipe.text_encoder.parameters(): p.requires_grad_(False)
for p in pipe.text_encoder_2.parameters(): p.requires_grad_(False)

dataset = FolderPromptDataset(train_dir, resolution, pipe.image_processor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate_batch)

optimizer = optim.Adam(pipe.unet.parameters(), lr=lr)

# -------------------------
# Helpers
# -------------------------
def make_time_ids(h, w, batch, dtype, device):
    t = torch.tensor([h, w, 0, 0, h, w], dtype=dtype, device=device)
    return t.unsqueeze(0).repeat(batch, 1)

# -------------------------
# Training loop
# -------------------------
for epoch in range(epochs):
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for step, batch in enumerate(loop):
        pixel_values = batch["pixel_values"]  # (B,C,H,W)
        prompts = batch["prompt"]

        B, C, H, W = pixel_values.shape
        assert pixel_values.ndim == 4, f"Expected [B,C,H,W], got {pixel_values.shape}"

        with torch.no_grad():
            pixel_values_cpu = pixel_values.to("cpu", dtype=torch.float32)
            latents_dist = pipe.vae.encode(pixel_values_cpu).latent_dist
            latents_cpu = latents_dist.sample() * vae_scale

        latents = latents_cpu.to(unet_device, dtype=unet_dtype)

        noise = torch.randn_like(latents, device=unet_device, dtype=unet_dtype)
        timesteps = torch.randint(
            0, pipe.scheduler.config.num_train_timesteps, (B,), device=unet_device
        ).long()
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # --- prompt encoding dinámico ---
        with torch.no_grad():
            prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompts,
                device="cpu",
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            prompt_embeds = prompt_embeds.to(unet_device, dtype=unet_dtype)
            pooled_prompt_embeds = pooled_prompt_embeds.to(unet_device, dtype=unet_dtype)

            try:
                add_time_ids = pipe._get_add_time_ids(
                    original_size=(H, W),
                    crops_coords_top_left=(0, 0),
                    target_size=(H, W),
                    dtype=prompt_embeds.dtype,
                    device=unet_device,
                    batch_size=B,
                )
            except Exception:
                add_time_ids = make_time_ids(H, W, B, dtype=prompt_embeds.dtype, device=unet_device)

# -------------------------
# Save finetuned pipeline
# -------------------------
output_dir = "sdxl-finetuned"
os.makedirs(output_dir, exist_ok=True)
pipe.save_pretrained(output_dir)
print(f"✅ Model saved to {output_dir}")
