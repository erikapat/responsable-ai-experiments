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
prompt = "a neutral portrait photo"
resolution = 512
batch_size = 1
epochs = 1
lr = 1e-5
vae_scale = 0.18215

use_mps = torch.backends.mps.is_available()
use_cuda = torch.cuda.is_available()
unet_device = "mps" if use_mps else ("cuda" if use_cuda else "cpu")
# Use fp32 on MPS; fp16 on CUDA; fall back to fp32 on CPU.
unet_dtype = torch.float32 if unet_device == "mps" else (torch.float16 if unet_device == "cuda" else torch.float32)

print(f"UNet device: {unet_device} | UNet dtype: {unet_dtype}")

# -------------------------
# Dataset
# -------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, folder, prompt, size, processor):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        if len(self.files) == 0:
            raise ValueError(f"No images found in {folder}")
        self.prompt = prompt
        self.size = size
        self.processor = processor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB").resize((self.size, self.size))
        out = self.processor.preprocess(img)  # -> dict with "pixel_values": [1,C,H,W]
        pixel_values = out["pixel_values"] if isinstance(out, dict) else out
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.from_numpy(pixel_values)
        if pixel_values.ndim == 4 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0)  # -> [C,H,W]
        return {"pixel_values": pixel_values, "prompt": self.prompt}

# -------------------------
# Pipeline + placements
# -------------------------
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
print("Loading pipeline components...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id, torch_dtype=unet_dtype if unet_device == "cuda" else torch.float32, use_safetensors=True
)

# Use a training scheduler (not the default inference scheduler)
pipe.scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="scaled_linear",
    prediction_type="epsilon",
)

# Place modules: keep memory-heavy UNet on GPU; keep VAE and text encoders on CPU.
pipe.unet.to(unet_device, dtype=unet_dtype)
pipe.unet.train()
pipe.unet.enable_gradient_checkpointing()

pipe.vae.to("cpu", dtype=torch.float32)
pipe.text_encoder.to("cpu", dtype=torch.float32)
pipe.text_encoder_2.to("cpu", dtype=torch.float32)

# Helpful for CPU VAE memory patterns (mostly helps on decode; harmless here)
try:
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
except Exception:
    pass

# Freeze non-UNet modules
for p in pipe.vae.parameters(): p.requires_grad_(False)
for p in pipe.text_encoder.parameters(): p.requires_grad_(False)
for p in pipe.text_encoder_2.parameters(): p.requires_grad_(False)

dataset = ImageFolderDataset(train_dir, prompt, resolution, pipe.image_processor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

optimizer = optim.Adam(pipe.unet.parameters(), lr=lr)

# -------------------------
# Helpers
# -------------------------
def make_time_ids(h, w, batch, dtype, device):
    # SDXL expects [orig_h, orig_w, crop_y, crop_x, target_h, target_w]
    t = torch.tensor([h, w, 0, 0, h, w], dtype=dtype, device=device)
    return t.unsqueeze(0).repeat(batch, 1)

# -------------------------
# Training loop
# -------------------------
for epoch in range(epochs):
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for step, batch in enumerate(loop):
        pixel_values = batch["pixel_values"]  # CPU tensor [B,C,H,W]
        if pixel_values.ndim == 5 and pixel_values.shape[1] == 1:
            pixel_values = pixel_values.squeeze(1)
        B, C, H, W = pixel_values.shape
        assert pixel_values.ndim == 4, f"Expected [B,C,H,W], got {pixel_values.shape}"

        # ----------- Encode to latents on CPU -----------
        with torch.no_grad():
            pixel_values_cpu = pixel_values.to("cpu", dtype=torch.float32, non_blocking=False)
            latents_dist = pipe.vae.encode(pixel_values_cpu).latent_dist
            latents_cpu = latents_dist.sample() * vae_scale  # [B,4,h/8,w/8]

        # Move latents to UNet device/dtype
        latents = latents_cpu.to(unet_device, dtype=unet_dtype, non_blocking=False)

        # ----------- Sample noise & timesteps on UNet device -----------
        noise = torch.randn_like(latents, device=unet_device, dtype=unet_dtype)
        timesteps = torch.randint(
            0, pipe.scheduler.config.num_train_timesteps, (B,), device=unet_device
        ).long()
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # ----------- Prompt encoding on CPU, then move embeddings -----------
        with torch.no_grad():
            prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt,
                device="cpu",  # run encoders on CPU to save VRAM
                num_images_per_prompt=B,
                do_classifier_free_guidance=False,
            )
            # Now move only the small embeddings to the UNet device/dtype
            prompt_embeds = prompt_embeds.to(unet_device, dtype=unet_dtype, non_blocking=False)
            pooled_prompt_embeds = pooled_prompt_embeds.to(unet_device, dtype=unet_dtype, non_blocking=False)

            # SDXL time ids on UNet device
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

        # ----------

# -------------------------
# Save finetuned pipeline
# -------------------------
output_dir = "sdxl-finetuned"   # or any path you prefer
os.makedirs(output_dir, exist_ok=True)

pipe.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")



