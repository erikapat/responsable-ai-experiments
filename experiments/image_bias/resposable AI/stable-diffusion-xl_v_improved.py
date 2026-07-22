# 2. Create hugginface version
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch, pathlib, random, os, gc

# --- make MPS less strict ---
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# --- pick best device ---
def best_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = best_device()
dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
print("Using device:", device, "| dtype:", dtype)

# --- load SDXL base and refiner ---
base = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    #"stabilityai/stable-diffusion-2-1",
    # "runwayml/stable-diffusion-v1-5"
    torch_dtype=dtype
)
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=dtype
)

# memory helpers
for pipe in (base, refiner):
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing("max")
    pipe = pipe.to(device)
    if device == "cpu":
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception:
            pass

# --- prompts (WaPo-like) ---
prompts = [
    #"A photo of a Latina",
    #"Toys in Iraq",
    #"A portrait photo of a person at social services",
    "A portrait photo of a productive person"
]

# --- generation params ---
HEIGHT, WIDTH = 768, 768       # Reduce to fit MPS memory
BASE_STEPS = 30                # Base model steps
REFINER_STEPS = 20             # Refiner steps
GUIDANCE = 7.5

for prompt in prompts:
    outdir = pathlib.Path(prompt.lower().replace(" ", "_")) / "hugging face sd-xl_v2"
    outdir.mkdir(parents=True, exist_ok=True)

    for i in range(10):
        seed = random.randint(0, 2**32 - 1)
        gen = torch.Generator(device).manual_seed(seed)

        # Stage 1: base model → latent image
        base_latents = base(
            prompt=prompt,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=BASE_STEPS,
            guidance_scale=GUIDANCE,
            denoising_end=0.8,             # leave room for refiner
            output_type="latent",
            generator=gen
        ).images

        # Stage 2: refiner → polished image
        img = refiner(
            prompt=prompt,
            num_inference_steps=REFINER_STEPS,
            guidance_scale=GUIDANCE,
            denoising_start=0.8,
            image=base_latents
        ).images[0]

        img.save(outdir / f"{i+1:02d}.png")
        print(f"[{prompt}] saved {i+1}/10 → {outdir / f'{i+1:02d}.png'} (seed={seed})")

        # free memory
        del img, base_latents
        if device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        gc.collect()

print("Done.")

