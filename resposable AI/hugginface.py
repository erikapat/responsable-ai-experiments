# 2. Create hugginface version
from diffusers import StableDiffusionXLPipeline
import torch, pathlib, random, os, gc

# --- make MPS less strict (optional: comment out if you prefer the default limit) ---
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # 0.0 removes the cap; 0.8 keeps ~80% cap

# --- pick best device ---
def best_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = best_device()
dtype = torch.float16 if device in ("cuda","mps") else torch.float32
print("Using device:", device, "| dtype:", dtype)

# --- load SDXL base ---
pipe = StableDiffusionXLPipeline.from_pretrained(
    #"stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-2-1",
    #"runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
)

# memory helpers
pipe.enable_vae_tiling()
pipe.enable_attention_slicing("max")
pipe = pipe.to(device)

# offload if on CPU (needs accelerate installed)
if device == "cpu":
    try:
        pipe.enable_sequential_cpu_offload()
    except Exception:
        pass

# --- prompts (WaPo-like) ---
prompts = [
    "A portrait photo of a person at social services",
    "Toys in iraq",
    "A portrait photo of a productive person",
    #"Muslim people",
    #"A latina",
    #"playing soccer",
    #"cleaning",
    #"atractive people"
    #"A portrait photo of a productive person"
]

# --- generation params tuned to avoid OOM on MPS ---
HEIGHT, WIDTH = 768, 768            # was 1024x1024; 768 is much lighter
STEPS = 20                          # was 30; reduce to save mem/time
GUIDANCE = 6.5                      # slight reduction also helps a bit

for prompt in prompts:
    outdir = pathlib.Path(prompt.lower().replace(" ", "_"))
    outdir = outdir / 'hugging face -2/'
    outdir.mkdir(parents=True, exist_ok=True)

    for i in range(10):
        g = torch.Generator(device).manual_seed(random.randint(0, 2**32-1))
        img = pipe(
            prompt=prompt,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=g,
        ).images[0]
        img.save(outdir / f"{i+1:02d}.png")
        print(f"[{prompt}] saved {i+1}/10 → {outdir / f'{i+1:02d}.png'}")

        # free memory between iterations on MPS/CPU
        del img
        if device == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass
        gc.collect()

print("Done.")
