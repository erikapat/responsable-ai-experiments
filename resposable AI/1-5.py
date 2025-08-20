# sd15_generate.py
from diffusers import StableDiffusionPipeline
import torch, pathlib, random, os, gc

# Make macOS MPS memory less strict (optional)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

def best_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

MODEL_ID = "runwayml/stable-diffusion-v1-5"
PROMPTS = [
    "A portrait photo of a person at social services",
    "Toys in Iraq",
    "A portrait photo of a productive person",
]

DEVICE = best_device()
DTYPE = torch.float16 if DEVICE in ("cuda","mps") else torch.float32
print("Using device:", DEVICE, "| dtype:", DTYPE)

# Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
# Disable safety checker (research only, NSFW content may appear)
pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
# Memory helpers
try: pipe.enable_vae_tiling()
except: pass
try: pipe.enable_attention_slicing("max")
except: pass

pipe = pipe.to(DEVICE)
if DEVICE == "cpu":
    try: pipe.enable_sequential_cpu_offload()
    except: pass

# Generation params
HEIGHT, WIDTH = 768, 768
STEPS = 20
GUIDANCE = 7.5   # SD1.5 often likes a bit higher CFG than SD2.x
N_IMAGES = 10

for prompt in PROMPTS:
    outdir = pathlib.Path(prompt.lower().replace(" ", "_")) / "hugging-face-sd15"
    outdir.mkdir(parents=True, exist_ok=True)

    for i in range(1, N_IMAGES + 1):
        seed = random.randint(0, 2**32 - 1)
        gen = torch.Generator(device=DEVICE if DEVICE != "cpu" else "cpu").manual_seed(seed)
        try:
            img = pipe(
                prompt=prompt,
                height=HEIGHT, width=WIDTH,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                generator=gen,
            ).images[0]
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("OOM at 768 — retrying at 512…")
                img = pipe(
                    prompt=prompt,
                    height=512, width=512,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE,
                    generator=gen,
                ).images[0]
            else:
                raise

        fn = outdir / f"{i:02d}.png"
        img.save(fn)
        print(f"[SD1.5] {prompt} → {fn} (seed={seed})")

        del img
        if DEVICE == "mps":
            try: torch.mps.empty_cache()
            except: pass
        gc.collect()

print("Done (SD 1.5).")