# postprocess_pipeline.py
import torch, pathlib, random, gc
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageStat

# -------------------------------
# Device setup
# -------------------------------
def best_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = best_device()
DTYPE = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32

# -------------------------------
# Safety checker dummy
# -------------------------------
def dummy_safety(images, clip_input):
    return images, [False] * len(images)

# -------------------------------
# Filters
# -------------------------------

def quality_filter(img: Image.Image, blur_threshold=100.0) -> bool:
    """
    Rejects images that are too blurry/low variance.
    """
    stat = ImageStat.Stat(img.convert("L"))
    variance = stat.var[0]
    return variance > blur_threshold  # True = keep

def safety_filter(img: Image.Image) -> bool:
    """
    Placeholder for NSFW/safety classifier.
    For now, always returns True (safe).
    Replace with your own classifier if needed.
    """
    return True

def postprocess_filters(img: Image.Image) -> bool:
    """
    Combine all filters. Returns True if image passes all checks.
    """
    return quality_filter(img) and safety_filter(img)

# -------------------------------
# Generation pipeline
# -------------------------------
def make_pipeline(model_id="runwayml/stable-diffusion-v1-5"):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE)
    pipe = pipe.to(DEVICE)
    pipe.safety_checker = dummy_safety
    try: pipe.enable_vae_tiling()
    except: pass
    try: pipe.enable_attention_slicing("max")
    except: pass
    return pipe

def generate_images(pipe, prompts, outdir="outputs", steps=20, guidance=7.5, n_images=3):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for prompt in prompts:
        prompt_dir = outdir / prompt.lower().replace(" ", "_")
        prompt_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, n_images + 1):
            seed = random.randint(0, 2**32 - 1)
            gen = torch.Generator(device=DEVICE).manual_seed(seed)

            try:
                img = pipe(
                    prompt=prompt,
                    height=512, width=512,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=gen,
                ).images[0]
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM for {prompt}, skipping.")
                    continue
                else:
                    raise

            # Postprocess filters
            if postprocess_filters(img):
                fn = prompt_dir / f"{i:02d}_seed{seed}.png"
                img.save(fn)
                print(f"[OK] Saved {fn}")
            else:
                print(f"[REJECTED] {prompt} seed={seed}")

            del img
            if DEVICE == "mps":
                try: torch.mps.empty_cache()
                except: pass
            gc.collect()

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    PROMPTS = [
        "A portrait photo of a productive person",
        "A classroom of students learning AI",
        "Toys on the floor of a kindergarten",
    ]

    pipe = make_pipeline()
    generate_images(pipe, PROMPTS, outdir="postprocess_outputs", steps=25, guidance=7.5, n_images=5)

    print("Done.")
