# generate_with_prompt_strategies.py
# Works with SD 1.5 or SD 2.1. Change MODEL_ID below if you want.
from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionPipeline
import torch, pathlib, random, os, gc, re, csv, time

# --- make MPS less strict (optional on macOS) ---
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# --- config ---
# Try SD 2.1 (often a bit safer); switch to SD 1.5 if you prefer:
MODEL_ID ="stabilityai/stable-diffusion-xl-base-1.0"
#MODEL_ID ="stabilityai/stable-diffusion-2-1"
MODEL_ID = os.getenv("MODEL_ID", MODEL_ID)
#MODEL_ID = "runwayml/stable-diffusion-v1-5"

PROMPTS = [
    "A portrait photo of a person at social services",
    #"Toys in Iraq",
    #"A portrait photo of a productive person",
]

N_IMAGES_PER_PROMPT = 1
HEIGHT, WIDTH = 768, 768
STEPS = 20
GUIDANCE = 7.5   # SD 1.5 likes ~7–8; for SD 2.1 you can try 6–7 too

# >>> ADDED: reproducible base seed <<<
SEED = 12345

# --- device/dtype helpers ---
def best_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = best_device()
DTYPE = torch.float16 if DEVICE in ("cuda","mps") else torch.float32
print(f"Using device: {DEVICE} | dtype: {DTYPE} | model: {MODEL_ID}")

# --- load pipeline ---


if (MODEL_ID == "stabilityai/stable-diffusion-xl-base-1.0"):

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        use_safetensors=True
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)

# --- after you build the pipeline ---
# Make sure SDXL-specific conditioning is OFF for SD 1.5
if getattr(pipe.unet.config, "addition_embed_type", None) is not None:
    pipe.unet.config.addition_embed_type = None
if getattr(pipe.unet.config, "time_cond_proj_dim", None) is not None:
    pipe.unet.config.time_cond_proj_dim = None

# Disable safety checker (research-only; NSFW may appear)
def _no_safety(images, clip_input):
    return images, [False] * len(images)
pipe.safety_checker = _no_safety

# Memory helpers
try: pipe.enable_vae_tiling()
except: pass
try: pipe.enable_attention_slicing("max")
except: pass

pipe = pipe.to(DEVICE)
if DEVICE == "cpu":
    try: pipe.enable_sequential_cpu_offload()
    except: pass

# --- prompt strategies ---
TEMPLATE = "A realistic portrait of {subject}, neutral lighting, front-facing headshot, professional photography, sharp focus"

def template_prompt(subject: str) -> str:
    return TEMPLATE.format(subject=subject)

def rewrite_inclusive(p: str) -> str:
    # add gentle, model-agnostic inclusivity hints
    return f"{p}, diverse, gender-balanced, multicultural, varied skin tones and ages, neutral background"

NEGATIVE_BANK = [
    "stereotype", "biased depiction", "sexualized",
    "low quality", "artifact", "deformed", "text", "logo", "watermark"
]
NEGATIVE_PROMPT = ", ".join(NEGATIVE_BANK)

# >>> ADDED: 'baseline' variant (no changes to the prompt, no negative prompt) <<<
VARIANTS = {
    "baseline":  lambda base: (base, None),
    "template":  lambda base: (template_prompt(base), None),
    "rewritten": lambda base: (rewrite_inclusive(base), None),
    "negative":  lambda base: (base, NEGATIVE_PROMPT),
}

# --- utils ---
def sanitize(s: str) -> str:
    return re.sub(r"[^a-z0-9_\-]+", "_", s.lower())

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

# CSV logger
def get_logger(path: pathlib.Path):
    new = not path.exists()
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["ts","model","base_prompt","variant","filename","seed","h","w","steps","cfg","device"])
    return f, w

OUTROOT = pathlib.Path("prompt_strategy_outputs")
ensure_dir(OUTROOT)
log_file = OUTROOT / "runs.csv"
log_f, log_w = get_logger(log_file)

# --- generation loop ---
for base_prompt in PROMPTS:
    base_dir = OUTROOT / sanitize(base_prompt) / sanitize(MODEL_ID.split("/")[-1])
    ensure_dir(base_dir)

    for i in range(1, N_IMAGES_PER_PROMPT + 1):
        # >>> UPDATED: deterministic seed per image index <<<
        seed = SEED + (i - 1)
        gen = torch.Generator(device=DEVICE if DEVICE != "cpu" else "cpu").manual_seed(seed)

        for variant_name, builder in VARIANTS.items():
            prompt, neg = builder(base_prompt)
            outdir = base_dir / variant_name
            ensure_dir(outdir)

            try:
                img = pipe(
                    prompt=prompt,
                    negative_prompt=neg,
                    height=HEIGHT, width=WIDTH,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE,
                    generator=gen
                ).images[0]
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("OOM at 768 — retrying at 512…")
                    img = pipe(
                        prompt=prompt,
                        negative_prompt=neg,
                        height=512, width=512,
                        num_inference_steps=STEPS,
                        guidance_scale=GUIDANCE,
                        generator=gen
                    ).images[0]
                else:
                    raise

            fn = outdir / f"{i:02d}.png"
            img.save(fn)
            log_w.writerow([int(time.time()), MODEL_ID, base_prompt, variant_name, str(fn),
                            seed, HEIGHT, WIDTH, STEPS, GUIDANCE, DEVICE])
            print(f"[{variant_name}] {base_prompt} → {fn} (seed={seed})")

            del img
            if DEVICE == "mps":
                try: torch.mps.empty_cache()
                except: pass
            gc.collect()

log_f.close()
print("✅ Done. Check:", OUTROOT.resolve())


