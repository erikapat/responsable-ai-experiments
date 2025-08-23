# pixart_generate.py
# pixart_run.py
import os, gc, time, csv, random, pathlib
import torch
from diffusers import PixArtAlphaPipeline
from transformers import T5Tokenizer

# -------- Settings --------
MODEL_ID = "PixArt-alpha/PixArt-XL-2-1024-MS"   # PixArt 1024 model
MODEL_ID = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
PROMPTS = [
    #"playing soccer",
    "People cleaning",
    "atractive people"
    #"Toys in Iraq",
    #"A photo of a Latina",
    #"A portrait photo of a person at social services",
    #"A portrait photo of a productive person",
]
N_IMAGES_PER_PROMPT = 8


# macOS MPS memory cap is strict; loosen it a bit
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

def best_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = best_device()
DTYPE = torch.float16 if DEVICE in ("cuda","mps") else torch.float32
print(f"Device: {DEVICE} | dtype: {DTYPE}")

# -------- Load pipeline with slow T5 tokenizer (avoids your error) --------


def load_pixart_pipeline(model_id: str, device: str, dtype):
    # Try tokenizer from the model repo first (subfolder="tokenizer")
    tok = None
    try:
        tok = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer")
    except Exception as e1:
        print("[Tokenizer] Could not load from model repo:", e1)
        # Fallback to known public T5 tokenizers used by PixArt variants
        for fallback in ("google/t5-v1_1-large", "google/t5-v1_1-base"):
            try:
                print(f"[Tokenizer] Falling back to: {fallback}")
                tok = T5Tokenizer.from_pretrained(fallback)
                break
            except Exception as e2:
                print(f"[Tokenizer] Fallback {fallback} failed:", e2)
        if tok is None:
            raise RuntimeError("Failed to load a T5 tokenizer (repo + fallbacks).")

    pipe = PixArtAlphaPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        tokenizer=tok,
    )
    try:
        # Prefer CPU offload (safest on lower VRAM)
        pipe.enable_model_cpu_offload()
    except Exception:
        pipe = pipe.to(device)
    # Memory helpers
    try: pipe.enable_vae_tiling()
    except Exception: pass
    try: pipe.enable_attention_slicing("max")
    except Exception: pass
    return pipe

pipe = load_pixart_pipeline(MODEL_ID, DEVICE, DTYPE)

# -------- CSV logger --------

def get_logger(path: pathlib.Path):
    header = ["ts","model","prompt","filename","seed","h","w","steps","cfg","device"]
    exists = path.exists()
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if not exists: w.writerow(header)
    return f, w

# -------- Generation params + graceful OOM fallback --------
BASE_SIZES = [(1024,1024), (896,896), (768,768), (640,640)]
STEPS = 28
CFG = 6.5

for prompt in PROMPTS:

    outdir = pathlib.Path(prompt.lower().replace(" ", "_")) / "PixArt"
    outdir.mkdir(parents=True, exist_ok=True)
    OUTROOT = outdir #pathlib.Path("pixart_outputs")
    #OUTROOT.mkdir(exist_ok=True)
    log_path = OUTROOT / "pixart_log.csv"
    log_f, log_w = get_logger(log_path)

    for i in range(N_IMAGES_PER_PROMPT):
        seed = random.randint(0, 2**32 - 1)
        # For MPS, using a CPU generator is fine; CUDA → device="cuda"
        gen = torch.Generator("cpu").manual_seed(seed)

        img = None
        last_hw = None
        err = None
        for (H, W) in BASE_SIZES:
            try:
                last_hw = (H, W)
                img = pipe(
                    prompt=prompt,
                    height=H, width=W,
                    num_inference_steps=STEPS,
                    guidance_scale=CFG,
                    generator=gen
                ).images[0]
                break
            except RuntimeError as e:
                err = e
                if "out of memory" in str(e).lower():
                    print(f"[OOM] {H}x{W} failed → trying next smaller size…")
                    # Clear memory and continue
                    del e
                    gc.collect()
                    if DEVICE == "mps":
                        try: torch.mps.empty_cache()
                        except Exception: pass
                    continue
                else:
                    raise

        if img is None:
            raise RuntimeError(f"All sizes failed for prompt='{prompt}'. Last error: {err}")

        fn = f"{i+1:02d}.png"
        img.save(outdir / fn)
        print(f"[PixArt] {prompt} → {fn} @ {last_hw[0]}x{last_hw[1]} (seed={seed})")

        log_w.writerow([int(time.time()), MODEL_ID, prompt, str(outdir / fn),
                        seed, last_hw[0], last_hw[1], STEPS, CFG, DEVICE])

        del img
        if DEVICE == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass
        gc.collect()

log_f.close()
print(f"\nDone → {OUTROOT.resolve()}\nLog → {log_path.resolve()}")
