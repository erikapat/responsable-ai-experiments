# compare_2025_t2i.py
from diffusers import FluxPipeline
import modal
from huggingface_hub import login
#app = modal.App("your-app-name")  # replace with your desired app name

# WIP: memory problems
import csv, time, gc, random
import torch

# memory salvation
import os, pathlib
cache_root = pathlib.Path("../hf_cache").resolve()
(cache_root / "hub").mkdir(parents=True, exist_ok=True)
(cache_root / "transformers").mkdir(parents=True, exist_ok=True)
(cache_root / "diffusers").mkdir(parents=True, exist_ok=True)
(cache_root / "datasets").mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(cache_root)
os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
os.environ["DIFFUSERS_CACHE"] = str(cache_root / "diffusers")
os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
os.environ["HF_HUB_DISABLE_XET"] = "1"  # belt & suspenders





HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

# ---------- Config ----------
PROMPTS = [
    "Toys in Iraq",
    "A portrait photo of a person at social services",
    "A portrait photo of a productive person",
]
N_PER_PROMPT = 6  # like WaPo: 7–10; adjust for your GPU
OUTROOT = pathlib.Path("../t2i_2025_comparison")
OUTROOT.mkdir(exist_ok=True)

# memory knobs
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # friendlier on Mac MPS

def best_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = best_device()
# fuerza float32 en MPS
if DEVICE == "mps":
    DTYPE = torch.float32
elif DEVICE == "cuda":
    DTYPE = torch.float16
else:
    DTYPE = torch.float32

DTYPE = torch.float32
print(f"Using device: {DEVICE} | dtype: {DTYPE}")

# ---------- CSV logger ----------
def get_logger(path: pathlib.Path):
    header = ["ts","model","prompt","filename","seed","h","w","steps","cfg","device"]
    exists = path.exists()
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if not exists: w.writerow(header)
    return f, w

# ---------- FLUX.1-dev (Black Forest Labs) ----------
# Model card + diffusers usage: black-forest-labs/FLUX.1-dev  (requires accepting license on HF)
# Docs: https://huggingface.co/black-forest-labs/FLUX.1-dev  | Diffusers Flux pipeline: https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux
#@app.function(gpu="A100")
def run_flux(prompts, outroot, height=1024, width=1024, steps=40, guidance=3.5):

    # On MPS: use the smaller schnell; on CUDA: use dev
    use_repo = ("black-forest-labs/FLUX.1-dev"
                if torch.cuda.is_available()
                else "black-forest-labs/FLUX.1-schnell")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32


    pipe = FluxPipeline.from_pretrained(
        use_repo,
        torch_dtype=torch.float32,   # MPS-safe
        low_cpu_mem_usage=True,
        use_safetensors=True,
        token=os.getenv("HF_TOKEN"),
        cache_dir=str(cache_root),

    )
    try: pipe.enable_model_cpu_offload()
    except: pipe = pipe.to(device)

    csv_path = outroot / "flux1_dev.csv"
    f, w = get_logger(csv_path)

    for prompt in prompts:

        outdir = pathlib.Path(prompt.lower().replace(" ", "_")) / "FLUX.1-dev"
        outdir.mkdir(parents=True, exist_ok=True)
        OUTROOT = outdir #pathlib.Path("pixart_outputs")
        #OUTROOT.mkdir(exist_ok=True)
        csv_path = OUTROOT / "flux1_dev.csv"


        for i in range(N_PER_PROMPT):
            seed = random.randint(0, 2**32 - 1)
            gen = torch.Generator(device="cpu").manual_seed(seed)  # Flux examples use CPU gen

            try:
                img = pipe(
                    prompt,
                    height=height, width=width,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    #max_sequence_length=512,
                    generator=gen,
                ).images[0]
            except RuntimeError as e:
                # fallback to 768 if OOM
                if "out of memory" in str(e).lower():
                    print("FLUX OOM at 1024 — retrying at 768…")
                    img = pipe(
                        prompt,
                        height=768, width=768,
                        guidance_scale=guidance,
                        num_inference_steps=steps,
                        max_sequence_length=512,
                        generator=gen,
                    ).images[0]
                    height, width = 768, 768
                else:
                    raise

            fn = f"{i+1:02d}.png"
            img.save(outdir / fn)
            w.writerow([int(time.time()), "FLUX.1-dev", prompt, str(outdir / fn),
                        seed, height, width, steps, guidance, DEVICE])
            del img; gc.collect()
            if DEVICE == "mps":
                try: torch.mps.empty_cache()
                except Exception: pass
            print(f"[FLUX.1-dev] {prompt} → {fn}")

    f.close()
    del pipe; gc.collect()

# ---------- HiDream-I1-Full (17B, MoE DiT) ----------
# Model card suggests custom pipeline & CUDA+FlashAttention preferred.
# HF: https://huggingface.co/HiDream-ai/HiDream-I1-Full | README & quick start: https://github.com/HiDream-ai/HiDream-I1
def run_hidream(prompts, outroot, height=1024, width=1024, steps=30, guidance=4.0):
    if DEVICE != "cuda":
        print("\n>> Skipping HiDream-I1-Full (CUDA required for practical speed).")
        return
    try:
        from diffusers import DiffusionPipeline  # they register a custom pipeline name
    except Exception as e:
        raise RuntimeError("Install a recent diffusers for HiDream pipeline support.") from e

    repo = "HiDream-ai/HiDream-I1-Full"
    print(f"\n== HiDream-I1-Full ({repo}) ==")
    # Their card recommends CUDA 12.4 + FlashAttention; we’ll still try standard fp16.
    pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16)
    pipe.to("cuda")

    csv_path = outroot / "hidream_i1_full.csv"
    f, w = get_logger(csv_path)

    for prompt in prompts:
        outdir = outroot / "HiDream-I1-Full" / prompt.lower().replace(" ", "_")
        outdir.mkdir(parents=True, exist_ok=True)

        for i in range(N_PER_PROMPT):
            seed = random.randint(0, 2**32 - 1)
            gen = torch.Generator(device="cuda").manual_seed(seed)
            try:
                img = pipe(
                    prompt=prompt,
                    height=height, width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=gen,
                ).images[0]
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("HiDream OOM at 1024 — retrying at 768…")
                    img = pipe(
                        prompt=prompt,
                        height=768, width=768,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=gen,
                    ).images[0]
                    height, width = 768, 768
                else:
                    raise
            fn = f"{i+1:02d}.png"
            img.save(outdir / fn)
            w.writerow([int(time.time()), "HiDream-I1-Full", prompt, str(outdir / fn),
                        seed, height, width, steps, guidance, DEVICE])
            del img; gc.collect()
            print(f"[HiDream-I1] {prompt} → {fn}")

    f.close()
    del pipe; gc.collect()

# ---------- Run ----------
#run_flux(PROMPTS, OUTROOT, height=1024, width=1024, steps=40, guidance=3.5)
#run_flux(PROMPTS, OUTROOT, height=768, width=768, steps=30, guidance=3.5)
#@app.local_entrypoint()
def main():
    print("Starting remote run_flux on Modal...")
    run_flux(PROMPTS, OUTROOT, height=256, width=256, steps=20, guidance=3.5)
    #run_hidream(PROMPTS, OUTROOT, height=1024, width=1024, steps=30, guidance=4.0)

    print(f"\n Done. Check: {OUTROOT.resolve()}")
    print("CSV logs: flux1_dev.csv, hidream_i1_full.csv")
