#!/usr/bin/env python3
import os
from accelerate import Accelerator
# Prevent early MPS OOM (must be set before torch import)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import glob
import argparse
import torch
from diffusers import StableDiffusionXLPipeline
from pathlib import Path
from PIL import Image
import time, re

# Initialize accelerator
accelerator = Accelerator()

# >>> ADDED: optional in-code multi-prompt list [(text, count), ...]
PROMPTS = [
    ("A portrait photo of a person at social services", 2),
    ("Toys in Iraq", 3),
    ("A productive person", 1),
]

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9_\-]+", "_", s.lower())

def save_image(img, base_dir: Path, prefix="fine_tuned_result", ext="png") -> Path:
    out_dir = Path(base_dir) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_{int(time.time())}.{ext}"

    if isinstance(img, Image.Image):
        img.save(path)
    elif isinstance(img, (bytes, bytearray)):
        path.write_bytes(img)
    elif isinstance(img, (str, Path)):
        raise TypeError(
            f"save_image(img=...) recibió {type(img)}. "
            "Pasa el objeto PIL.Image (ej. out.images[0]), no una ruta."
        )
    else:
        raise TypeError(f"Tipo no soportado para img: {type(img)}")
    return path

def parse_args():
    p = argparse.ArgumentParser(description="SDXL inference with optional finetuned dir or LoRA (MPS-safe).")
    p.add_argument("--prompt", default="A portrait photo of a person at social services")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--scale", type=float, default=6.5)
    p.add_argument("--seed", type=int, default=38)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--finetuned_dir", default=None)
    p.add_argument("--lora", default=None)
    p.add_argument("--base_model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--device", default=None, choices=["cpu","mps","cuda"])
    p.add_argument("--use_prompts", action="store_true")
    p.add_argument("--count", type=int, default=1)
    return p.parse_args()

def resolve_relative(base_dir, path_or_none):
    if not path_or_none:
        return None
    return path_or_none if os.path.isabs(path_or_none) else os.path.join(base_dir, path_or_none)

def find_lora_inside(dir_path):
    for pat in [
        "pytorch_lora_weights.safetensors",
        "lora.safetensors",
        "*.safetensors",
        "pytorch_lora_weights.bin",
        "lora.bin",
        "*.bin",
    ]:
        matches = sorted(glob.glob(os.path.join(dir_path, pat)))
        if matches:
            return matches[0]
    return None

def diagnose_paths(base_dir):
    print("\n--- Path diagnostics ---")
    print(f"Script base_dir: {base_dir}")
    print(f"CWD: {os.getcwd()}")
    try:
        for item in os.listdir(base_dir):
            print("  -", item)
    except Exception as e:
        print("  (failed to list base_dir)", e)
    print("------------------------\n")

def pick_device(forced=None):
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def main():
    args = parse_args()

    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    device = pick_device(args.device)
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Requested device={device}, dtype={torch_dtype}")

    finetuned_dir = resolve_relative(BASE_DIR, args.finetuned_dir)
    lora_path = resolve_relative(BASE_DIR, args.lora)

    if not finetuned_dir:
        maybe_ft = os.path.join(BASE_DIR, "sdxl-finetuned")
        if os.path.isdir(maybe_ft):
            finetuned_dir = maybe_ft
    if not lora_path:
        maybe_lora = os.path.join(BASE_DIR, "sdxl-lora-mps")
        if os.path.exists(maybe_lora):
            lora_path = maybe_lora

    if not (finetuned_dir and os.path.isdir(finetuned_dir)) and not (lora_path and os.path.exists(lora_path)):
        print("No finetuned model directory or LoRA file found. Exiting.")
        diagnose_paths(BASE_DIR)
        return

    if finetuned_dir and os.path.isdir(finetuned_dir):
        print(f"→ Loading finetuned pipeline from: {finetuned_dir}")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            finetuned_dir,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
    else:
        print(f"→ Loading base model: {args.base_model}")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.base_model,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        if lora_path:
            print(f"Looking for LoRA at: {lora_path}")
            if not os.path.exists(lora_path):
                diagnose_paths(BASE_DIR)
                raise FileNotFoundError(f"LoRA path not found: {lora_path}")
            if os.path.isdir(lora_path):
                file_inside = find_lora_inside(lora_path)
                if file_inside:
                    print(f"→ Loading LoRA file: {file_inside}")
                    pipe.load_lora_weights(file_inside, use_safetensors=file_inside.endswith(".safetensors"))
                else:
                    print("→ Loading LoRA from directory (no explicit file found).")
                    pipe.load_lora_weights(lora_path, use_safetensors=True)
            else:
                print(f"→ Loading LoRA file: {lora_path}")
                pipe.load_lora_weights(lora_path, use_safetensors=lora_path.endswith(".safetensors"))
        else:
            print("→ No finetuned dir and no LoRA. Running base SDXL.")

    pipe = accelerator.prepare(pipe)

    try: pipe.enable_attention_slicing()
    except Exception: pass
    try:
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    except Exception:
        pass

    try:
        pipe.to(device)
    except RuntimeError as e:
        if device == "mps" and "MPS backend out of memory" in str(e):
            print("MPS OOM when moving pipeline. Falling back to CPU.")
            device = "cpu"
            torch_dtype = torch.float32
            pipe.to(device)
        else:
            raise

    def run_one(prompt_text: str, count: int):
        prompt_slug = slugify(prompt_text)
        out_base = Path(BASE_DIR) / prompt_slug
        for i in range(count):
            seed_i = (args.seed or 0) + i
            gen = torch.Generator(device=device).manual_seed(seed_i)
            out = pipe(
                prompt_text,
                num_inference_steps=args.steps,
                guidance_scale=args.scale,
                generator=gen,
                height=args.height,
                width=args.width,
            )
            image = out.images[0]
            save_path = save_image(image, out_base, prefix="sdxl_ft")
            print(f"{prompt_text} (seed={seed_i}) → {save_path}")

    if args.use_prompts:
        for p_text, p_count in PROMPTS:
            run_one(p_text, p_count)
        return

    run_one(args.prompt, args.count)

if __name__ == "__main__":
    main()

'''
RUN WITH:
python use_tunned_difusser_v2.py \
  --prompts "A portrait photo of a person at social services||Toys in Iraq||A productive person" \
  --counts "2||3||1" \
  --seed 123 --steps 40 --width 768 --height 768
'''

