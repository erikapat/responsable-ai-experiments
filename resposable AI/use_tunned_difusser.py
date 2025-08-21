#!/usr/bin/env python3
import os
# Prevent early MPS OOM (must be set before torch import)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import glob
import time
import argparse
import torch
from diffusers import StableDiffusionXLPipeline
from pathlib import Path
from PIL import Image
import time, os, re

prompt = "A portrait photo of a person at social services"
    #"Toys in Iraq",
    #"A portrait photo of a productive person",



def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9_\-]+", "_", s.lower())

def save_image(img, base_dir: Path, prefix="fine_tuned_result", ext="png") -> Path:
    """
    Guarda una imagen PIL.Image en base_dir/results/ con un nombre único.
    También acepta bytes. Si recibe str/Path por error, lanza TypeError claro.
    """
    out_dir = Path(base_dir) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_{int(time.time())}.{ext}"

    # Protección de tipos
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
    p.add_argument("--prompt", default=prompt)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--scale", type=float, default=6.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=512, help="Reduce to avoid MPS OOM (e.g., 512 or 384)")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--finetuned_dir", default=None, help="Path to a saved finetuned pipeline dir")
    p.add_argument("--lora", default=None, help="Path to LoRA file or directory")
    p.add_argument("--base_model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--device", default=None, choices=["cpu","mps","cuda"], help="Force device; default picks best")
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
    print("base_dir contents:")
    try:
        for item in os.listdir(base_dir):
            print("  -", item)
    except Exception as e:
        print("  (failed to list base_dir)", e)
    print("------------------------\n")

def save_image(img, base_dir, prefix="fine_tuned_result"):
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}_{int(time.time())}.png")
    img.save(path)
    return path

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

    # Device & dtype (fp32 for MPS/CPU; fp16 for CUDA)
    device = pick_device(args.device)
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Requested device={device}, dtype={torch_dtype}")
    # Resolve paths relative to script folder
    finetuned_dir = resolve_relative(BASE_DIR, args.finetuned_dir)
    lora_path = resolve_relative(BASE_DIR, args.lora)

    # Auto-discover defaults next to script if not provided
    if not finetuned_dir:
        maybe_ft = os.path.join(BASE_DIR, "sdxl-finetuned")
        if os.path.isdir(maybe_ft):
            finetuned_dir = maybe_ft
    if not lora_path:
        maybe_lora = os.path.join(BASE_DIR, "sdxl-lora-mps")
        if os.path.exists(maybe_lora):
            lora_path = maybe_lora

    # --- Require tuned model (finetuned or LoRA) ---
    if not (finetuned_dir and os.path.isdir(finetuned_dir)) and not (lora_path and os.path.exists(lora_path)):
        print("❌ No finetuned model directory or LoRA file found. Exiting.")
        diagnose_paths(BASE_DIR)
        return

    # Load pipeline
    if finetuned_dir and os.path.isdir(finetuned_dir):
        print(f"→ Loading finetuned pipeline from: {finetuned_dir}")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            finetuned_dir,
            torch_dtype=torch_dtype if device == "cuda" else torch.float32,
            use_safetensors=True,
        )
    else:
        print(f"→ Loading base model: {args.base_model}")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.base_model,
            torch_dtype=torch_dtype if device == "cuda" else torch.float32,
            use_safetensors=True,
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

    # Light memory tweaks
    try: pipe.enable_attention_slicing()
    except Exception: pass
    try:
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    except Exception:
        pass

    # Try to move to requested device; on MPS OOM fallback to CPU
    try:
        pipe.to(device)
    except RuntimeError as e:
        if device == "mps" and "MPS backend out of memory" in str(e):
            print("⚠️ MPS OOM when moving pipeline. Falling back to CPU.")
            device = "cpu"
            torch_dtype = torch.float32
            pipe.to(device)
        else:
            raise

    # Inference
    gen = None
    if args.seed is not None:
        gen = torch.Generator(device=device).manual_seed(args.seed)

    out = pipe(
        args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.scale,
        generator=gen,
        height=args.height,
        width=args.width,   # crucial: avoids 1024×1024 default OOM on MPS
    )

    image = out.images[0]

    prompt_slug = slugify(prompt)
    save_path = save_image(image, Path(BASE_DIR) / prompt_slug, prefix="sdxl_ft")
    print(f"✅ Saved image to: {save_path}")

if __name__ == "__main__":
    main()


