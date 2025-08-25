# gen_postprocess_enhanced_logging.py (FAST)
# ------------------------------------------------------------
# Same features as before (quality+seguridad+selección+logging),
# pero con optimizaciones fuertes de rendimiento:
# - Generación por lotes (num_images_per_prompt) con seeds reproducibles
# - CLIP en batch: text feature 1x por prompt + image features 1x por imagen
# - Reutiliza embeddings para CLIP alignment y estética (sin recomputar)
# - Safety en batch: NSFW para todas; Caption sólo si es necesario (modo AUTO)
# - Opciones de scheduler rápido (DPMSolver) y xFormers si está disponible
# - Batches configurables para generación/CLIP/NSFW/Caption
#
# USO sugerido (rápido y con buen equilibrio):
#   python gen_postprocess_enhanced_logging.py \
#     --prompts "Toys in Iraq" "A portrait photo of a productive person" \
#     --n-candidates 40 --top-k 6 --steps 18 --guidance 6.5 \
#     --gen-batch-size 8 --clip-batch-size 16 --safety-batch-size 16 \
#     --caption-mode auto --log-level INFO --save-metrics --metrics-fmt csv
#
# ------------------------------------------------------------

# gen_postprocess_enhanced_logging.py (FAST)
# ------------------------------------------------------------
# Same features as before (quality+seguridad+selección+logging),
# pero con optimizaciones fuertes de rendimiento:
# - Generación por lotes (num_images_per_prompt) con seeds reproducibles
# - CLIP en batch: text feature 1x por prompt + image features 1x por imagen
# - Reutiliza embeddings para CLIP alignment y estética (sin recomputar)
# - Safety en batch: NSFW para todas; Caption sólo si es necesario (modo AUTO)
# - Opciones de scheduler rápido (DPMSolver) y xFormers si está disponible
# - Batches configurables para generación/CLIP/NSFW/Caption
#
# USO sugerido (rápido y con buen equilibrio):
#   python gen_postprocess_enhanced_logging.py \
#     --prompts "Toys in Iraq" "A portrait photo of a productive person" \
#     --n-candidates 40 --top-k 6 --steps 18 --guidance 6.5 \
#     --gen-batch-size 8 --clip-batch-size 16 --safety-batch-size 16 \
#     --caption-mode auto --log-level INFO --save-metrics --metrics-fmt csv
#
# ------------------------------------------------------------

# gen_postprocess_enhanced_logging.py (FAST, FIXED)
# ------------------------------------------------------------
# Text-to-image generation with upgraded postprocessing + RICH LOGGING
# Optimized & bugfixed version:
# - Batched generation, CLIP features, and safety checks
# - DPMSolver option, xFormers, optional torch.compile
# - Logging controls: progress bar, batch progress, DEBUG details
# - Metrics CSV/JSON, candidate saving, pHash dedup, FPS diversity
# - Robust error handling (no UnboundLocalError / NameError)
# ------------------------------------------------------------

# gen_postprocess_enhanced_logging.py (FAST, FIXED)
# ------------------------------------------------------------
# Text-to-image generation with upgraded postprocessing + RICH LOGGING
# Optimized & bugfixed version:
# - Batched generation, CLIP features, and safety checks
# - DPMSolver option, xFormers, optional torch.compile
# - Logging controls: progress bar, batch progress, DEBUG details
# - Metrics CSV/JSON, candidate saving, pHash dedup, FPS diversity
# - Robust error handling (no UnboundLocalError / NameError)
# ------------------------------------------------------------

# gen_postprocess_enhanced_logging.py (FAST, FIXED)
# ------------------------------------------------------------
# Text-to-image generation with upgraded postprocessing + RICH LOGGING
# Optimized & bugfixed version:
# - Batched generation, CLIP features, and safety checks
# - DPMSolver option, xFormers, optional torch.compile
# - Logging controls: progress bar, batch progress, DEBUG details
# - Metrics CSV/JSON, candidate saving, pHash dedup, FPS diversity
# - Robust error handling (no UnboundLocalError / NameError)
# ------------------------------------------------------------

import argparse
import csv
import datetime as dt
import gc
import json
import logging
import os
import random
import re
import sys
import pathlib
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image

# Optional: OpenCV for IQA (sharpness)
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# Optional: perceptual hash for dedup
try:
    import imagehash
    _HAS_PHASH = True
except Exception:
    _HAS_PHASH = False

import torch
from transformers import (
    CLIPModel, CLIPProcessor,
    pipeline as hf_pipeline,
)
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)

# -------------------------------
# Device & dtype
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
# Logging
# -------------------------------

def setup_logger(outdir: pathlib.Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("gen_postprocess")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(outdir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.info(f"[Device] {DEVICE} | dtype={DTYPE}")
    return logger

# -------------------------------
# SD pipeline
# -------------------------------

def dummy_safety(images, clip_input):
    return images, [False] * len(images)


def _maybe_enable_speed_tricks(pipe, logger: logging.Logger, allow_compile: bool, *, progress_bar: bool):
    try:
        pipe.set_progress_bar_config(disable=not progress_bar)
    except Exception:
        pass
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.vae.to(memory_format=torch.channels_last)
        except Exception:
            pass
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("[Speed] xFormers attention activado")
        except Exception:
            logger.info("[Speed] xFormers no disponible")
        if allow_compile and hasattr(torch, "compile"):
            try:
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
                logger.info("[Speed] torch.compile aplicado a UNet")
            except Exception as e:
                logger.info(f"[Speed] torch.compile no aplicado: {e}")


def make_pipeline(model_id: str, use_dpm: bool, allow_compile: bool, logger: logging.Logger, *, progress_bar: bool):
    if "stable-diffusion-xl" in model_id:
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=DTYPE, use_safetensors=True)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE, use_safetensors=True)
    if use_dpm:
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            logger.info("[Scheduler] DPMSolverMultistep activado")
        except Exception:
            logger.info("[Scheduler] No se pudo activar DPM, usando el por defecto")
    pipe = pipe.to(DEVICE)
    pipe.safety_checker = dummy_safety
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass
    try:
        pipe.enable_attention_slicing("max")
    except Exception:
        pass
    _maybe_enable_speed_tricks(pipe, logger, allow_compile, progress_bar=progress_bar)
    return pipe

# -------------------------------
# CLIP (alignment + embeddings)
# -------------------------------

_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
_clip_model: Optional[CLIPModel] = None
_clip_proc: Optional[CLIPProcessor] = None


def load_clip():
    global _clip_model, _clip_proc
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained(_CLIP_MODEL_ID).to(DEVICE)
        _clip_proc = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)

@torch.no_grad()

def clip_text_features(prompt: str) -> torch.Tensor:
    load_clip()
    inputs = _clip_proc(text=[prompt], return_tensors="pt", padding=True).to(DEVICE)
    outputs = _clip_model.get_text_features(**inputs)
    feats = outputs / outputs.norm(dim=-1, keepdim=True)
    return feats.squeeze(0)

@torch.no_grad()

def clip_image_embeddings_batch(images: List[Image.Image], batch_size: int = 16) -> np.ndarray:
    load_clip()
    embs: List[np.ndarray] = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = _clip_proc(images=batch, return_tensors="pt").to(DEVICE)
        feats = _clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.detach().cpu().numpy())
    if not embs:
        return np.zeros((0, 512), dtype=np.float32)
    return np.vstack(embs)

# -------------------------------
# Aesthetic score (proxy from CLIP emb)
# -------------------------------

def aesthetic_from_emb(emb_row: np.ndarray) -> float:
    return float(np.std(emb_row))

# -------------------------------
# IQA (sharpness)
# -------------------------------

def iqa_sharpness_batch(images: List[Image.Image]) -> List[float]:
    vals: List[float] = []
    for img in images:
        arr = np.array(img.convert("RGB"))
        if _HAS_CV2:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        else:
            gray = arr.mean(axis=2)
            gy, gx = np.gradient(gray.astype(np.float32))
            vals.append(float((gx.var() + gy.var()) / 2.0))
    return vals

# -------------------------------
# Safety & bias filters (batched)
# -------------------------------

_nsfw = None
_captioner = None

BANNED_KEYWORDS = {
    "gore", "blood", "weapon", "gun", "nudity", "nude", "explicit",
    "violence", "beheading", "genital", "nsfw"
}


def load_safety():
    global _nsfw, _captioner
    if _nsfw is None:
        _nsfw = hf_pipeline("image-classification", model="AdamCodd/vit-base-nsfw-detector", device=0 if DEVICE=="cuda" else -1)
    if _captioner is None:
        _captioner = hf_pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=0 if DEVICE=="cuda" else -1)


def nsfw_scores_batch(images: List[Image.Image], batch_size: int, logger: logging.Logger) -> List[Optional[float]]:
    load_safety()
    scores: List[Optional[float]] = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        try:
            out = _nsfw(batch)
            if out and isinstance(out[0], dict):
                out = [out]
            for res in out:
                ns = None
                for r in res:
                    if r.get("label", "").lower() == "nsfw":
                        ns = float(r.get("score", 0.0))
                        break
                scores.append(ns)
        except Exception as e:
            logger.warning(f"[SAFETY WARN] NSFW batch error: {e}")
            scores.extend([None] * len(batch))
    return scores


def captions_batch(images: List[Image.Image], batch_size: int, max_new_tokens: int, logger: logging.Logger) -> List[str]:
    load_safety()
    caps: List[str] = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        try:
            outs = _captioner(batch, max_new_tokens=max_new_tokens)
            if outs and isinstance(outs[0], dict):
                outs = [outs]
            for res in outs:
                txt = res[0].get("generated_text", "") if res else ""
                caps.append(txt)
        except Exception as e:
            logger.warning(f"[SAFETY WARN] Caption batch error: {e}")
            caps.extend([""] * len(batch))
    return caps


def safety_gate(images: List[Image.Image], *, nsfw_threshold: float, nsfw_margin: float,
                caption_mode: str, safety_batch_size: int, caption_batch_size: int,
                caption_max_tokens: int, logger: logging.Logger,
                log_captions: bool) -> Tuple[List[bool], List[Dict[str, Any]]]:
    metrics: List[Dict[str, Any]] = []
    nsfw_scores = nsfw_scores_batch(images, safety_batch_size, logger)

    need_caption_idx: List[int] = []
    if caption_mode in ("auto", "unsafe", "all"):
        for i, s in enumerate(nsfw_scores):
            if caption_mode == "all":
                need_caption_idx.append(i)
            elif caption_mode == "unsafe":
                if s is not None and s > (nsfw_threshold - nsfw_margin):
                    need_caption_idx.append(i)
            else:  # auto
                if s is not None and (nsfw_threshold - nsfw_margin) < s <= nsfw_threshold:
                    need_caption_idx.append(i)

    captions_map: Dict[int, str] = {}
    if need_caption_idx:
        subset = [images[i] for i in need_caption_idx]
        caps = captions_batch(subset, caption_batch_size, caption_max_tokens, logger)
        for j, i in enumerate(need_caption_idx):
            captions_map[i] = caps[j]

    accepts: List[bool] = []
    for i, s in enumerate(nsfw_scores):
        info = {"nsfw_score": s, "caption": None, "keyword_hit": None, "reason": None}
        if s is not None and s > nsfw_threshold:
            info["reason"] = f"nsfw_score>{nsfw_threshold:.2f}"
            accepts.append(False)
            metrics.append(info)
            continue
        cap = captions_map.get(i)
        if cap is not None:
            cap_l = cap.lower()
            hit = next((k for k in BANNED_KEYWORDS if k in cap_l), None)
            info["caption"] = cap if log_captions else None
            info["keyword_hit"] = hit
            if hit is not None:
                info["reason"] = f"keyword:{hit}"
                accepts.append(False)
                metrics.append(info)
                continue
        accepts.append(True)
        metrics.append(info)

    return accepts, metrics

# -------------------------------
# Dedup + Diversity
# -------------------------------

def phash_value(img: Image.Image) -> Optional[str]:
    if not _HAS_PHASH:
        return None
    return str(imagehash.phash(img))


def diversify_by_fps(cands: List[Dict[str, Any]], k: int, logger: logging.Logger) -> List[Dict[str, Any]]:
    if not cands:
        return []
    selected = [max(cands, key=lambda x: x["score"])]
    remaining = [c for c in cands if c is not selected[0]]

    def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
        return 1.0 - float(np.dot(a, b))

    while len(selected) < min(k, len(cands)) and remaining:
        best, best_d = None, -1.0
        for c in remaining:
            d = min(cos_dist(c["emb"], s["emb"]) for s in selected)
            if d > best_d:
                best_d, best = d, c
        selected.append(best)
        remaining.remove(best)
        logger.debug(f"  [FPS] Added cand_id={best['cand_id']} minDist={best_d:.4f} score={best['score']:.3f}")
    return selected

# -------------------------------
# Scoring & selection
# -------------------------------

def zscale(values: List[float]) -> List[float]:
    arr = np.array(values, dtype=np.float32)
    if len(arr) < 2:
        return arr.tolist()
    mu, sd = float(arr.mean()), float(arr.std() + 1e-6)
    return ((arr - mu) / sd).tolist()


def score_and_select(prompt: str, images: List[Image.Image], top_k: int, phash_thresh: int, *,
                     logger: logging.Logger,
                     nsfw_threshold: float, nsfw_margin: float,
                     caption_mode: str, safety_batch_size: int, caption_batch_size: int,
                     caption_max_tokens: int, log_captions: bool,
                     clip_batch_size: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    metrics: List[Dict[str, Any]] = []

    # 1) Safety batched
    accepts, saf_metrics = safety_gate(images, nsfw_threshold=nsfw_threshold, nsfw_margin=nsfw_margin,
                                       caption_mode=caption_mode, safety_batch_size=safety_batch_size,
                                       caption_batch_size=caption_batch_size, caption_max_tokens=caption_max_tokens,
                                       logger=logger, log_captions=log_captions)
    for idx, m in enumerate(saf_metrics):
        metrics.append({"cand_id": idx, "safety_ok": accepts[idx], **m})
        if not accepts[idx]:
            logger.info(f"  [DROP] cand_id={idx} safety reject: reason={m.get('reason')} nsfw={m.get('nsfw_score')}")

    safe_idx = [i for i, ok in enumerate(accepts) if ok]
    if not safe_idx:
        logger.warning("  [WARN] All candidates rejected by safety.")
        return [], metrics

    safe_imgs = [images[i] for i in safe_idx]

    # 2) CLIP emb batch + alignment
    text_f = clip_text_features(prompt)
    img_embs = clip_image_embeddings_batch(safe_imgs, batch_size=clip_batch_size)
    text_np = text_f.detach().cpu().numpy()
    clip_scores = [float(np.dot(text_np, e)) for e in img_embs]

    # 3) Aesthetic proxy + IQA
    aes_scores = [aesthetic_from_emb(e) for e in img_embs]
    iqa_scores = iqa_sharpness_batch(safe_imgs)

    # 4) z-score & combine
    clip_z = zscale(clip_scores)
    aes_z = zscale(aes_scores)
    iqa_z = zscale(iqa_scores)
    combined = [0.5*clip_z[i] + 0.4*aes_z[i] + 0.1*iqa_z[i] for i in range(len(safe_imgs))]

    # 5) Bundle
    bundled: List[Dict[str, Any]] = []
    for j, si in enumerate(safe_idx):
        bundle = {
            "cand_id": si,
            "img": safe_imgs[j],
            "score": float(combined[j]),
            "clip": float(clip_scores[j]),
            "aesthetic": float(aes_scores[j]),
            "iqa": float(iqa_scores[j]),
            "clip_z": float(clip_z[j]),
            "aesthetic_z": float(aes_z[j]),
            "iqa_z": float(iqa_z[j]),
            "emb": img_embs[j],
            "phash": phash_value(safe_imgs[j]),
            **saf_metrics[si],
        }
        bundled.append(bundle)
        for m in metrics:
            if m["cand_id"] == si:
                m.update({
                    "clip": bundle["clip"],
                    "aesthetic": bundle["aesthetic"],
                    "iqa": bundle["iqa"],
                    "clip_z": bundle["clip_z"],
                    "aesthetic_z": bundle["aesthetic_z"],
                    "iqa_z": bundle["iqa_z"],
                    "combined": bundle["score"],
                    "phash": bundle["phash"],
                })
                break

    # 6) pHash dedup
    if _HAS_PHASH:
        uniq: List[Dict[str, Any]] = []
        for c in bundled:
            dup_of = None
            for u in uniq:
                if c["phash"] and u["phash"]:
                    d = imagehash.hex_to_hash(c["phash"]) - imagehash.hex_to_hash(u["phash"])
                    if d <= phash_thresh:
                        dup_of = (u["cand_id"], d)
                        break
            if dup_of is None:
                uniq.append(c)
            else:
                logger.info(f"  [DROP] cand_id={c['cand_id']} near-duplicate of cand_id={dup_of[0]} (pHash dist={dup_of[1]})")
                for m in metrics:
                    if m["cand_id"] == c["cand_id"]:
                        m.update({"dedup_dropped": True, "dup_of": dup_of[0], "phash_dist": dup_of[1]})
                        break
    else:
        uniq = bundled
        logger.info("  [INFO] pHash no disponible -> dedup omitido.")

    # 7) Sort & diversify
    uniq.sort(key=lambda x: x["score"], reverse=True)
    winners = diversify_by_fps(uniq, top_k, logger)

    win_ids = {w["cand_id"] for w in winners}
    for m in metrics:
        m["selected"] = m.get("cand_id") in win_ids

    logger.info(f"  [SELECT] winners={len(winners)} / uniq={len(uniq)} / safe={len(safe_idx)}")
    return winners, metrics

# -------------------------------
# Generation (batched)
# -------------------------------

def sanitize_dir_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "prompt"


def save_metrics_csv(metrics_path: pathlib.Path, prompt: str, cols: List[str], rows: List[Dict[str, Any]]):
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            r2 = {**r, "prompt": prompt}
            w.writerow(r2)


def save_metrics_json(metrics_path: pathlib.Path, prompt: str, rows: List[Dict[str, Any]]):
    payload = {"prompt": prompt, "metrics": rows}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def batched_generate(
        pipe,
        prompt: str,
        n: int,
        steps: int,
        guidance: float,
        height: int,
        width: int,
        gen_batch_size: int,
        logger: logging.Logger,
        log_batch_progress: bool = False,
        step_progress_every: int = 0,
) -> Tuple[List[Image.Image], List[int]]:
    images: List[Image.Image] = []
    seeds: List[int] = []

    for start in range(0, n, gen_batch_size):
        b = min(gen_batch_size, n - start)
        seed_list = [random.randint(0, 2**32 - 1) for _ in range(b)]
        gens = [torch.Generator(device=DEVICE).manual_seed(s) for s in seed_list]
        try:
            if step_progress_every and step_progress_every > 0:
                def _cb(i, t, kwargs):
                    if i % step_progress_every == 0:
                        try:
                            logger.info(f"  [STEP] batch {start + b}/{n} step {i}/{steps} t={t}")
                        except Exception:
                            pass
                out = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    num_images_per_prompt=b,
                    generator=gens,
                    callback=_cb,
                    callback_steps=1,
                )
            else:
                out = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    num_images_per_prompt=b,
                    generator=gens,
                )
            batch_imgs = out.images
            images.extend(batch_imgs)
            seeds.extend(seed_list)

            if log_batch_progress or logger.level <= logging.DEBUG:
                logger.info(f"  [GEN] batch {start + b}/{n} (+{len(batch_imgs)})")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("  [OOM] batch: liberando caché y continuando")
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            else:
                logger.exception("  [ERROR] Generación batch %d: %s", start, e)
        except Exception as e:
            logger.exception("  [ERROR] Generación batch %d: %s", start, e)
        finally:
            if DEVICE == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    return images, seeds


def generate_for_prompt(
        pipe,
        prompt: str,
        outdir: pathlib.Path,
        n_candidates: int,
        steps: int,
        guidance: float,
        height: int,
        width: int,
        top_k: int,
        phash_thresh: int,
        logger: logging.Logger,
        nsfw_threshold: float,
        nsfw_margin: float,
        caption_mode: str,
        safety_batch_size: int,
        caption_batch_size: int,
        caption_max_tokens: int,
        log_captions: bool,
        save_candidates: bool,
        save_metrics: bool,
        metrics_fmt: str,
        clip_batch_size: int,
        gen_batch_size: int,
        progress_bar: bool = False,
        log_batch_progress: bool = False,
        step_progress_every: int = 0,
):
    logger.info(f"[START] Prompt: {prompt}")
    prompt_dir = outdir / sanitize_dir_name(prompt)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    cand_dir = prompt_dir / "_candidates"
    if save_candidates:
        cand_dir.mkdir(exist_ok=True)

    # Generate once (batched)
    candidates: List[Image.Image] = []
    seeds: List[int] = []
    try:
        candidates, seeds = batched_generate(
            pipe, prompt, n_candidates, steps, guidance, height, width,
            gen_batch_size, logger, log_batch_progress=log_batch_progress,
        )
    except Exception as e:
        logger.exception("  [ERROR] batched_generate failed; continuing with 0 candidates: %s", e)

    logger.info("  Generated candidates: %d / requested: %d", len(candidates), n_candidates)

    if not candidates:
        logger.warning("  [WARN] No candidates generated; skipping selection.")
        return

    if save_candidates:
        for i, img in enumerate(candidates):
            fp = cand_dir / f"cand_{i:04d}.png"
            try:
                img.save(fp)
            except Exception as e:
                logger.warning(f"  [SAVE WARN] cand {i}: {e}")

    # Scoring + selection
    winners, per_metrics = score_and_select(
        prompt, candidates, top_k=top_k, phash_thresh=phash_thresh, logger=logger,
        nsfw_threshold=nsfw_threshold, nsfw_margin=nsfw_margin,
        caption_mode=caption_mode, safety_batch_size=safety_batch_size,
        caption_batch_size=caption_batch_size, caption_max_tokens=caption_max_tokens,
        log_captions=log_captions, clip_batch_size=clip_batch_size,
    )

    # Attach seeds
    for m in per_metrics:
        cid = m.get("cand_id")
        if cid is not None and cid < len(seeds):
            m["seed"] = int(seeds[cid])

    logger.info(f"  Selected winners: {len(winners)}")

    # Save winners
    for idx, w in enumerate(winners, start=1):
        fp = prompt_dir / f"{idx:02d}.png"
        try:
            w["img"].save(fp)
            logger.info(f"  [OK] Saved: {fp.name} (cand_id={w['cand_id']} score={w['score']:.3f})")
        except Exception as e:
            logger.error(f"  [SAVE ERROR] {e}")

    # Save metrics
    if save_metrics:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        if metrics_fmt.lower() == "csv":
            cols = [
                "prompt","cand_id","seed","safety_ok","nsfw_score","keyword_hit","reason",
                "clip","aesthetic","iqa","clip_z","aesthetic_z","iqa_z","combined","phash",
                "dedup_dropped","dup_of","phash_dist","selected","caption"
            ]
            metrics_path = prompt_dir / f"metrics_{ts}.csv"
            save_metrics_csv(metrics_path, prompt, cols, per_metrics)
            logger.info(f"  [METRICS] CSV saved: {metrics_path.name}")
        else:
            metrics_path = prompt_dir / f"metrics_{ts}.json"
            save_metrics_json(metrics_path, prompt, per_metrics)
            logger.info(f"  [METRICS] JSON saved: {metrics_path.name}")

    # Cleanup
    del candidates, winners
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# -------------------------------
# CLI
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="+", required=False,
                    default=["Toys in Iraq", "A portrait photo of a productive person"])
    ap.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--outdir", type=str, default="postprocess_outputs_v2")
    ap.add_argument("--n-candidates", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--phash-thresh", type=int, default=6)

    # Progreso/logs
    ap.add_argument("--progress-bar", action="store_true", help="Muestra la barra de progreso de diffusers")
    ap.add_argument("--log-batch-progress", action="store_true", help="Loguea por INFO cada batch generado")
    ap.add_argument("--step-progress-every", type=int, default=0, help="Log de paso cada K steps (0=off)")

    # Velocidad
    ap.add_argument("--use-dpm", action="store_true", help="Usar DPMSolverMultistepScheduler")
    ap.add_argument("--compile-unet", action="store_true", help="Intentar torch.compile en la UNet (CUDA)")
    ap.add_argument("--gen-batch-size", type=int, default=8)
    ap.add_argument("--clip-batch-size", type=int, default=16)
    ap.add_argument("--safety-batch-size", type=int, default=16)
    ap.add_argument("--caption-batch-size", type=int, default=8)

    # Safety/caption
    ap.add_argument("--nsfw-threshold", type=float, default=0.7)
    ap.add_argument("--nsfw-margin", type=float, default=0.05, help="Zona borderline para activar caption en modo AUTO")
    ap.add_argument("--caption-mode", type=str, default="auto", choices=["auto","winners","unsafe","all","none"],
                    help="AUTO: caption solo borderline; WINNERS: caption a ganadores (no bloquea); UNSAFE: caption a NSFW borderline; ALL/NONE")
    ap.add_argument("--caption-max-tokens", type=int, default=30)

    # Logging/metrics
    ap.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    ap.add_argument("--save-candidates", action="store_true")
    ap.add_argument("--save-metrics", action="store_true")
    ap.add_argument("--metrics-fmt", type=str, default="csv", choices=["csv","json"])
    ap.add_argument("--log-captions", action="store_true")

    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(outdir, args.log_level)

    pipe = make_pipeline(args.model_id, use_dpm=args.use_dpm, allow_compile=args.compile_unet, logger=logger, progress_bar=args.progress_bar)

    for prompt in args.prompts:
        generate_for_prompt(
            pipe,
            prompt=prompt,
            outdir=outdir,
            n_candidates=args.n_candidates,
            steps=args.steps,
            guidance=args.guidance,
            height=args.height,
            width=args.width,
            top_k=args.top_k,
            phash_thresh=args.phash_thresh,
            logger=logger,
            nsfw_threshold=args.nsfw_threshold,
            nsfw_margin=args.nsfw_margin,
            caption_mode=args.caption_mode,
            safety_batch_size=args.safety_batch_size,
            caption_batch_size=args.caption_batch_size,
            caption_max_tokens=args.caption_max_tokens,
            log_captions=args.log_captions,
            save_candidates=args.save_candidates,
            save_metrics=args.save_metrics,
            metrics_fmt=args.metrics_fmt,
            clip_batch_size=args.clip_batch_size,
            gen_batch_size=args.gen_batch_size,
            progress_bar=args.progress_bar,
            log_batch_progress=args.log_batch_progress,
            step_progress_every=args.step_progress_every,
        )

    logger.info("Done.")

if __name__ == "__main__":
    main()


"""
python bias_preprocessing_v2_comments.py  
--prompts "Toys in Iraq" \
  --n-candidates 2 --steps 18 --guidance 6.5 \
  --use-dpm --gen-batch-size 8 \
  --log-level INFO \
  --log-batch-progress \                # ← log por LOTE generado
  --step-progress-every 2 \             # ← log cada 2 pasos de difusión
  --progress-bar  




python -u bias_preprocessing.py \
  --model-id stabilityai/stable-diffusion-xl-base-1.0 \
  --prompts "Toys in Iraq" \
  --n-candidates 20 --top-k 8 \
    --steps 24 --guidance 5.5 \
  --height 512 --width 512 \
  --gen-batch-size 1 \
  --log-level DEBUG \
  --log-batch-progress \
  --step-progress-every 1 \
  --progress-bar \
  --save-candidates \
  --save-metrics --metrics-fmt csv


  python -u bias_preprocessing_v2_comments.py \
  --model-id stabilityai/stable-diffusion-xl-base-1.0 \
  --prompts "A portrait photo of a person at social services" \
  --n-candidates 20 --top-k 8 \
    --steps 34 --guidance 5.5 \
  --gen-batch-size 1 \
  --log-level DEBUG \
  --log-batch-progress \
  --step-progress-every 1 \
  --progress-bar \
  --save-candidates \
  --save-metrics --metrics-fmt csv
  

python -u bias_preprocessing_v2_comments.py \
  --model-id stabilityai/stable-diffusion-xl-base-1.0 \
  --prompts "A portrait photo of a productive persom" \
  --n-candidates 5 --top-k 3 \
    --steps 24 --guidance 5.5 \
  --gen-batch-size 1 \
  --log-level DEBUG \
  --log-batch-progress \
  --step-progress-every 1 \
  --progress-bar \
  --save-candidates \
  --save-metrics --metrics-fmt csv

# low memory
python -u bias_preprocessing.py \
  --model-id runwayml/stable-diffusion-v1-5 \
  --prompts "Toys in Iraq" \
  --n-candidates 1 --top-k 1 \
  --steps 20 --guidance 7 \
  --height 512 --width 512 \
  --gen-batch-size 1 --use-dpm \
  --log-level INFO --log-batch-progress --progress-bar \
  --caption-mode none

  """