"""Build the v2 grounding/decision dataset for ALL 218 APTOS-2021 eyes.

v1 had Claude-vision 6x6 fluid masks for the 35 TEST eyes only. v2 extends to the
FULL cohort (train+test, 218 eyes with a pre-injection image), at 12x12 grid
resolution, with richer per-eye metadata and the continue/stop decision label.

This script is DETERMINISTIC (no vision judgement). It:
  1. renders each eye's representative pre-injection macular B-scan, contrast-
     stretched + upscaled, both as a clean crop and with a 12x12 grid + axis
     labels (the canvas the annotator marks fluid cells on);
  2. assembles metadata_v2.json (demographics, drug, VA/CST pre&post + deltas,
     biomarkers, fluid types, train/test split, continue_injection label);
  3. emits annotation_manifest.json — the work-list for the vision annotation
     pass (Phase 2), each entry carrying the GT biomarkers that constrain which
     fluid TYPES may be localized.

Run in the `aptos2021` env (no GPU):
  PYTHONNOUSERSITE=1 conda run -n aptos2021 python3 -u gen_dataset_v2.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import data

OUT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf/fluid_masks_v2"
GRID_N = 12                      # 12x12 cells
UPSCALE_W = 360                  # target render width (px); height scales w/ aspect
PROBLEM_FLUID = ("IRF", "SRF", "PED")   # HRF is a marker, not fluid


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def contrast_stretch(img: Image.Image, lo=2, hi=98) -> Image.Image:
    a = np.asarray(img.convert("L"), dtype=np.float32)
    p_lo, p_hi = np.percentile(a, lo), np.percentile(a, hi)
    if p_hi <= p_lo:
        p_hi = p_lo + 1
    a = np.clip((a - p_lo) / (p_hi - p_lo), 0, 1) * 255.0
    return Image.fromarray(a.astype(np.uint8)).convert("RGB")


def render_eye(rec) -> Image.Image | None:
    bscan = data.representative_pre_bscan(rec)
    if bscan is None:
        return None
    img = contrast_stretch(bscan)
    w, h = img.size
    scale = UPSCALE_W / w
    img = img.resize((UPSCALE_W, int(round(h * scale))), Image.LANCZOS)
    return img


def draw_grid(img: Image.Image) -> Image.Image:
    """12x12 grid + row/col axis labels on a padded canvas (for annotation)."""
    w, h = img.size
    pad = 22
    canvas = Image.new("RGB", (w + pad, h + pad), (20, 20, 20))
    canvas.paste(img, (pad, pad))
    d = ImageDraw.Draw(canvas)
    f = _font(11)
    cw, ch = w / GRID_N, h / GRID_N
    for c in range(GRID_N + 1):
        x = pad + c * cw
        d.line([(x, pad), (x, pad + h)], fill=(0, 255, 0), width=1)
        if c < GRID_N:
            d.text((pad + c * cw + cw / 2 - 4, 4), str(c), fill=(0, 255, 0), font=f)
    for r in range(GRID_N + 1):
        y = pad + r * ch
        d.line([(pad, y), (pad + w, y)], fill=(0, 255, 0), width=1)
        if r < GRID_N:
            d.text((4, pad + r * ch + ch / 2 - 6), str(r), fill=(0, 255, 0), font=f)
    return canvas


def main():
    os.makedirs(os.path.join(OUT, "clean"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "grid"), exist_ok=True)

    recs = data.build_eye_records()
    tr, te = data.stratified_split(recs)
    test_ids = {r.eye_id for r in te}

    meta, manifest, skipped = [], [], []
    for rec in recs:
        img = render_eye(rec)
        if img is None:
            skipped.append(rec.eye_id)
            continue
        img.save(os.path.join(OUT, "clean", f"{rec.eye_id}.png"))
        draw_grid(img).save(os.path.join(OUT, "grid", f"{rec.eye_id}.png"))

        bm = {b: int(rec.biomarkers.get(b, 0)) for b in data.BIOMARKERS}
        fluid_types = [b for b in PROBLEM_FLUID if bm[b]]
        d_cst = (rec.cst - rec.pre_cst) if not (np.isnan(rec.cst) or np.isnan(rec.pre_cst)) else None
        d_va = (rec.va - rec.pre_va) if not (np.isnan(rec.va) or np.isnan(rec.pre_va)) else None
        row = {
            "eye_id": rec.eye_id,
            "split": "test" if rec.eye_id in test_ids else "train",
            "continue_injection": int(rec.continue_injection),   # LABEL
            "diagnosis": rec.diagnosis,
            "drug": rec.drug,
            "age": rec.age,
            "gender": rec.gender,
            "pre_va": None if np.isnan(rec.pre_va) else round(float(rec.pre_va), 4),
            "pre_cst": None if np.isnan(rec.pre_cst) else float(rec.pre_cst),
            "post_va": None if np.isnan(rec.va) else round(float(rec.va), 4),
            "post_cst": None if np.isnan(rec.cst) else float(rec.cst),
            "delta_cst": None if d_cst is None else round(float(d_cst), 2),
            "delta_va": None if d_va is None else round(float(d_va), 4),
            "biomarkers": bm,
            "fluid_types": fluid_types,
            "has_fluid": int(len(fluid_types) > 0),
            "img_size": list(img.size),
            "grid_n": GRID_N,
        }
        meta.append(row)
        manifest.append({
            "eye_id": rec.eye_id,
            "grid_png": f"grid/{rec.eye_id}.png",
            "diagnosis": rec.diagnosis,
            "fluid_types_present": fluid_types,   # constrains annotation
            "has_fluid": row["has_fluid"],
        })

    with open(os.path.join(OUT, "metadata_v2.json"), "w") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT, "annotation_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)

    n_tr = sum(m["split"] == "train" for m in meta)
    n_te = sum(m["split"] == "test" for m in meta)
    n_fl = sum(m["has_fluid"] for m in meta)
    print(f"rendered {len(meta)} eyes (train={n_tr} test={n_te}) | has_fluid={n_fl} | skipped(no img)={skipped}")
    print(f"out: {OUT}/  [clean/ grid/ metadata_v2.json annotation_manifest.json]")


if __name__ == "__main__":
    main()
