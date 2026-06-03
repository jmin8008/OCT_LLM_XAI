"""Assemble v2 fluid-mask dataset from the per-batch annotation parts.

  annot_parts/batch_*.json  ->  masks_12x12.{json,npz}  +  overlays + montage
                            ->  metadata_v2.json (augmented w/ mask stats)
Cross-checks each mask against the GT biomarker `has_fluid` flag and reports
mismatches (fluid present but mask empty, or vice-versa).
"""
from __future__ import annotations
import glob, json, os
import numpy as np
from PIL import Image

ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf/fluid_masks_v2"
N = 12


def load_parts() -> dict:
    masks = {}
    for p in sorted(glob.glob(f"{ROOT}/annot_parts/batch_*.json")):
        for eye, v in json.load(open(p)).items():
            masks[eye] = v
    return masks


def to_grid(cells) -> np.ndarray:
    g = np.zeros((N, N), dtype=np.uint8)
    for c in cells:
        r, col = int(c[0]), int(c[1])
        if 0 <= r < N and 0 <= col < N:
            g[r, col] = 1
    return g


def main():
    meta = {m["eye_id"]: m for m in json.load(open(f"{ROOT}/metadata_v2.json"))}
    ann = load_parts()

    json_masks, npz_masks = {}, {}
    mismatches, missing = [], []
    os.makedirs(f"{ROOT}/overlay", exist_ok=True)

    for eye, m in meta.items():
        if eye not in ann:
            missing.append(eye)
            continue
        a = ann[eye]
        g = to_grid(a.get("fluid_cells", []))
        json_masks[eye] = g.tolist()
        npz_masks[eye] = g
        ncell = int(g.sum())
        # augment metadata
        m["fluid_cell_count"] = ncell
        m["mask_confidence"] = a.get("confidence", "med")
        m["mask_types_seen"] = a.get("types_seen", [])
        m["mask_note"] = a.get("note", "")
        # cross-check vs GT
        if m["has_fluid"] == 1 and ncell == 0:
            mismatches.append((eye, "GT fluid but EMPTY mask", m["fluid_types"]))
        if m["has_fluid"] == 0 and ncell > 0:
            mismatches.append((eye, "GT no-fluid but mask non-empty", ncell))

        # overlay (red) on clean crop for visual QC
        clean = Image.open(f"{ROOT}/clean/{eye}.png").convert("RGB")
        w, h = clean.size
        ov = np.asarray(clean).copy()
        cw, ch = w / N, h / N
        for r in range(N):
            for col in range(N):
                if g[r, col]:
                    y0, y1 = int(r * ch), int((r + 1) * ch)
                    x0, x1 = int(col * cw), int((col + 1) * cw)
                    ov[y0:y1, x0:x1, 0] = np.minimum(255, ov[y0:y1, x0:x1, 0].astype(int) + 90)
                    ov[y0:y1, x0:x1, 1] = (ov[y0:y1, x0:x1, 1] * 0.6).astype(np.uint8)
                    ov[y0:y1, x0:x1, 2] = (ov[y0:y1, x0:x1, 2] * 0.6).astype(np.uint8)
        Image.fromarray(ov).save(f"{ROOT}/overlay/{eye}.png")

    # save masks
    json.dump(json_masks, open(f"{ROOT}/masks_12x12.json", "w"))
    np.savez_compressed(f"{ROOT}/masks_12x12.npz", **npz_masks)
    json.dump(list(meta.values()), open(f"{ROOT}/metadata_v2.json", "w"), indent=1, ensure_ascii=False)

    # montage (grid of overlays, sorted)
    eyes = sorted(json_masks)
    cols = 16
    rows = (len(eyes) + cols - 1) // cols
    thumb = (70, 190)
    canvas = Image.new("RGB", (cols * thumb[0], rows * thumb[1]), (10, 10, 10))
    for i, eye in enumerate(eyes):
        t = Image.open(f"{ROOT}/overlay/{eye}.png").convert("RGB").resize(thumb)
        canvas.paste(t, ((i % cols) * thumb[0], (i // cols) * thumb[1]))
    canvas.save(f"{ROOT}/ALL_overlays_montage_v2.png")

    cells = [int(np.array(g).sum()) for g in json_masks.values()]
    print(f"assembled {len(json_masks)} masks | missing={missing}")
    print(f"cells/eye: mean={np.mean(cells):.1f} median={np.median(cells):.0f} max={max(cells)}")
    conf = {}
    for m in meta.values():
        conf[m.get("mask_confidence")] = conf.get(m.get("mask_confidence"), 0) + 1
    print(f"confidence dist: {conf}")
    print(f"MISMATCHES vs GT ({len(mismatches)}):")
    for x in mismatches:
        print("  ", x)
    print(f"out: masks_12x12.json/.npz, overlay/, ALL_overlays_montage_v2.png, metadata_v2.json")


if __name__ == "__main__":
    main()
