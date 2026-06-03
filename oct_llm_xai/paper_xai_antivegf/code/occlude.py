"""Generate counterfactual occluded OCT images for the SFT counterfactual pairs.

For each fluid-present eye (occlusion_worklist.json) we remove the fluid region so
the macula reads as "dry", training the model that the continue/stop decision must
depend on the fluid PIXELS (the causal signal), not text priors.

Two design safeguards against shortcut learning:
  1. NO black box. The fluid cells are INPAINTED (cv2.INPAINT_TELEA) so surrounding
     retinal texture fills the dark fluid pocket -> a plausible dry macula, not a
     detectable blackout cue.
  2. Negative-control occlusion. The SAME inpaint operation is applied to an equal
     number of NON-fluid cells in the same retinal rows -> `occluded_negctrl/`.
     A faithfully-grounded model must FLIP (continue->stop) on `occluded/` but NOT
     on `occluded_negctrl/`; the control proves the flip is due to fluid removal,
     not the inpaint artifact.

Run (aptos2021 env):
  PYTHONNOUSERSITE=1 conda run -n aptos2021 python3 -u occlude.py
"""
from __future__ import annotations
import json, os
import numpy as np
from PIL import Image

ROOT = "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf/fluid_masks_v2"
N = 12
SEED = 42

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False


def cell_mask_to_pixels(grid: np.ndarray, h: int, w: int, dilate_px: int = 3) -> np.ndarray:
    """Upsample a 12x12 binary grid to a HxW uint8 mask, with a small dilation."""
    m = np.zeros((h, w), dtype=np.uint8)
    ch, cw = h / N, w / N
    for r in range(N):
        for c in range(N):
            if grid[r, c]:
                y0, y1 = int(r * ch), int((r + 1) * ch)
                x0, x1 = int(c * cw), int((c + 1) * cw)
                m[y0:y1, x0:x1] = 255
    if dilate_px and HAVE_CV2:
        k = np.ones((dilate_px, dilate_px), np.uint8)
        m = cv2.dilate(m, k, iterations=1)
    return m


def inpaint(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if HAVE_CV2:
        return cv2.inpaint(img_rgb, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    # fallback: fill masked pixels with column-wise median of unmasked pixels, then blur
    out = img_rgb.copy()
    g = img_rgb[:, :, 0].astype(float)
    for x in range(img_rgb.shape[1]):
        col_mask = mask[:, x] > 0
        if col_mask.any():
            unmasked = g[~col_mask, x]
            fill = np.median(unmasked) if unmasked.size else g[:, x].mean()
            out[col_mask, x, :] = int(fill)
    return out


def neg_ctrl_grid(grid: np.ndarray, rng) -> np.ndarray:
    """Pick an equal number of NON-fluid cells in the same rows as the fluid cells."""
    fluid_rows = sorted({r for r in range(N) for c in range(N) if grid[r, c]})
    n = int(grid.sum())
    cand = [(r, c) for r in fluid_rows for c in range(N) if not grid[r, c]]
    out = np.zeros_like(grid)
    if not cand or n == 0:
        return out
    idx = rng.choice(len(cand), size=min(n, len(cand)), replace=False)
    for i in idx:
        r, c = cand[i]
        out[r, c] = 1
    return out


def main():
    os.makedirs(f"{ROOT}/occluded", exist_ok=True)
    os.makedirs(f"{ROOT}/occluded_negctrl", exist_ok=True)
    masks = np.load(f"{ROOT}/masks_12x12.npz")
    work = json.load(open(f"{ROOT.replace('fluid_masks_v2','sft_data')}/occlusion_worklist.json"))
    rng = np.random.default_rng(SEED)

    done = 0
    montage_eyes = []
    for item in work:
        eye = item["eye_id"]
        grid = np.asarray(masks[eye])
        clean = Image.open(f"{ROOT}/clean/{eye}.png").convert("RGB")
        arr = np.asarray(clean)
        h, w = arr.shape[:2]
        # fluid occlusion
        m = cell_mask_to_pixels(grid, h, w)
        Image.fromarray(inpaint(arr, m)).save(f"{ROOT}/occluded/{eye}.png")
        # negative-control occlusion (same op, non-fluid cells)
        ng = neg_ctrl_grid(grid, rng)
        mn = cell_mask_to_pixels(ng, h, w)
        Image.fromarray(inpaint(arr, mn)).save(f"{ROOT}/occluded_negctrl/{eye}.png")
        done += 1
        montage_eyes.append(eye)

    # QC montage: clean | occluded | negctrl for first 18 eyes
    cols = 3
    sample = montage_eyes[:18]
    thumb = (90, 250)
    canvas = Image.new("RGB", (cols * thumb[0], len(sample) * thumb[1]), (10, 10, 10))
    for i, eye in enumerate(sample):
        for j, sub in enumerate(("clean", "occluded", "occluded_negctrl")):
            t = Image.open(f"{ROOT}/{sub}/{eye}.png").convert("RGB").resize(thumb)
            canvas.paste(t, (j * thumb[0], i * thumb[1]))
    canvas.save(f"{ROOT}/occlusion_QC_montage.png")

    print(f"cv2 inpaint: {HAVE_CV2}")
    print(f"occluded {done} eyes -> occluded/ + occluded_negctrl/ ; QC: occlusion_QC_montage.png")


if __name__ == "__main__":
    main()
