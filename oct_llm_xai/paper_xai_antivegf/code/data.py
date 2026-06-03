"""APTOS-2021 anti-VEGF data loading for VLM inference.

- Loads case-level labels (train_anno_case.csv) and image-level biomarker labels
  (train_anno_pic.csv).
- Resolves per-eye pre/post OCT B-scan paths.
- Picks a representative pre-injection center B-scan (G-channel row-sum heuristic,
  ported from data_response/experiment/autoresearch_task2/prepare.py:159) and
  crops the central macular OCT.
- Eye-level stratified split (diagnosis x continue-injection).
- make_kfold_splits() is provided but its USE is commented out by default; the
  default pipeline uses a single stratified split. Uncomment in experiments.py to
  enable k-fold (statistical-power robustness check).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

# --- constants (ported from autoresearch_task2/prepare.py) ----------------
OCT_X_START = 632          # 1264-wide image: left half = fundus, right half = OCT
FUNDUS_X_END = 632
MACULAR_CROP_RATIO = 1.0 / 3.0
RADIAL_IMAGE_THRESHOLD = 6

DATA_ROOT = (
    "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/data_response/anti-vegf-dataset/"
    "APTOS-2021/Final Datasets"
)
TRAIN_IMG_ROOT = os.path.join(DATA_ROOT, "Training Set", "Training Set")
CASE_CSV = os.path.join(DATA_ROOT, "train_anno_case.csv")
PIC_CSV = os.path.join(DATA_ROOT, "train_anno_pic.csv")

BIOMARKERS = ["IRF", "SRF", "PED", "HRF"]


# --- image utilities (ported) ---------------------------------------------
def detect_center_image_fundus(raw_img: Image.Image) -> int:
    """Brightest fundus row (G-channel row sum) -> central B-scan index proxy."""
    arr = np.array(raw_img)
    fundus = arr[:, :FUNDUS_X_END, :]
    g_channel = fundus[:, :, 1]
    row_sums = g_channel.sum(axis=1)
    return int(np.argmax(row_sums))


def extract_macular_oct_crop(
    img: Image.Image, crop_ratio: float = MACULAR_CROP_RATIO, random_offset: int = 0
) -> Image.Image:
    w, h = img.size
    oct_width = w - OCT_X_START
    crop_width = int(oct_width * crop_ratio)
    center_x = OCT_X_START + oct_width // 2
    x_start = center_x - crop_width // 2 + random_offset
    x_start = max(OCT_X_START, min(x_start, w - crop_width))
    return img.crop((x_start, 0, x_start + crop_width, h))


def classify_scan_type(num_images: int) -> str:
    return "radial" if num_images <= RADIAL_IMAGE_THRESHOLD else "horizontal"


# --- records ---------------------------------------------------------------
@dataclass
class EyeRecord:
    eye_id: str                       # e.g. "102R"
    diagnosis: str                    # DME / CNVM / PCV
    drug: str                         # anti-VEGF agent
    age: int
    gender: str
    pre_va: float
    pre_cst: float
    va: float                         # post-treatment (label)
    cst: float                        # post-treatment (label)
    continue_injection: int           # target
    pre_dir: Optional[str]            # Pre Injection OCT Images dir
    post_dir: Optional[str]
    biomarkers: dict = field(default_factory=dict)  # image-level OR-aggregated {IRF:0/1,...}

    @property
    def biomarker_vector(self) -> np.ndarray:
        return np.array([self.biomarkers.get(b, 0) for b in BIOMARKERS], dtype=float)


def _coerce_float(x) -> float:
    """Handle string anomalies (e.g. 'NLP') in VA columns."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_case_table(path: str = CASE_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df


def load_pic_table(path: str = PIC_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for c in ("patient ID", "injection", "image name"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def _eye_image_dirs(eye_id: str) -> tuple[Optional[str], Optional[str]]:
    base = os.path.join(TRAIN_IMG_ROOT, eye_id)
    pre = os.path.join(base, "Pre Injection OCT Images")
    post = os.path.join(base, "Post Injection OCT Images")
    return (pre if os.path.isdir(pre) else None, post if os.path.isdir(post) else None)


def build_eye_records(
    case_df: Optional[pd.DataFrame] = None,
    pic_df: Optional[pd.DataFrame] = None,
) -> list[EyeRecord]:
    """One EyeRecord per row of the case table, with image-level biomarkers
    OR-aggregated over the eye's PRE-injection images (decision uses pre-scan)."""
    case_df = load_case_table() if case_df is None else case_df
    pic_df = load_pic_table() if pic_df is None else pic_df

    pre_pics = pic_df[pic_df["injection"].str.lower().str.startswith("pre")]
    bm_by_eye = (
        pre_pics.groupby("patient ID")[BIOMARKERS].max().astype(int).to_dict("index")
    )

    records: list[EyeRecord] = []
    for _, r in case_df.iterrows():
        eye_id = str(r["patient ID"]).strip()
        pre_dir, post_dir = _eye_image_dirs(eye_id)
        records.append(
            EyeRecord(
                eye_id=eye_id,
                diagnosis=str(r["diagnosis"]).strip(),
                drug=str(r["anti-VEGF"]).strip(),
                age=int(_coerce_float(r["age"])) if not pd.isna(_coerce_float(r["age"])) else -1,
                gender=str(r["gender"]).strip(),
                pre_va=_coerce_float(r["preVA"]),
                pre_cst=_coerce_float(r["preCST"]),
                va=_coerce_float(r["VA"]),
                cst=_coerce_float(r["CST"]),
                continue_injection=int(_coerce_float(r["continue injection"])),
                pre_dir=pre_dir,
                post_dir=post_dir,
                biomarkers=bm_by_eye.get(eye_id, {}),
            )
        )
    return records


def representative_pre_bscan(rec: EyeRecord) -> Optional[Image.Image]:
    """Return the central pre-injection B-scan, macular-cropped, or None.

    Note: APTOS B-scans are exported as already-split OCT panels in many eyes; if
    the image is wider than OCT_X_START we treat it as a fundus+OCT composite and
    crop, otherwise we use it as-is.
    """
    if not rec.pre_dir:
        return None
    files = sorted(
        f for f in os.listdir(rec.pre_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not files:
        return None
    imgs = [Image.open(os.path.join(rec.pre_dir, f)).convert("RGB") for f in files]
    # Composite (fundus+OCT) layout -> use row-sum center detection; else middle.
    composite = imgs[0].size[0] > OCT_X_START + 50
    if composite:
        # pick the panel whose detected center row is most central (sharpest fundus)
        idx = len(imgs) // 2
        img = imgs[idx]
        return extract_macular_oct_crop(img)
    return imgs[len(imgs) // 2]


# --- splits ----------------------------------------------------------------
def _strata(records: list[EyeRecord]) -> np.ndarray:
    return np.array([f"{r.diagnosis}|{r.continue_injection}" for r in records])


def stratified_split(
    records: list[EyeRecord], test_size: float = 0.15, seed: int = 42
) -> tuple[list[EyeRecord], list[EyeRecord]]:
    """Eye-level stratified split by (diagnosis x continue-injection).

    numpy-only (no sklearn) so inference runs in any conda env. Within each
    stratum, a deterministic shuffle assigns ceil(test_size*n) eyes to test.
    """
    strata = _strata(records)
    rng = np.random.default_rng(seed)
    test_idx: list[int] = []
    for s in np.unique(strata):
        members = np.where(strata == s)[0]
        rng.shuffle(members)
        n_test = int(np.ceil(len(members) * test_size)) if len(members) > 1 else 0
        test_idx.extend(members[:n_test].tolist())
    test_set = set(test_idx)
    tr = [records[i] for i in range(len(records)) if i not in test_set]
    te = [records[i] for i in range(len(records)) if i in test_set]
    return tr, te


def make_kfold_splits(records: list[EyeRecord], n_splits: int = 5, seed: int = 42):
    """Eye-level stratified k-fold (statistical-power robustness check), numpy-only.

    NOTE: NOT used by default. The default pipeline calls stratified_split().
    Uncomment the call site in the notebook to enable k-fold reporting.
    """
    strata = _strata(records)
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(records), dtype=int)
    for s in np.unique(strata):
        members = np.where(strata == s)[0]
        rng.shuffle(members)
        fold_of[members] = np.arange(len(members)) % n_splits
    folds = []
    for k in range(n_splits):
        te_idx = np.where(fold_of == k)[0]
        tr_idx = np.where(fold_of != k)[0]
        folds.append(([records[i] for i in tr_idx], [records[i] for i in te_idx]))
    return folds


if __name__ == "__main__":
    recs = build_eye_records()
    print(f"loaded {len(recs)} eyes")
    print("example:", recs[0].eye_id, recs[0].diagnosis, recs[0].continue_injection,
          "biomarkers=", recs[0].biomarkers, "pre_dir?", bool(recs[0].pre_dir))
    tr, te = stratified_split(recs)
    print(f"split train={len(tr)} test={len(te)}")
    # k-fold is available but disabled by default:
    # folds = make_kfold_splits(recs); print(f"{len(folds)} folds")
