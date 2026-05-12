"""
prepare.py — APTOS 2021 autoresearch read-only harness.

Provides constants, data loading, train/val split, evaluation metrics,
and MIL bag construction for the APTOS-2021 OCT biomarker dataset.
Imported by train.py; never modified at runtime.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

TIME_BUDGET: int = 900  # seconds (15 min — allows ~6 epochs for Swin-Base)
DATA_ROOT: str = (
    "/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/data_response"
    "/anti-vegf-dataset/APTOS-2021/Final Datasets"
)
SEED: int = 42
NUM_CLASSES: int = 4
CLASS_NAMES: List[str] = ["IRF", "SRF", "PED", "HRF"]

# ---------------------------------------------------------------------------
# 2. Data loading
# ---------------------------------------------------------------------------


def load_pic_csv() -> pd.DataFrame:
    """Load train_anno_pic.csv.

    Returns:
        DataFrame with columns:
            patient ID, injection, image name, IRF, SRF, PED, HRF
    """
    path = os.path.join(DATA_ROOT, "train_anno_pic.csv")
    df = pd.read_csv(path)
    # Normalise column names to strip leading/trailing whitespace
    df.columns = df.columns.str.strip()
    return df


def load_case_csv() -> pd.DataFrame:
    """Load train_anno_case.csv.

    Returns:
        DataFrame with columns:
            patient ID, gender, age, diagnosis, anti-VEGF,
            preVA, preCST, VA, CST, continue injection
    """
    path = os.path.join(DATA_ROOT, "train_anno_case.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def _title_case_injection(injection: str) -> str:
    """Convert CSV injection value to folder-name casing.

    Examples:
        'Pre injection'  -> 'Pre Injection'
        'Post injection' -> 'Post Injection'
    """
    return " ".join(word.capitalize() for word in injection.strip().split())


def get_image_path(patient_id: str, injection: str, image_name: str) -> str:
    """Build the absolute path to an OCT image file.

    Path template:
        {DATA_ROOT}/Training Set/Training Set/{patient_id}/{Injection} OCT Images/{image_name}.jpg

    The *injection* parameter from the CSV (e.g. 'Pre injection') is
    title-cased to match the folder name ('Pre Injection').

    Args:
        patient_id: Patient identifier (e.g. '2L').
        injection: Injection string from CSV (e.g. 'Pre injection').
        image_name: Image name from CSV (e.g. '10').

    Returns:
        Absolute path string for the image.
    """
    injection_folder = f"{_title_case_injection(injection)} OCT Images"
    return os.path.join(
        DATA_ROOT,
        "Training Set",
        "Training Set",
        str(patient_id),
        injection_folder,
        f"{image_name}.jpg",
    )


# ---------------------------------------------------------------------------
# 3. Train / Val split (patient-level, stratified by diagnosis)
# ---------------------------------------------------------------------------


def get_patient_split(
    pic_df: pd.DataFrame,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Patient-level stratified train/val split.

    Stratification key is the *diagnosis* column from the case CSV
    (CNVM / DME / PCV), ensuring each diagnosis group is proportionally
    represented in both splits.

    Args:
        pic_df: DataFrame returned by ``load_pic_csv``.
        val_ratio: Fraction of patients allocated to validation.
        seed: Random seed for reproducibility.

    Returns:
        (train_patient_ids, val_patient_ids) — each a sorted list of
        patient ID strings.
    """
    case_df = load_case_csv()

    # Build patient -> diagnosis mapping
    diag_map: Dict[str, str] = dict(
        zip(case_df["patient ID"].astype(str), case_df["diagnosis"].astype(str))
    )

    # Unique patients that appear in *both* DataFrames
    pic_patients = set(pic_df["patient ID"].astype(str).unique())
    case_patients = set(diag_map.keys())
    patients = sorted(pic_patients & case_patients)

    # Group patients by diagnosis for stratified sampling
    rng = np.random.RandomState(seed)
    train_ids: List[str] = []
    val_ids: List[str] = []

    # Patients not found in case_csv go entirely to train (cannot stratify)
    unmapped = sorted(pic_patients - case_patients)
    train_ids.extend(unmapped)

    diagnosis_groups: Dict[str, List[str]] = {}
    for pid in patients:
        d = diag_map[pid]
        diagnosis_groups.setdefault(d, []).append(pid)

    for _diag, group in diagnosis_groups.items():
        group = sorted(group)
        n_val = max(1, round(len(group) * val_ratio))
        perm = rng.permutation(len(group))
        val_indices = perm[:n_val]
        train_indices = perm[n_val:]
        val_ids.extend(group[i] for i in val_indices)
        train_ids.extend(group[i] for i in train_indices)

    return sorted(train_ids), sorted(val_ids)


# ---------------------------------------------------------------------------
# 4. Evaluation metrics
# ---------------------------------------------------------------------------


def compute_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 4,
) -> Tuple[Dict[str, float], float]:
    """Compute per-class AUC and mean AUC.

    Uses a simple trapezoidal approximation over all operating points
    (no sklearn dependency).

    Args:
        y_true: Ground-truth binary labels, shape (N, num_classes).
        y_pred: Predicted probabilities, shape (N, num_classes).
        num_classes: Number of classes.

    Returns:
        (per_class_auc_dict, mean_auc)
        e.g. ({'IRF': 0.93, 'SRF': 0.91, 'PED': 0.88, 'HRF': 0.97}, 0.9225)
    """
    per_class: Dict[str, float] = {}

    for c in range(num_classes):
        yt = y_true[:, c]
        yp = y_pred[:, c]

        # Skip classes with only one label value
        if yt.sum() == 0 or yt.sum() == len(yt):
            per_class[CLASS_NAMES[c]] = float("nan")
            continue

        # Sort by predicted score descending
        order = np.argsort(-yp)
        yt_sorted = yt[order]

        n_pos = int(yt.sum())
        n_neg = len(yt) - n_pos

        tpr_prev = 0.0
        fpr_prev = 0.0
        auc_val = 0.0
        tp = 0
        fp = 0

        for i in range(len(yt_sorted)):
            if yt_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc_val += 0.5 * (fpr - fpr_prev) * (tpr + tpr_prev)
            tpr_prev = tpr
            fpr_prev = fpr

        per_class[CLASS_NAMES[c]] = round(auc_val, 6)

    valid_aucs = [v for v in per_class.values() if not np.isnan(v)]
    mean_auc = round(float(np.mean(valid_aucs)), 6) if valid_aucs else float("nan")

    return per_class, mean_auc


def compute_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Compute Mean Absolute Percentage Error.

    MAPE = (1/N) * sum(|y_true - y_pred| / max(|y_true|, epsilon))

    Args:
        y_true: Ground-truth values, any shape (flattened internally).
        y_pred: Predicted values, same shape as y_true.
        epsilon: Small constant to avoid division by zero.

    Returns:
        MAPE as a float (not percentage).
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    denom = np.maximum(np.abs(yt), epsilon)
    return float(np.mean(np.abs(yt - yp) / denom))


# ---------------------------------------------------------------------------
# 5. MIL Bag construction
# ---------------------------------------------------------------------------


def build_mil_bags(
    pic_df: pd.DataFrame,
    patient_ids: List[str],
) -> List[Dict]:
    """Construct MIL bags from the annotation DataFrame.

    Two stages:

    **Stage 1** — (patient, injection) level bags.
    All images belonging to the same (patient, injection) pair are grouped
    into one bag.  The bag label is the element-wise *max* of the
    per-image binary labels (presence if any image shows the biomarker).

    **Stage 2** — image level bags.
    Each individual image forms its own bag with its own label.

    Images whose file path does not exist on disk are silently excluded.

    Args:
        pic_df: DataFrame returned by ``load_pic_csv``.
        patient_ids: List of patient IDs to include (e.g. train split).

    Returns:
        List of bag dictionaries, each containing:

        - ``bag_id`` (str): unique identifier
        - ``stage`` (int): 1 or 2
        - ``patient_id`` (str)
        - ``injection`` (str): original injection string from CSV
        - ``image_paths`` (List[str]): absolute paths to existing images
        - ``labels`` (np.ndarray): shape (4,), binary labels for IRF/SRF/PED/HRF
    """
    patient_set = set(str(p) for p in patient_ids)
    df = pic_df[pic_df["patient ID"].astype(str).isin(patient_set)].copy()
    df["patient ID"] = df["patient ID"].astype(str)

    label_cols = CLASS_NAMES  # ["IRF", "SRF", "PED", "HRF"]

    bags: List[Dict] = []

    # ---- Stage 1: (patient, injection) level --------------------------------
    for (pid, inj), group in df.groupby(["patient ID", "injection"]):
        image_paths: List[str] = []
        for _, row in group.iterrows():
            p = get_image_path(pid, inj, row["image name"])
            if os.path.isfile(p):
                image_paths.append(p)

        if not image_paths:
            continue

        # Bag label = max across all images in the bag
        bag_labels = group[label_cols].values.max(axis=0).astype(np.float32)

        bags.append(
            {
                "bag_id": f"stage1_{pid}_{inj.replace(' ', '_')}",
                "stage": 1,
                "patient_id": pid,
                "injection": inj,
                "image_paths": image_paths,
                "labels": bag_labels,
            }
        )

    # ---- Stage 2: image level -----------------------------------------------
    for _, row in df.iterrows():
        pid = row["patient ID"]
        inj = row["injection"]
        img_name = row["image name"]
        p = get_image_path(pid, inj, img_name)

        if not os.path.isfile(p):
            continue

        img_labels = row[label_cols].values.astype(np.float32)

        bags.append(
            {
                "bag_id": f"stage2_{pid}_{inj.replace(' ', '_')}_{img_name}",
                "stage": 2,
                "patient_id": pid,
                "injection": inj,
                "image_paths": [p],
                "labels": img_labels,
            }
        )

    return bags
