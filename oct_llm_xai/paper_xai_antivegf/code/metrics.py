"""Scoring and statistics for the anti-VEGF VLM spectrum study.

Adopts the official APTOS-2021 scoring (CI-AUC, VA/CST tolerance, biomarker-AUC)
so results are directly comparable to BlueSky (1st place) and the repo CNN refs.

- score_va / score_cst_tolerance : official tolerance metrics (see
  data_response/experiment/aptos_scoring_formula.md). score_cst_tolerance mirrors
  data_response/experiment/task2_v2/prepare.py:120.
- compute_auc                     : per-class + mean AUC.
- bootstrap_ci                    : 95% CI for any metric (1000 resamples).
- delong_test                     : DeLong test for a difference between two AUCs.
- jonckheere_terpstra             : trend test for monotone increase across tiers (H1/H2/H5).
- aggregate_subtasks              : per-subtask reporting dict.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Official APTOS-2021 tolerance metrics
# ---------------------------------------------------------------------------
def score_cst_tolerance(y_true, y_pred, tolerance: float = 0.075) -> float:
    """CST ±7.5% relative-tolerance score (fraction within tolerance).

    Ported from data_response/experiment/task2_v2/prepare.py:120.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    safe = np.where(yt == 0, np.finfo(np.float64).eps, yt)
    return float(np.mean(np.abs(yp - yt) / np.abs(safe) <= tolerance))


def score_va(y_true, y_pred) -> float:
    """VA tolerance score (NEW).

    Per aptos_scoring_formula.md:22-28:
      - y < 1 : hit if |y_hat - y| <= 0.05            (absolute tolerance)
      - y >= 1: hit if |y_hat - y| / y <= 0.075        (relative 7.5%)
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    abs_hit = np.abs(yp - yt) <= 0.05
    rel_hit = np.abs(yp - yt) / np.where(yt == 0, np.finfo(np.float64).eps, yt) <= 0.075
    hits = np.where(yt < 1, abs_hit, rel_hit)
    return float(np.mean(hits))


# ---------------------------------------------------------------------------
# AUC (classification subtasks: CI, IRF/SRF/PED/HRF)
# ---------------------------------------------------------------------------
def compute_auc(labels, scores) -> dict:
    """Binary or multi-label AUC.

    labels/scores: shape (N,) for a single task or (N, C) for multi-label.
    Returns {"class_k": auc, ..., "mean_auc": mean}. Classes with a single
    present label are skipped (AUC undefined).
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.ndim == 1:
        labels = labels[:, None]
        scores = scores[:, None]
    out: dict[str, float] = {}
    per_class = []
    for k in range(labels.shape[1]):
        yk = labels[:, k]
        if len(np.unique(yk)) < 2:
            out[f"class_{k}"] = float("nan")
            continue
        auc = float(roc_auc_score(yk, scores[:, k]))
        out[f"class_{k}"] = auc
        per_class.append(auc)
    out["mean_auc"] = float(np.mean(per_class)) if per_class else float("nan")
    return out


# ---------------------------------------------------------------------------
# Bootstrap CI for any metric
# ---------------------------------------------------------------------------
def bootstrap_ci(
    y_true,
    y_pred,
    metric: Callable,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (point_estimate, lo, hi) for `metric(y_true, y_pred)`."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = len(yt)
    rng = np.random.default_rng(seed)
    point = float(metric(yt, yp))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boots[b] = metric(yt[idx], yp[idx])
        except ValueError:  # e.g. AUC on a degenerate resample
            boots[b] = np.nan
    lo = float(np.nanpercentile(boots, 100 * alpha / 2))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


# ---------------------------------------------------------------------------
# DeLong test for two correlated/independent AUCs
# ---------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranked = x[order]
    n = len(x)
    mid = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j] == ranked[i]:
            j += 1
        mid[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = mid
    return out


def _fast_delong(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """Single-AUC DeLong variance (Sun & Xu, 2014). Returns (auc, variance)."""
    m, n = len(pos), len(neg)
    tx = _midrank(pos)
    ty = _midrank(neg)
    txy = _midrank(np.concatenate([pos, neg]))
    auc = (np.sum(txy[:m]) - m * (m + 1) / 2) / (m * n)
    v01 = (txy[:m] - tx) / n
    v10 = 1.0 - (txy[m:] - ty) / m
    s01 = np.var(v01, ddof=1) / m
    s10 = np.var(v10, ddof=1) / n
    return float(auc), float(s01 + s10)


def delong_test(y_true, score_a, score_b) -> dict:
    """Two-sided test for AUC(a) != AUC(b) on the SAME labels.

    Treats the two scorers as independent (sufficient for tier comparison where
    models differ). Returns {auc_a, auc_b, z, p}.
    """
    from scipy.stats import norm

    y = np.asarray(y_true)
    pos = y == 1
    auc_a, var_a = _fast_delong(np.asarray(score_a)[pos], np.asarray(score_a)[~pos])
    auc_b, var_b = _fast_delong(np.asarray(score_b)[pos], np.asarray(score_b)[~pos])
    se = np.sqrt(var_a + var_b)
    z = (auc_a - auc_b) / se if se > 0 else 0.0
    p = float(2 * (1 - norm.cdf(abs(z))))
    return {"auc_a": auc_a, "auc_b": auc_b, "z": float(z), "p": p}


# ---------------------------------------------------------------------------
# Jonckheere-Terpstra trend test (H1/H2/H5 monotone increase across tiers)
# ---------------------------------------------------------------------------
def jonckheere_terpstra(groups: Sequence[Sequence[float]]) -> dict:
    """Test ordered alternative group_1 <= group_2 <= ... (one-sided, increasing).

    `groups` is a list of samples per tier, in ascending tier order.
    Returns {J, z, p} with a normal approximation.
    """
    from scipy.stats import norm

    arrs = [np.asarray(g, dtype=np.float64) for g in groups]
    J = 0.0
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            xi, xj = arrs[i][:, None], arrs[j][None, :]
            J += np.sum(xi < xj) + 0.5 * np.sum(xi == xj)
    ns = np.array([len(a) for a in arrs], dtype=np.float64)
    N = ns.sum()
    mu = (N**2 - np.sum(ns**2)) / 4.0
    var = (N**2 * (2 * N + 3) - np.sum(ns**2 * (2 * ns + 3))) / 72.0
    z = (J - mu) / np.sqrt(var) if var > 0 else 0.0
    p = float(1 - norm.cdf(z))  # one-sided (increasing)
    return {"J": float(J), "z": float(z), "p": p}


# ---------------------------------------------------------------------------
# Per-subtask reporting (we report individual subtasks, NOT the 14-subtask mean)
# ---------------------------------------------------------------------------
def aggregate_subtasks(records: dict) -> dict:
    """Assemble the individual-subtask report.

    `records` keys (any subset):
      ci      -> (y_true, y_score)            : CI-AUC
      va      -> (y_true, y_pred)             : VA tolerance
      cst     -> (y_true, y_pred)             : CST tolerance
      biomarkers -> (labels[N,4], scores[N,4]): per-class + mean AUC
    """
    out: dict = {}
    if "ci" in records:
        yt, ys = records["ci"]
        pt, lo, hi = bootstrap_ci(yt, ys, lambda a, b: roc_auc_score(a, b))
        out["CI_AUC"] = {"value": pt, "ci95": [lo, hi]}
    if "va" in records:
        yt, yp = records["va"]
        pt, lo, hi = bootstrap_ci(yt, yp, score_va)
        out["VA_tol"] = {"value": pt, "ci95": [lo, hi]}
    if "cst" in records:
        yt, yp = records["cst"]
        pt, lo, hi = bootstrap_ci(yt, yp, score_cst_tolerance)
        out["CST_tol"] = {"value": pt, "ci95": [lo, hi]}
    if "biomarkers" in records:
        labels, scores = records["biomarkers"]
        out["biomarker_AUC"] = compute_auc(labels, scores)
    return out


if __name__ == "__main__":
    # Self-check against aptos_scoring_formula.md reference snippet.
    yt = np.array([0.3, 0.8, 1.2, 0.5])
    yp = np.array([0.32, 0.70, 1.25, 0.5])
    print("score_va  =", score_va(yt, yp))            # 0.3->hit, 0.8->miss(0.1>0.05), 1.2->hit, 0.5->hit
    print("score_cst =", score_cst_tolerance([300, 400], [310, 360]))  # 1/2
