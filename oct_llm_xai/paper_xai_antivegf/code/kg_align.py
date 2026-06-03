"""Neuro-symbolic KG alignment metrics (E7) — Wang 2025 (Sensors 25:6879).

Two axes, mirroring Wang's headline interpretability numbers:
  (a) Text-KG alignment   : does the VLM's stated reasoning follow a guideline
                            rule? -> corresponds to Wang's ">85% rule-supported
                            reasoning".
  (b) Attn-KG consistency : does the model's attention (rollout) concentrate on
                            the KG decision-driver biomarker regions? ->
                            corresponds to Wang's ">90% biomarker-citation".

Inputs are the prediction rows produced by infer.run_inference plus a kg.GuidelineKG.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

WANG_TEXT_BASELINE = 0.85   # Wang 2025: >85% rule-supported reasoning
WANG_ATTN_BASELINE = 0.90   # Wang 2025: >90% biomarker-citation accuracy

_DECISION_TO_CI = {"continue": 1, "stop": 0}


def text_kg_aligned(pred_row: dict, kg) -> Optional[bool]:
    """Is one prediction's stated reasoning KG-consistent?

    Uses the biomarkers the VLM itself reported (`bm_pred`) -> KG forward_chain ->
    expected decision; aligned if the VLM's CI answer (`ci_pred`) matches it.
    Returns None when unresolvable (uncertain answer or missing/NaN biomarkers).
    """
    ci = pred_row.get("ci_pred")
    if ci not in (0, 1):
        return None
    bm = pred_row.get("bm_pred")
    if not bm or any(
        (v is None or (isinstance(v, float) and math.isnan(v))) for v in bm.values()
    ):
        return None
    fc = kg.forward_chain({k: int(v) for k, v in bm.items()})
    expected = _DECISION_TO_CI.get(fc["decision"])
    if expected is None:          # case_dependent / uncertain -> not scored
        return None
    return ci == expected


def text_kg_alignment_rate(preds: Sequence[dict], kg) -> dict:
    """Fraction of resolvable predictions whose reasoning follows the KG."""
    flags = [text_kg_aligned(r, kg) for r in preds]
    resolved = [f for f in flags if f is not None]
    rate = (sum(resolved) / len(resolved)) if resolved else float("nan")
    return {
        "text_kg_alignment": rate,
        "n_resolved": len(resolved),
        "n_total": len(preds),
        "meets_wang_baseline": (not math.isnan(rate)) and rate >= WANG_TEXT_BASELINE,
    }


def attn_kg_consistency_one(
    attn_map, decision: str, kg, biomarker_masks: Optional[dict]
) -> Optional[float]:
    """Attention mass on the KG decision-driver biomarker regions.

    attn_map        : 2-D rollout map (rollout.image_attention_map).
    decision        : the model's decision ('continue'/'stop').
    biomarker_masks : {biomarker_id: 2-D bool mask}. Without masks (no clinician
                      segmentation) returns None — kept honest, not fabricated.
    """
    if not biomarker_masks:
        return None
    from saliency import fluid_energy_ratio

    drivers = kg.decision_drivers(decision)
    masks = [biomarker_masks[b] for b in drivers if b in biomarker_masks]
    if not masks:
        return None
    import numpy as np

    union = np.zeros_like(np.asarray(masks[0], dtype=float))
    for m in masks:
        union = np.maximum(union, np.asarray(m, dtype=float))
    return fluid_energy_ratio(attn_map, union)


def attn_kg_consistency_rate(
    rows: Sequence[dict], kg, threshold: float = 0.5
) -> dict:
    """Aggregate Attn-KG consistency. Each row needs 'attn_map','decision',
    'biomarker_masks'. Rows without masks are skipped (reported as n_skipped)."""
    vals = []
    skipped = 0
    for r in rows:
        v = attn_kg_consistency_one(
            r.get("attn_map"), r.get("decision"), kg, r.get("biomarker_masks")
        )
        if v is None:
            skipped += 1
        else:
            vals.append(v)
    import numpy as np

    mean = float(np.mean(vals)) if vals else float("nan")
    frac_above = float(np.mean([v >= threshold for v in vals])) if vals else float("nan")
    return {
        "attn_kg_consistency_mean": mean,
        "frac_above_threshold": frac_above,
        "n_scored": len(vals),
        "n_skipped_no_mask": skipped,
        "note": "no clinician masks available -> Attn-KG is conditional (see ROCO/E8)"
        if not vals
        else "",
    }


if __name__ == "__main__":
    import kg as kg_mod

    k = kg_mod.GuidelineKG.load_default()
    # aligned: VLM reports SRF present and answers continue
    r1 = {"ci_pred": 1, "bm_pred": {"IRF": 0, "SRF": 1, "PED": 0, "HRF": 1}}
    # misaligned: VLM reports dry but answers continue
    r2 = {"ci_pred": 1, "bm_pred": {"IRF": 0, "SRF": 0, "PED": 0, "HRF": 0}}
    # unresolvable: uncertain
    r3 = {"ci_pred": "uncertain", "bm_pred": {"IRF": 1, "SRF": 0, "PED": 0, "HRF": 0}}
    print("r1 aligned:", text_kg_aligned(r1, k))
    print("r2 aligned:", text_kg_aligned(r2, k))
    print("r3 aligned:", text_kg_aligned(r3, k))
    print(text_kg_alignment_rate([r1, r2, r3], k))
