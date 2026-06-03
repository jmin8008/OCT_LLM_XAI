"""Perturbation-based ROCO (Remove-and-Observe Causal Outcome) — CONDITIONAL / NOT IMPLEMENTED.

Status (per EXPERIMENTAL_PROTOCOL.md §4.4 and the approved plan): E8 is **deferred**.
ROCO requires a fluid-region mask to occlude, but APTOS-2021 provides only
image-level biomarker labels (no segmentation masks). Estimating the region by a
heuristic would make the causal-specificity metric self-referential and
unreliable, so this experiment is gated on obtaining **clinician-provided masks**.

This module documents the intended interface so it can be filled in when masks
become available; calling it raises NotImplementedError by design.

Intended behaviour (once masks exist):
  delta_logit_ci = logit_CI(image) - logit_CI(image with fluid region occluded)
  causal_specificity = delta_logit_ci(fluid region) - delta_logit_ci(control region)
  -> compare across tiers (H6).
"""
from __future__ import annotations

from typing import Optional


def occlude_region(image, mask, mode: str = "gaussian"):  # pragma: no cover
    raise NotImplementedError("ROCO is deferred until clinician masks are available.")


def causal_specificity(
    backend,
    image,
    fluid_mask,
    control_mask: Optional[object] = None,
    prompt: Optional[str] = None,
) -> float:  # pragma: no cover
    """ΔlogitCI(fluid occlusion) − ΔlogitCI(control occlusion). NOT IMPLEMENTED."""
    raise NotImplementedError(
        "Perturbation-ROCO (E8) is a conditional experiment: requires clinician-"
        "provided fluid masks. See module docstring and EXPERIMENTAL_PROTOCOL.md §4.4."
    )


MASK_AVAILABLE = False  # flip to True and implement above when masks are obtained.
