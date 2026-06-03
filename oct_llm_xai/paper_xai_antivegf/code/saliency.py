"""Saliency fidelity metrics (E3) and a keyword-conditioned GradCAM adapter.

Metric cores (unit-testable, no weights):
  fluid_energy_ratio          : fraction of saliency mass inside the fluid region.
  label_conditioned_concentration : saliency concentration in a labelled region
                                    vs. background (ratio of mean intensities).

GradCAM/VL-saliency extraction itself is adapted from the existing prototypes
(experiments/_test_vl_saliency.py L80-189, _test_exp3.py L298-343) and is wired
lazily because it needs torch + a loaded backend.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _norm_map(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    m = np.clip(m, 0, None)
    s = m.sum()
    return m / s if s > 0 else m


def fluid_energy_ratio(saliency_map: np.ndarray, fluid_mask: np.ndarray) -> float:
    """Fraction of (normalized) saliency mass falling inside `fluid_mask` (bool/0-1).

    Mask is resized by nearest-neighbour if its shape differs from the map.
    """
    sal = _norm_map(saliency_map)
    mask = _resize_mask(fluid_mask, sal.shape)
    return float((sal * (mask > 0)).sum())


def label_conditioned_concentration(
    saliency_map: np.ndarray, region_mask: np.ndarray, eps: float = 1e-8
) -> float:
    """Mean saliency inside the region / mean saliency outside (>1 == concentrated)."""
    sal = np.asarray(saliency_map, dtype=np.float64)
    mask = _resize_mask(region_mask, sal.shape) > 0
    inside = sal[mask]
    outside = sal[~mask]
    if inside.size == 0 or outside.size == 0:
        return float("nan")
    return float((inside.mean() + eps) / (outside.mean() + eps))


def _resize_mask(mask: np.ndarray, shape) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.shape == tuple(shape):
        return mask
    # nearest-neighbour index resampling (no scipy dependency)
    rows = (np.linspace(0, mask.shape[0] - 1, shape[0])).round().astype(int)
    cols = (np.linspace(0, mask.shape[1] - 1, shape[1])).round().astype(int)
    return mask[np.ix_(rows, cols)]


# ---------------------------------------------------------------------------
# Keyword-conditioned GradCAM adapter (lazy; needs torch + loaded backend).
# Adapts experiments/_test_vl_saliency.py.
# ---------------------------------------------------------------------------
def keyword_gradcam(backend, image, keyword: str, target_layer=None) -> np.ndarray:  # pragma: no cover
    """Compute a keyword-conditioned GradCAM map on the vision encoder.

    Thin adapter over the existing prototype: hooks the vision encoder's last
    conv block and back-props the `softmax_logits` of `keyword`. Returns an [H, W]
    saliency map. Requires a loaded backend exposing the vision encoder.
    """
    raise NotImplementedError(
        "keyword_gradcam: wire to experiments/_test_vl_saliency.py (L80-189) with a "
        "loaded backend; metric cores fluid_energy_ratio/label_conditioned_"
        "concentration are usable independently of extraction."
    )


if __name__ == "__main__":
    sal = np.zeros((8, 8))
    sal[2:4, 2:4] = 1.0           # all energy in a 2x2 block
    mask = np.zeros((8, 8))
    mask[2:4, 2:4] = 1
    print("fluid_energy_ratio:", fluid_energy_ratio(sal, mask))          # 1.0
    print("concentration:", round(label_conditioned_concentration(sal, mask), 3))  # high
