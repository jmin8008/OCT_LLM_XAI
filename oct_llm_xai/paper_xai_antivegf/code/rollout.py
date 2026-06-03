"""Cross-Attention Rollout for VLM text->image grounding (E3b).

Implements attention rollout (Abnar & Zuidema, 2020): average heads, add the
residual identity, row-normalize, and multiply across layers. The rolled-out
attention from a chosen OUTPUT token (e.g. the token that says "continue" or
"fluid") onto the IMAGE tokens is reshaped to a spatial grid.

Tier-3 RetinaVLM exposes the full attention stack via
models.RetinaVLMBackend.attention -> mini_gpt4.attention (L392); Tier-1/2 expose
HF `output_attentions`. This module operates on the raw tensors so it is backend-
agnostic and unit-testable without weights.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().to("cpu").float().numpy()
    return np.asarray(x, dtype=np.float64)


def attention_rollout(attentions, add_residual: bool = True) -> np.ndarray:
    """Roll out a stack of attention matrices into a single [T, T] matrix.

    attentions: array-like [L, H, T, T] or [L, T, T] (heads already averaged).
    Returns the [T, T] rolled-out attention (row-normalized).
    """
    a = _to_numpy(attentions)
    if a.ndim == 4:          # [L, H, T, T] -> average heads
        a = a.mean(axis=1)
    if a.ndim != 3:
        raise ValueError(f"expected [L,H,T,T] or [L,T,T], got shape {a.shape}")
    L, T, _ = a.shape
    eye = np.eye(T)
    rolled = np.eye(T)
    for l in range(L):
        layer = a[l]
        if add_residual:
            layer = 0.5 * layer + 0.5 * eye
        layer = layer / np.clip(layer.sum(axis=-1, keepdims=True), 1e-12, None)
        rolled = layer @ rolled
    return rolled


def image_attention_map(
    attentions,
    query_token_idx: int,
    image_token_slice: slice,
    grid_hw: Optional[tuple] = None,
    add_residual: bool = True,
) -> np.ndarray:
    """Rolled-out attention from one output token onto the image tokens.

    query_token_idx   : index of the output token of interest (e.g. "continue").
    image_token_slice : slice selecting image tokens along the key axis.
    grid_hw           : (H, W) to reshape; if None, infer a near-square grid.
    Returns a 2-D [H, W] map normalized to sum 1.
    """
    rolled = attention_rollout(attentions, add_residual=add_residual)
    row = rolled[query_token_idx, image_token_slice]
    s = row.sum()
    if s > 0:
        row = row / s
    n = row.shape[0]
    if grid_hw is None:
        side = int(round(math.sqrt(n)))
        grid_hw = (side, side)
        if grid_hw[0] * grid_hw[1] != n:  # pad to fit
            pad = grid_hw[0] * grid_hw[1] - n
            row = np.concatenate([row, np.zeros(max(pad, 0))])[: grid_hw[0] * grid_hw[1]]
    return row.reshape(grid_hw)


def rollout_from_retinavlm(attn_bundle, query_token_idx: int, grid_hw=None) -> np.ndarray:
    """Adapter for models.RetinaVLMBackend.attention output bundle:
    (samples, tokens, subsequence_indices, sequence_attentions, image_attention)

    sequence_attentions shape: [L, B, H, T, T] — squeeze batch dim first.
    subsequence_indices: [(pre_len, img_len, post_len)] — locates image tokens.
    query_token_idx: index in the POST-image part (text+generated tokens).
      Use -1 for the last generated token, or a specific text token index.
    """
    _, _, subsequence_indices, sequence_attentions, _ = attn_bundle
    attn = _to_numpy(sequence_attentions)      # [L, B, H, T, T]
    if attn.ndim == 5:
        attn = attn[:, 0, :, :, :]            # squeeze batch → [L, H, T, T]

    pre_len, img_len, _ = subsequence_indices[0]
    img_slice = slice(int(pre_len), int(pre_len) + int(img_len))

    # query_token_idx relative to the end of the sequence (e.g. -1 = last output token)
    T = attn.shape[-1]
    if query_token_idx < 0:
        q_idx = T + query_token_idx
    else:
        q_idx = int(pre_len) + int(img_len) + query_token_idx

    # Build spatial grid size from img_len
    import math
    side = int(round(math.sqrt(img_len)))
    if grid_hw is None:
        grid_hw = (side, side)

    return image_attention_map(attn, q_idx, img_slice, grid_hw=grid_hw)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    L, H, T = 4, 8, 20          # 16 image tokens + 4 text tokens
    attn = rng.random((L, H, T, T))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    m = image_attention_map(attn, query_token_idx=18, image_token_slice=slice(0, 16))
    print("map shape:", m.shape, "sum:", round(float(m.sum()), 4))
