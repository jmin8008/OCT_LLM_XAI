"""Arm C core: differentiable attention-rollout KL loss against a fluid mask.

The arm-C training signal is  L = L_LM + lambda * KL( q_fluid || p_attn ), where
  p_attn  = attention-rollout distribution from the answer tokens onto the IMAGE
            tokens, reshaped to the model's image-token grid (RetinaVLM = 6x6).
  q_fluid = the 12x12 Claude-vision fluid mask AVERAGE-POOLED down to that same
            grid and renormalised to a probability distribution.

Everything here is pure torch and stays in the autograd graph (no .detach / numpy),
so KL backprops into the LoRA q_proj/v_proj that produce the attention queries.
This mirrors rollout.py's numpy rollout exactly (Abnar & Zuidema 2020: head-mean,
0.5*A+0.5*I residual, row-normalise, multiply across layers) but differentiably.

Backend-agnostic: it takes raw attention tensors + an image-token slice + a target
grid, so the same loss drives RetinaVLM (6x6), LLaVA-Med, and Qwen (their own grids).
"""
from __future__ import annotations
from typing import Sequence
import torch
import torch.nn.functional as F


def downsample_mask_to_grid(mask_2d, out_h: int, out_w: int) -> torch.Tensor:
    """Average-pool a [Hm, Wm] fluid mask (e.g. 12x12) down to [out_h, out_w].

    Average pooling = the fraction of each coarse cell covered by fluid, which is
    the correct continuous target for an attention distribution over coarse image
    tokens.  Uses adaptive pooling so non-divisible grids (12->6, 12->5, ...) work.
    Returns a float tensor on the same device/dtype as `mask_2d` (if tensor).
    """
    if not torch.is_tensor(mask_2d):
        mask_2d = torch.as_tensor(mask_2d, dtype=torch.float32)
    m = mask_2d.float()[None, None]                       # [1,1,Hm,Wm]
    pooled = F.adaptive_avg_pool2d(m, (out_h, out_w))[0, 0]
    return pooled.to(mask_2d.device if mask_2d.is_cuda else "cpu")


def mask_to_distribution(pooled: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Flatten a pooled mask grid and renormalise to a probability vector (sum=1).
    Returns None-equivalent (all-zero flag) handled by caller: if the mask is empty
    the sum is 0 -> caller should skip the attn term for that eye."""
    flat = pooled.reshape(-1).float()
    s = flat.sum()
    if float(s) <= eps:
        return None
    return flat / s


def torch_attention_rollout(attn: torch.Tensor, add_residual: bool = True) -> torch.Tensor:
    """Differentiable attention rollout.  attn: [L, H, T, T] (or [L, T, T]).
    Returns the [T, T] rolled-out attention (row-normalised), in fp32, grad-tracking.
    Identical recipe to rollout.attention_rollout but in torch."""
    a = attn.float()
    if a.dim() == 4:                      # [L,H,T,T] -> mean heads
        a = a.mean(dim=1)
    if a.dim() != 3:
        raise ValueError(f"expected [L,H,T,T] or [L,T,T], got {tuple(a.shape)}")
    L, T, _ = a.shape
    eye = torch.eye(T, device=a.device, dtype=a.dtype)
    rolled = eye
    for l in range(L):
        layer = a[l]
        if add_residual:
            layer = 0.5 * layer + 0.5 * eye
        layer = layer / layer.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rolled = layer @ rolled
    return rolled


def image_attn_distribution(rolled: torch.Tensor, query_positions: Sequence[int],
                            img_start: int, img_len: int, eps: float = 1e-8) -> torch.Tensor:
    """From a rolled [T,T] matrix, take the mean attention over the given query
    token rows onto the image-token columns [img_start:img_start+img_len], then
    renormalise to a probability vector of length img_len. Differentiable."""
    rows = rolled[list(query_positions), img_start:img_start + img_len]   # [Q, img_len]
    row = rows.mean(dim=0)                                                # [img_len]
    s = row.sum().clamp_min(eps)
    return row / s


def _pool_distribution(flat: torch.Tensor, grid_hw, compare_hw, eps: float) -> torch.Tensor:
    """Reshape a flat distribution over the native image-token grid to grid_hw, average-
    pool to compare_hw, and renormalise. Lets every backbone be compared at a common
    coarse grid (6x6) regardless of native token resolution (36, 576, dynamic...)."""
    gh, gw = grid_hw
    grid = flat.reshape(gh, gw)[None, None]
    pooled = F.adaptive_avg_pool2d(grid, compare_hw)[0, 0].reshape(-1)
    return pooled / pooled.sum().clamp_min(eps)


def attn_kl_loss(attn_layers, img_start: int, img_len: int, query_positions,
                 mask_2d, grid_hw, compare_hw=(6, 6), smooth: float = 1e-3, eps: float = 1e-8):
    """Full arm-C attention term for ONE sample.

    attn_layers     : tuple/list of per-layer [B,H,T,T] (B=1) OR a [L,H,T,T] tensor.
    img_start,img_len: image-token span on the key axis (RetinaVLM 41/36, LLaVA-Med 5/576).
    query_positions : answer-token indices whose grounding we align (target!=-100).
    mask_2d         : the eye's fluid mask grid (e.g. 12x12), tensor or array.
    grid_hw         : (H,W) NATIVE image-token grid (RetinaVLM (6,6), LLaVA-Med (24,24)).
    compare_hw      : common COARSE grid both attention and mask are pooled to before KL.
                      Coarsening stabilises the grad (avoids -q/p blow-up on peaky fine
                      grids) and makes backbones comparable at one resolution.
    smooth          : uniform floor mixed into p so -q/p stays bounded (label-smoothing).
    Returns (kl_scalar_tensor_or_None, p_grid_detached, q_grid_detached) at compare_hw.
    Returns (None, ...) when the mask is empty (dry eye) -> skip attn term.
    """
    if isinstance(attn_layers, (list, tuple)):
        attn = torch.stack([t[0] if t.dim() == 4 else t for t in attn_layers], dim=0)  # [L,H,T,T]
    else:
        attn = attn_layers
        if attn.dim() == 5:                 # [L,B,H,T,T] -> squeeze batch
            attn = attn[:, 0]
    gh, gw = grid_hw
    assert gh * gw == img_len, f"grid {grid_hw} != img_len {img_len}"
    ch, cw = compare_hw

    # target q: pool the fluid mask straight to the compare grid
    q = mask_to_distribution(downsample_mask_to_grid(mask_2d, ch, cw), eps)
    if q is None:
        return None, None, None
    q = q.to(attn.device).float()

    # pred p: rollout -> native-grid attention -> pool to compare grid -> smooth
    rolled = torch_attention_rollout(attn)
    p_native = image_attn_distribution(rolled, query_positions, img_start, img_len, eps)
    p = _pool_distribution(p_native, grid_hw, compare_hw, eps)
    u = torch.full_like(p, 1.0 / p.numel())
    p = (1 - smooth) * p + smooth * u                    # bounded -q/p

    # forward KL( q || p ): mass-covering, pushes attention onto fluid
    kl = (q * (torch.log(q + eps) - torch.log(p + eps))).sum()
    return kl, p.detach().reshape(ch, cw), q.detach().reshape(ch, cw)


if __name__ == "__main__":
    # unit check (CPU, no weights): rollout shape, KL>=0, downsample 12->6 conserves mass
    torch.manual_seed(0)
    L, H, T = 6, 8, 50
    img_start, img_len = 10, 36
    a = torch.rand(L, H, T, T, requires_grad=True)
    a = a / a.sum(-1, keepdim=True)
    mask = torch.zeros(12, 12); mask[3:7, 4:8] = 1.0           # a fluid blob
    pooled = downsample_mask_to_grid(mask, 6, 6)
    assert abs(float(pooled.sum()) - float(mask.sum()) / 4) < 1e-4, "avg-pool mass"
    kl, p, q = attn_kl_loss([t for t in a], img_start, img_len,
                            query_positions=[45, 46, 47], mask_2d=mask, grid_hw=(6, 6))
    assert kl is not None and float(kl) == float(kl), "kl finite"
    print(f"rollout OK  KL={float(kl):.4f}  p.sum={float(p.sum()):.3f}  q.sum={float(q.sum()):.3f}")
    # empty mask -> skip
    kl0, _, _ = attn_kl_loss([t for t in a], img_start, img_len, [45], torch.zeros(12, 12), (6, 6))
    assert kl0 is None, "empty mask should skip"
    print("empty-mask skip OK")
    # grad flows to attention (check the LEAF tensor)
    leaf = torch.rand(L, H, T, T, requires_grad=True)
    a2 = leaf / leaf.sum(-1, keepdim=True)
    kl2, _, _ = attn_kl_loss([t for t in a2], img_start, img_len, [45, 46], mask, (6, 6))
    kl2.backward(); assert leaf.grad is not None and leaf.grad.abs().sum() > 0, "grad flows"
    print("grad-flow OK")
