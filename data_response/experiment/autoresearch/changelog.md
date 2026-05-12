# APTOS 2021 Task 1 — AutoResearch Changelog

## 2026-04-26 Session

### Phase 1: Baseline & Early Experiments (01:45–03:35)

- **Exp #0 (baseline)**: Swin-Base(87M) + MIL Max Pool → **0.9400** (Epoch 2)
  - VRAM 39.4GB, 2 epochs only (TIME_BUDGET 300s hit)
  - Adjusted TIME_BUDGET 300→900s for more epochs

- **Exp #1**: Swin-Small(50M) → **0.9408** (Epoch 2, +0.0008)
  - REVERT: marginal improvement, HRF still unstable
  - VRAM saved: 39.4→29.5GB

- **Exp #2**: Swin-Base + Strong Reg (dropout=0.5, wd=1e-2, label_smooth=0.1, batch=16) → **0.9247**
  - REVERT: over-regularized, underfitting
  - VRAM 77.3GB (batch=16 too large for H100 with Swin-Base)

- **Exp #3**: Swin-Base + Attention MIL + Focal Loss → **0.9285**
  - REVERT: attention overhead + Focal Loss convergence slowdown

### Phase 2: ConvNeXt-Tiny Discovery (03:20–03:35)

- **Exp #4A**: ConvNeXt-Tiny(28M) + MIL Max Pool → **0.9418** (Epoch 7)
  - **FIRST KEEP**: beat baseline! Faster epochs (90s vs 150s), 10 epochs in 900s
  - VRAM 16.6GB — huge savings, room for batch expansion
  - Key insight: smaller model + more epochs > larger model + fewer epochs

- **Exp #5**: ConvNeXt-Tiny + MixUp(alpha=0.2) → **0.9475** (Epoch 2)
  - **NEW BEST**: +0.0057 vs Exp#4A
  - HRF AUC improved: 0.8945→0.9165
  - But: Epoch 2 peak then collapse (0.9475→0.8859) — variance issue

### Phase 3: Regularization Exploration (08:20–09:40) — ALL REVERT

- **Exp #6**: + HRF pos_weight=0.5 + batch=16 + MixUp(alpha=0.3) → **0.9467**
  - REVERT: HRF AUC collapsed to 0.5025 at epoch 10
  - Lesson: HRF pos_weight < 1.0 is dangerous

- **Exp #7**: + dropout=0.5 + batch=16 → **0.9444**
  - REVERT: over-regularized, same mistake as Exp#2
  - Lesson: dropout 0.5 always causes underfitting on this dataset

- **Exp #8**: + SWA(swa_lr=5e-5) → **0.9413** (regular), SWA=0.9256
  - REVERT: SWA averaging diluted good weights, hurt convergence
  - Lesson: too few epochs for SWA to work

- **Exp #9**: + Label Smoothing(0.05) + TTA(3-view) → **0.9326**
  - REVERT: label smoothing hurt convergence, TTA tripled val time
  - Lesson: any regularization beyond MixUp+dropout(0.3) hurts

### Phase 4: Strategic Pivot (09:40–10:10) — ALL REVERT

- **Exp #10**: ConvNeXt-Tiny + 384x384 → **0.9461** (Epoch 1)
  - REVERT: only 5 epochs in 900s (189s/epoch vs 53s/epoch at 224)
  - HRF 0.9006 < Exp#5's 0.9165 — resolution didn't help
  - Lesson: epoch count > resolution under TIME_BUDGET

- **Exp #11**: ConvNeXt-Base(88M) + batch=4 → **0.9268** (Epoch 3)
  - REVERT: batch=4 too noisy, convergence too slow
  - Lesson: larger backbone + smaller batch = worse convergence

### Current Best: Exp #5 = 0.9475

| Exp | Method | AUC | Epoch | Status |
|-----|--------|-----|-------|--------|
| #5 | ConvNeXt-Tiny + MixUp(0.2) | **0.9475** | 2 | BEST |
| #6 | + HRF pw=0.5 + batch16 | 0.9467 | 5 | |
| #10 | + 384x384 | 0.9461 | 1 | |
| #4A | ConvNeXt-Tiny (no MixUp) | 0.9418 | 7 | |
| #8 | + SWA | 0.9413 | 5 | |
| #0 | Swin-Base baseline | 0.9400 | 2 | |
| #1 | Swin-Small | 0.9408 | 2 | |
| #7 | + dropout=0.5 | 0.9444 | 2 | |
| #3 | + Attention MIL + Focal | 0.9285 | 8 | |
| #11 | ConvNeXt-Base | 0.9268 | 3 | |
| #2 | + Strong Reg | 0.9247 | 2 | |
| #9 | + Label Smooth + TTA | 0.9326 | 2 | |

### Phase 5: Ensemble Breakthrough (10:30)

- **Exp #12**: Multi-seed Ensemble (seeds 42/123/2024, each 300s) → **0.9491**
  - **NEW BEST**: +0.0016 vs Exp#5
  - Seed 42: 0.9418 (Epoch 2), Seed 123: 0.9318 (Epoch 3), Seed 2024: 0.9286 (Epoch 2)
  - Ensemble: IRF=0.9615, SRF=0.9721, PED=0.9718, HRF=0.8911
  - Each seed only 4-5 epochs (vs Exp#5's 17), but ensemble variance reduction compensates
  - SRF AUC jumped to 0.9721 (Exp#5: 0.9555)
  - HRF still unstable: 0.8911 (Exp#5: 0.9165)

### Current Best: Exp #12 = 0.9491

| Exp | Method | AUC | Epoch | Status |
|-----|--------|-----|-------|--------|
| #12 | Multi-seed Ensemble (3) | **0.9491** | — | BEST |
| #5 | ConvNeXt-Tiny + MixUp(0.2) | 0.9475 | 2 | |
| #6 | + HRF pw=0.5 + batch16 | 0.9467 | 5 | |
| #10 | + 384x384 | 0.9461 | 1 | |
| #4A | ConvNeXt-Tiny (no MixUp) | 0.9418 | 7 | |
| #8 | + SWA | 0.9413 | 5 | |
| #0 | Swin-Base baseline | 0.9400 | 2 | |
| #1 | Swin-Small | 0.9408 | 2 | |
| #7 | + dropout=0.5 | 0.9444 | 2 | |
| #9 | + Label Smooth + TTA | 0.9326 | 2 | |
| #3 | + Attention MIL + Focal | 0.9285 | 8 | |
| #11 | ConvNeXt-Base | 0.9268 | 3 | |
| #2 | + Strong Reg | 0.9247 | 2 | |

### Next Strategy: Expand Ensemble or Snapshot Ensemble
- 5-seed ensemble for further variance reduction
- Snapshot ensemble (cosine annealing checkpoints) for more diversity
- Weighted ensemble by per-seed AUC
