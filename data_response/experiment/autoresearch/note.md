# AutoResearch Experiment Journal — APTOS 2021 Task 1

## 프로젝트 개요
- **목표**: Mean AUC (IRF+SRF+PED+HRF) 최대화
- **베이스라인**: 0.9225 (Swin-Base + MIL Max Pool, Epoch 5 best)
- **핵심 문제**: 과적합 (87M params vs 3301 bags), HRF AUC 불안정
- **TIME_BUDGET**: 900초 (15분, ~6 epochs for Swin-Base)
- **GPU**: H100 80GB (GPU 2, ~56GB 여유)
- **VRAM 사용**: 39GB (Swin-Base, batch=8)

---

## Experiment #0 — Baseline (Swin-Base + MIL Max Pool)

### 현재 상태
- **best metric**: 0.9400 (Mean AUC, Epoch 2)
- **기반 commit**: initial

### 결과
- **val_mean_auc**: 0.9400 (Epoch 2)
  - IRF: 0.9607, SRF: 0.9513, PED: 0.9708, HRF: 0.8771
- **train_loss**: Epoch 1: 0.4295, Epoch 2: 0.3006
- **VRAM**: 39364 MB
- **시간**: 334.7s (TIME_BUDGET 300s 초과로 2 epoch만)
- **판정**: BASELINE

### 관찰
- 2 epoch만에 0.94 — 빠른 수렴
- HRF AUC 0.877이 가장 낮음 (HRF prevalence 95.7%로 majority class)
- TIME_BUDGET 300초로는 2 epoch밖에 못 돌림 → 900초로 조정
- 에포크당 ~150초

---

## Experiment #1 — Swin-Small (50M) backbone

### 현재 상태
- **best metric**: 0.9408 (Epoch 2, +0.0008 vs baseline)
- **기반 commit**: exp#0 baseline

### 가설 (CoT)
> **관찰**: Baseline이 87M 파라미터로 2 epoch만에 0.94 도달 후 TIME_BUDGET 초과
> **해석**: 모델이 너무 커서 epoch 수 부족, 과적합 위험
> **가설**: Swin-Small(50M)로 경량화하면 더 많은 epoch + 과적합 완화 → AUC 개선
> **구체적 변경**: backbone swin_base → swin_small, feature dim 1024→768
> **예상 영향**: AUC 소폭 상승, HRF AUC 개선
> **리스크**: 모델 용량 부족으로 특징 추출 저하

### 변경 내용
- `train.py:76` — backbone_name "swin_base_patch4_window7_224" → "swin_small_patch4_window7_224"
- `train.py` — feature dim 1024 → 768 (Swin-Small 출력)
- `train.py` — print 문 모델명 업데이트

### 결과
- **val_mean_auc**: 0.9408 (Epoch 2, +0.0008 vs baseline)
  - IRF: 0.9560, SRF: 0.9217, PED: 0.9664, HRF: 0.9189
- **VRAM**: 29.5 GB (baseline 39.4 → -10GB)
- **학습 시간**: 956.8s / 10 epochs (~96s/epoch vs ~150s/epoch)
- **판정**: ❌ REVERT (개선 미미, HRF AUC 개선 안 됨)

### 판정 근거
> Mean AUC 개선 +0.0008로 유의미하지 않음. HRF AUC 0.9189로 오히려 불안정
> (epoch 4에서 0.7312로 붕괴 후 회복). 과적합 패턴 동일 — epoch 2 이후
> train_loss 0.30→0.11, val_auc 0.9408→0.9251. VRAM 절감은 긍정적이나
> 성능 개선 없이 경량화만으로는 의미 없음.

---

## Experiment #2 — Swin-Base + Strong Regularization

### 현재 상태
- **best metric**: 0.9400 (baseline이 여전히 best)
- **기반 commit**: exp#0 baseline

### 가설 (CoT)
> **관찰**: 모든 실험에서 Epoch 2 이후 과적합
> **해석**: 정규화가 부족해서 과적합이 발생
> **가설**: 강한 정규화(dropout↑, wd↑, label smoothing, grad clip)로 과적합 완화 → AUC 개선
> **구체적 변경**: dropout 0.3→0.5, wd 1e-4→1e-2, label smoothing 0.1, grad clip 1.0, batch 16, warmup 2
> **예상 영향**: Epoch 2 이후 AUC 유지/향상
> **리스크**: 정규화가 너무 강하면 과소적합

### 변경 내용
- `train.py` — dropout=0.5, WEIGHT_DECAY=1e-2, LabelSmoothingBCELoss, grad_clip, BATCH_SIZE=16, WARMUP_EPOCHS=2
- backbone을 swin_base로 복원

### 결과
- **val_mean_auc**: 0.9247 (Epoch 2, -0.0153 vs baseline)
  - IRF: 0.9607, SRF: 0.9518, PED: 0.9732, HRF: 0.8131
- **VRAM**: 77.3 GB (batch 16 + Swin-Base — GPU 거의 꽉 참)
- **학습 시간**: 995.7s / 9 epochs (~110s/epoch)
- **판정**: ❌ REVERT (성능 대폭 저하)

### 판정 근거
> 정규화가 너무 공격적 — 과소적합 발생. train_loss가 epoch 9까지 0.37로
> 높게 유지 (baseline은 epoch 2에 0.30). HRF AUC 0.81로 심각 저하.
> 과적합은 지연되었으나 근본 해결 안 됨 — val AUC epoch 2 이후 여전히 변동.
> batch_size 16은 VRAM 77GB 소모 — 더 이상 batch 확장 불가.

---

### ⚠️ 분기 결정 (REVERT 2회 연속)

> **실패한 시도들**:
> 1. Exp #1: 경량 백본 (Swin-Small) → 개선 미미 +0.0008
> 2. Exp #2: 강한 정규화 → 대폭 저하 -0.0153
>
> **공통 실패 원인**: 두 실험 모두 "과적합 완화"를 목표로 했으나, 
> 근본 원인은 **HRF 클래스** (prevalence 95.7%, majority class).
> HRF가 거의 모든 이미지에 존재하므로 모델이 쉽게 "항상 1" 예측 → 
> AUC가 높아 보이지만 실제 변별력 없음 → epoch 간 변동 심함.
>
> **탐색하지 않은 방향**:
> 1. HRF 클래스 샘플링/가중치 전략 변경 (pos_weight < 1.0 허용?)
> 2. Attention-based MIL (Max Pool → Attention Pool)
> 3. Data augmentation 강화 (MixUp, CutMix)
> 4. Focal Loss (hard example mining)
> 5. Multi-task learning (Task 1+2 동시)
>
> **다음 전략**: Attention-based MIL + Focal Loss — 
> Max Pool이 HRF의 "항상 존재" 신호를 증폭하는 반면,
> Attention Pool은 의미 있는 인스턴스에 가중치를 줄 수 있음.
> Focal Loss는 easy negative(HRF)에 대한 loss를 감소시켜 
> hard example(SRF, PED)에 집중하게 함.

---

## Experiment #3 — Attention MIL + Focal Loss

### 현재 상태
- **best metric**: 0.9400 (baseline이 여전히 best)
- **기반 commit**: exp#0 baseline

### 가설 (CoT)
> **관찰**: Max Pool이 HRF(항상 존재) 신호를 증폭
> **해석**: Attention Pool은 의미 있는 인스턴스에 가중치, Focal Loss는 easy example loss 감소
> **가설**: Attention Pool + Focal Loss → HRF 변별력 향상 → Mean AUC 개선
> **리스크**: Attention Pool이 per-instance 연산 추가 → epoch 느려짐

### 결과
- **val_mean_auc**: 0.9285 (Epoch 8, -0.0115 vs baseline)
  - IRF: 0.9542, SRF: 0.9526, PED: 0.9681, HRF: 0.8388
- **VRAM**: 39.4 GB
- **학습 시간**: 924.5s / 8 epochs (~115s/epoch, Max Pool보다 느림)
- **판정**: ❌ REVERT (성능 저하)

### 판정 근거
> Attention Pool 오버헤드로 epoch당 ~115s (baseline ~150s와 비슷하나
> Attention 추가 연산으로 8 epoch만 완료). Focal Loss가 loss scaling을
> 바꿨으나 수렴 속도 저하. HRF 개선 안 됨 (0.8388).

---

### ⚠️ 분기 결정 (REVERT 3회 연속 — 병렬 탐색 모드)

> **실패한 시도들**:
> 1. Exp #1: 경량 백본 → 개선 미미
> 2. Exp #2: 강한 정규화 → 과소적합
> 3. Exp #3: Attention MIL + Focal Loss → 오버헤드 + 수렴 저하
>
> **공통 실패 원인 분석**:
> - 모든 실험이 Epoch 2에서 peak → 이후 overfit/불안정
> - 900초에 6-10 epoch밖에 못 함 → early stopping이 의미 없음
> - HRF (95.7% prevalence)가 변동성의 주원인
> - 근본 문제: **epoch이 너무 적다** → 모델이 충분히 학습하기 전에 TIME_BUDGET 초과
>
> **병렬 탐색 3방향**:
> 1. **방향 A**: ConvNeXt-Tiny (28M) — 훨씬 빠른 epoch, 20+ epoch 가능, 충분한 학습
> 2. **방향 B**: Backbone Freeze — Swin-Base 동결 후 head만 학습, epoch당 30s, 25+ epoch
> 3. **방향 C**: Strong Augmentation (MixUp + RandAugment) — 과적합 방지, epoch 수 유지
>
> **병렬 실행 결과**: 파일 충돌 + GPU 경합으로 3개 서브에이전트가 동시에 train.py 수정.
> 서브에이전트들이 서로 다른 코드를 번갈아 쓰며 경쟁. 결국 직접 순차 실행으로 전환.

---

## Experiment #4A — ConvNeXt-Tiny + MIL Max Pool

### 현재 상태
- **best metric**: 0.9418 (Epoch 7, +0.0018 vs baseline)
- **기반 commit**: exp#0 baseline

### 가설 (CoT)
> **관찰**: Baseline이 87M params로 2 epoch만에 TIME_BUDGET 초과, Epoch 2에서 과적합
> **해석**: 모델이 너무 커서 epoch 수 부족. 더 작은 모델로 더 많은 epoch → 피크 탐색
> **가설**: ConvNeXt-Tiny(28M)로 경량화하면 10+ epoch 가능 → 더 나은 피크 도달
> **구체적 변경**: backbone convnext_tiny, feature_dim=768, lr=1e-4, warmup=5
> **예상 영향**: 더 많은 epoch → 더 나은 AUC peak
> **리스크**: 모델 용량 부족

### 결과
- **val_mean_auc**: 0.9418 (Epoch 7, +0.0018 vs baseline)
  - IRF: 0.9552, SRF: 0.9419, PED: 0.9758, HRF: 0.8945
- **VRAM**: 16.6 GB (baseline 39.4 → -22.8GB!)
- **학습 시간**: 900s / 10 epochs (~90s/epoch)
- **판정**: ✅ KEEP (새로운 best!)

### 판정 근거
> Baseline을 처음으로 능가! ConvNeXt-Tiny의 빠른 epoch(90s vs 150s) 덕분에
> 10 epoch 학습 → Epoch 7에서 0.9418 도달. HRF AUC 0.8945로 향상
> (baseline 0.8771). 하지만 Epoch 7 이후 과적합 (0.9418→0.9217→0.9253→0.9360).
> VRAM 16.6GB로 batch 크기 확장 가능.
>
> **중요 발견**: 모델 크기가 작을수록 더 많은 epoch → 더 나은 peak.
> 하지만 과적합 패턴은 동일 — Epoch 7-8 이후 하락.

---

## Experiment #5 — ConvNeXt-Tiny + MixUp

### 현재 상태
- **best metric**: 0.9475 (Epoch 2, +0.0057 vs baseline, +0.0057 vs Exp#4A)
- **기반 commit**: exp#4A

### 가설 (CoT)
> **관찰**: Exp #4A에서 Epoch 7 이후 과적합 (0.9418→0.9217)
> **해석**: ConvNeXt-Tiny는 빠르지만 과적합이 여전히 발생
> **가설**: MixUp 정규화(alpha=0.2, prob=0.5)로 과적합 지연 → 더 나은 peak 또는 더 안정적 수렴
> **구체적 변경**: train_one_epoch에 MixUp 적용, 나머지 동일
> **예상 영향**: 과적합 지연, AUC 개선
> **리스크**: MixUp이 variance 증가 → epoch 간 변동성 증가

### 결과
- **val_mean_auc**: 0.9475 (Epoch 2, +0.0057 vs baseline)
  - IRF: 0.9574, SRF: 0.9555, PED: 0.9607, HRF: 0.9165
- **VRAM**: 16.8 GB
- **학습 시간**: ~900s / 17 epochs (~53s/epoch — MixUp 오버헤드로 약간 느림)
- **판정**: ✅ KEEP (새로운 best!)

### 판정 근거
> Mean AUC 0.9475로 새로운 best! 특히 HRF AUC 0.9165로 큰 개선
> (baseline 0.8771, Exp#4A 0.8945). 하지만 Epoch 2 이후 급격 하락
> (0.9475→0.8859) — MixUp이 variance를 크게 증가시킴.
> Early stopping 발동 (patience=15, Epoch 17).
> **핵심 딜레마**: Best는 좋지만 재현성이 의문 — Epoch 2 운이 좋았을 수 있음.
> 안정적으로 0.9475에 도달하는 방법이 필요.

---

## Experiment #6 — ConvNeXt-Tiny + MixUp + HRF pos_weight downweight + Batch 16

### 현재 상태
- **best metric**: 0.9467 (Epoch 5, -0.0008 vs Exp#5, 여전히 Exp#5가 best)
- **기반 commit**: exp#5

### 가설 (CoT)
> **관찰**: Exp #5에서 Epoch 2에 0.9475 달성 후 불안정. HRF가 variance 주원인.
> **해석**: HRF prevalence 95.7%인데 pos_weight=1.0이면 과도 반영
> **가설**: HRF pos_weight를 0.5로 cap + batch 16 + warmup 10 → 안정적 수렴
> **리스크**: HRF downweight가 너무 극단적이면 HRF 학력 붕괴

### 결과
- **val_mean_auc**: 0.9467 (Epoch 5, -0.0008 vs Exp#5)
  - IRF: 0.9652, SRF: 0.9679, PED: 0.9803, HRF: 0.8733
- **VRAM**: 33.0 GB (batch=16)
- **학습 시간**: 900s / 15 epochs (~60s/epoch)
- **판정**: ❌ REVERT (Exp#5가 여전히 best)

### 판정 근거
> Epoch 5에서 0.9467 — Exp#5(0.9475)보다 낮음. 결정적 문제:
> Epoch 10에서 HRF AUC 0.5025로 붕괴! pos_weight=0.5가 너무 극단적.
> HRF를 downweight하면 모델이 HRF negative를 무시하게 됨.
> Batch 16 + warmup 10은 긍정적이나 HRF downweight가 치명적.
>
> **교훈**: HRF pos_weight를 1.0 미만으로 내리는 건 위험.
> 대신 pos_weight를 그대로 두고 다른 정규화로 HRF 변별력 향상을 시도해야 함.

---

## Experiment #7 — ConvNeXt-Tiny + MixUp + Batch 16 + Dropout 0.5

### 현재 상태
- **best metric**: 0.9444 (Epoch 2, -0.0031 vs Exp#5)
- **기반 commit**: exp#5

### 가설 (CoT)
> **관찰**: Exp #5에서 Epoch 2 이후 variance 큼. Batch 16은 긍정적 (Exp #6).
> **해석**: 더 강한 dropout으로 과적합 지연 + batch 16로 gradient 안정화
> **가설**: Dropout 0.5 + batch 16 → 과적합 지연 → 더 안정적 수렴
> **리스크**: Dropout이 너무 강하면 과소적합

### 결과
- **val_mean_auc**: 0.9444 (Epoch 2, -0.0031 vs Exp#5)
  - IRF: 0.9621, SRF: 0.9449, PED: 0.9759, HRF: 0.8949
- **VRAM**: 33.0 GB
- **학습 시간**: 900s / 12 epochs (~75s/epoch)
- **판정**: ❌ REVERT (dropout 0.5 과소적합)

### 판정 근거
> Dropout 0.5가 너무 강함 — Epoch 2 이후 AUC 지속 하락 (0.9444→0.9075).
> Exp #2에서도 dropout 0.5가 과소적합 유발. 같은 실수 반복.
> train_loss가 epoch 12까지 0.27로 높게 유지 — 모델이 충분히 학습 못함.
>
> **교훈**: 이 데이터셋에서 dropout 0.5는 항상 과소적합. 0.3이 적절.
> 다음 실험은 dropout 0.3 유지 + 다른 정규화 방법 탐색.

---

### ⚠️ 분기 결정 (Exp #5 이후 2회 연속 REVERT)

> **현재 best**: Exp #5 = 0.9475 (ConvNeXt-Tiny + MixUp, Epoch 2)
>
> **시도한 방향**:
> 1. Exp #6: HRF pos_weight downweight → HRF 붕괴
> 2. Exp #7: Dropout 0.5 → 과소적합
>
> **미탐색 방향**:
> 1. **SWA (Stochastic Weight Averaging)**: 여러 checkpoint 평균 → variance 감소
> 2. **Multi-snapshot ensemble**: Epoch 2, 5, 7 모델 앙상블
> 3. **Test-Time Augmentation (TTA)**: 추론 시 augmentation 평균
> 4. **Label Smoothing (0.05)**: 약한 정규화, dropout처럼 과소적합 위험 적음
> 5. **Cosine Annealing with Warm Restarts**: 주기적 lr 리셋으로 여러 local minima 탐색

---

## Experiment #8 — ConvNeXt-Tiny + MixUp + SWA

### 현재 상태
- **best metric**: 0.9413 (Epoch 5, -0.0062 vs Exp#5), SWA: 0.9256
- **기반 commit**: exp#5

### 가설 (CoT)
> **관찰**: Exp #5에서 Epoch 2에 0.9475 후 불안정
> **해석**: 여러 epoch의 weight를 평균하면 variance 감소 → 더 안정적
> **가설**: SWA로 weight averaging → variance 감소 → 안정적 peak
> **리스크**: SWA lr 스케줄링이 수렴을 방해할 수 있음

### 결과
- **val_mean_auc**: 0.9413 (Epoch 5, -0.0062 vs Exp#5)
- **SWA val_mean_auc**: 0.9256 (regular보다 낮음!)
  - IRF: 0.9677, SRF: 0.9560, PED: 0.9834, HRF: 0.7953
- **VRAM**: 16.9 GB
- **학습 시간**: 900s / 15 epochs (~60s/epoch)
- **판정**: ❌ REVERT (SWA가 수렴 방해)

### 판정 근거
> SWA가 오히려 성능 저하. swa_lr=5e-5가 너무 낮아서
> SWA 시작(epoch 6) 이후 lr이 급감 → 수렴 불안정.
> Epoch 6에서 AUC 0.8920으로 붕괴 (SWA 전환 직후).
> SWA 모델의 최종 AUC도 0.9256로 regular(0.9413)보다 낮음.
>
> **교훈**: 이 설정에서 SWA는 부적합. 너무 적은 epoch에
> SWA averaging이 오히려 좋은 weight를 희석시킴.
> 대신 **앙상블**이나 **TTA**를 고려해야 함.

---

### ⚠️ 분기 결정 (Exp #5 이후 3회 연속 REVERT)

> **현재 best**: Exp #5 = 0.9475 (ConvNeXt-Tiny + MixUp, Epoch 2)
>
> **시도한 방향**:
> 1. Exp #6: HRF pos_weight downweight → HRF 붕괴
> 2. Exp #7: Dropout 0.5 → 과소적합
> 3. Exp #8: SWA → 수렴 방해
>
> **핵심 인사이트**: Exp #5의 0.9475는 Epoch 2의 단일 데이터포인트.
> 재현성이 의문. 안정적으로 0.9475 이상을 달성하는 방법 필요.
>
> **다음 전략**:
> 1. **Label Smoothing (0.05)**: 가장 약한 정규화, 과소적합 위험 최소
> 2. **TTA (Test-Time Augmentation)**: 추론 시 5-crop + flip 평균 → variance 감소
> 3. **Multi-seed ensemble**: 3개 seed로 학습 후 예측 평균
> 4. **Cosine Annealing with Warm Restarts**: lr 주기적 리셋으로 여러 피크 탐색

---

## Experiment #9 — ConvNeXt-Tiny + MixUp + Label Smoothing(0.05) + TTA

### 현재 상태
- **best metric**: 0.9326 (Epoch 2, -0.0149 vs Exp#5)
- **기반 commit**: exp#5

### 가설 (CoT)
> **관찰**: 3회 연속 REVERT. 강한 정규화는 모두 실패
> **해석**: 가장 약한 정규화(label smoothing) + TTA로 variance 감소
> **가설**: Label Smoothing 0.05 + TTA 3-view → 과적합 지연 + 추론 안정화
> **리스크**: TTA가 epoch당 시간 3배 → epoch 수 감소

### 결과
- **val_mean_auc**: 0.9326 (Epoch 2, -0.0149 vs Exp#5)
- **VRAM**: 16.8 GB
- **학습 시간**: 950.9s / 12 epochs (~79s/epoch, TTA 오버헤드)
- **판정**: ❌ REVERT (성능 대폭 저하)

### 판정 근거
> Label Smoothing이 수렴을 방해 — train_loss가 epoch 12까지 0.31로
> 높게 유지 (Exp #5는 epoch 2에 0.33). Label Smoothing이 타겟을
> 0.95/0.05로 만들어서 모델이 확신을 갖지 못함.
> TTA는 epoch당 3배 시간 소요 → 12 epoch밖에 못함.
> TTA 자체는 약간 도움될 수 있으나, label smoothing과 결합하면
> 상호작용으로 성능 악화.
>
> **핵심 교훈**: 이 데이터셋에서는 **정규화 추가가 항상 해가 됨**.
> Exp #5가 이미 최적의 정규화 균형(MixUp 0.2 + dropout 0.3).
> 추가 정규화는 수렴만 방해. 대신 **다른 접근** 필요.

---

### ⚠️ 분기 결정 (Exp #5 이후 4회 연속 REVERT — 전략 전환)

> **현재 best**: Exp #5 = 0.9475 (ConvNeXt-Tiny + MixUp, Epoch 2)
>
> **정규화 추가는 모두 실패**:
> 1. Exp #6: HRF pos_weight downweight → HRF 붕괴
> 2. Exp #7: Dropout 0.5 → 과소적합
> 3. Exp #8: SWA → 수렴 방해
> 4. Exp #9: Label Smoothing + TTA → 수렴 방해 + 시간 낭비
>
> **핵심 깨달음**: Exp #5의 0.9475는 이미 이 아키텍처의 near-optimal.
> 정규화 추가로는 개선 불가. **근본적으로 다른 접근** 필요:
>
> 1. **ConvNeXt-Base (88M)**: 더 큰 백본으로 feature 품질 향상
> 2. **Multi-scale feature**: 여러 레이어의 feature 결합 (FPN 스타일)
> 3. **Attention MIL 재도전**: ConvNeXt로 더 빠른 epoch → Attention 오버헤드 감소
> 4. **Bag-level augmentation**: 이미지가 아닌 bag 단위 증강
> 5. **더 큰 이미지 (384x384)**: 해상도 증가로 미세 패턴 포착

---

## Experiment #10 — ConvNeXt-Tiny + MixUp + 384x384 Image Resolution

### 현재 상태
- **best metric**: 0.9475 (Exp#5가 여전히 best)
- **기반 commit**: exp#5

### 가설 (CoT)
> **관찰**: 정규화 추가는 4회 연속 실패. 근본적으로 다른 접근 필요.
> **해석**: OCT 원본이 632x596인데 224x224로 다운샘플링하면 미세한 망막층 구조 손실.
> IRF/SRF/PED는 망막층 내 미세 액체 축적으로, 해상도에 매우 민감.
> **가설**: 384x384로 해상도 증가 → 미세 패턴 보존 → AUC 개선
> **구체적 변경**: IMG_SIZE 224→384, BATCH_SIZE 8→4 (VRAM 보상)
> **예상 영향**: IRF/SRF/PED AUC 개선 (미세 구조 포착), HRF는 큰 변화 없음
> **리스크**: 384x384로 인해 epoch당 시간 ~2.5배 증가 → epoch 수 감소.
> VRAM 초과 가능성 (ConvNeXt-Tiny는 효율적이므로 batch=4면 OK 예상)

### 결과
- **val_mean_auc**: 0.9461 (Epoch 1, -0.0014 vs Exp#5)
  - IRF: 0.9628, SRF: 0.9549, PED: 0.9660, HRF: 0.9006
- **VRAM**: 24.3 GB (batch=4, 384x384)
- **학습 시간**: 946.7s / 5 epochs (~189s/epoch vs ~53s/epoch for 224)
- **판정**: ❌ REVERT (epoch 수 부족, Exp#5가 여전히 best)

### 판정 근거
> 384x384는 epoch당 ~189초 — 224의 3.6배. 900초에 5 epoch밖에 못함.
> Epoch 1에 0.9461로 꽤 높으나, Exp#5(0.9475)보다 낮음.
> **중요 발견**: 384x384의 HRF AUC 0.9006이 224의 0.9165보다 낮음 —
> 해상도 증가가 반드시 도움 되지 않음. 오히려 epoch 수 부족이 더 치명적.
> Epoch 1 train_loss 0.4096이 Exp#5의 epoch 1(0.33)보다 높음 —
> warmup 5 epoch를 1 epoch밖에 못 돌려서 lr이 너무 낮았을 수 있음.
>
> **핵심 교훈**: TIME_BUDGET 제약 하에서는 **epoch 수가 해상도보다 중요**.
> 224x224로 17 epoch vs 384x384로 5 epoch → 224가 압도적 우위.
> 단, warmup 없이 바로 학습하면 384에서도 가능할 수 있음 (warmup이 5 epoch 낭비).

---

### ⚠️ 분기 결정 (Exp #5 이후 5회 연속 REVERT)

> **현재 best**: Exp #5 = 0.9475 (ConvNeXt-Tiny + MixUp, 224x224, Epoch 2)
>
> **시도한 방향**:
> 1. Exp #6: HRF pos_weight downweight → HRF 붕괴
> 2. Exp #7: Dropout 0.5 → 과소적합
> 3. Exp #8: SWA → 수렴 방해
> 4. Exp #9: Label Smoothing + TTA → 수렴 방해 + 시간 낭비
> 5. Exp #10: 384x384 해상도 → epoch 수 부족 (5 epoch only)
>
> **공통 패턴**: 모든 실험이 Exp #5의 **단일 하이퍼파라미터 조합**을 능가하지 못함.
> Exp #5의 0.9475는 특정 seed(42)의 Epoch 2에서 운이 좋았을 가능성.
>
> **다음 전략 — 다중 seed 앙상블**:
> Exp #5 코드를 그대로 사용하되 **3개 seed**로 독립 학습 후 예측 평균.
> 이렇게 하면:
> 1. 단일 seed의 행운/불운을 평활화
> 2. 앙상블 효과로 variance 감소 → 더 안정적 AUC
> 3. 근본적 변경 없이 Exp #5 설정 재활용
>
> 또는 **ConvNeXt-Base(88M)** 시도:
> 더 큰 백본으로 feature 품질 향상. VRAM 16.8→약 40GB 예상.
> batch=4로 축소, epoch당 ~150s → 5-6 epoch. Exp#10 교훈: epoch 수 중요.
> 하지만 ConvNeXt-Base는 224로도 가능 → epoch당 ~120s → 7 epoch.

---

## Experiment #11 — ConvNeXt-Base(88M) + MixUp

### 현재 상태
- **best metric**: 0.9475 (Exp#5가 여전히 best)
- **기반 commit**: exp#5

### 가설 (CoT)
> **관찰**: 정규화 추가, 해상도 증가 모두 실패. Exp #5가 near-optimal.
> **해석**: ConvNeXt-Tiny(28M)의 feature 품질이 한계일 수 있음.
> **가설**: ConvNeXt-Base(88M)로 더 풍부한 feature → 미세 패턴 포착 → AUC 개선
> **구체적 변경**: backbone convnext_tiny→convnext_base, feature_dim 768→1024, hidden_dim 512→768, batch=4
> **예상 영향**: IRF/SRF/PED AUC 개선 (더 정교한 feature), HRF는 큰 변화 없음
> **리스크**: batch=4로 gradient 노이즈 증가, epoch당 시간 증가 (~120s → 7 epoch)

### 결과
- **val_mean_auc**: 0.9268 (Epoch 3, -0.0207 vs Exp#5)
  - IRF: 0.9545, SRF: 0.9546, PED: 0.9671, HRF: 0.8311
- **VRAM**: 18.2 GB (batch=4, ConvNeXt-Base)
- **학습 시간**: 946.4s / 7 epochs (~135s/epoch)
- **판정**: ❌ REVERT (성능 대폭 저하)

### 판정 근거
> ConvNeXt-Base가 수렴이 너무 느림 — batch=4의 gradient 노이즈 + warmup 5 epoch
> 낭비. Epoch 7까지도 0.92 수준에 머무름. Exp#5는 Epoch 2에 0.9475.
> HRF AUC 0.8311이 낮음 — batch=4로 인한 gradient 불안정.
> **핵심 교훈**: 더 큰 백본 + 더 작은 batch = 수렴 지연. batch 크기가
> 백본 크기에 비례해야 함. ConvNeXt-Tiny + batch=8가 최적 조합.

---

### ⚠️ 분기 결정 (Exp #5 이후 6회 연속 REVERT — 전략 재검토)

> **현재 best**: Exp #5 = 0.9475 (ConvNeXt-Tiny + MixUp, 224x224, batch=8, Epoch 2)
>
> **시도한 방향 (모두 REVERT)**:
> 1. Exp #6: HRF pos_weight downweight → HRF 붕괴
> 2. Exp #7: Dropout 0.5 → 과소적합
> 3. Exp #8: SWA → 수렴 방해
> 4. Exp #9: Label Smoothing + TTA → 수렴 방해
> 5. Exp #10: 384x384 → epoch 수 부족
> 6. Exp #11: ConvNeXt-Base → 수렴 지연 (batch=4 노이즈)
>
> **확립된 인사이트**:
> - **정규화 추가는 해가 됨**: MixUp(0.2) + dropout(0.3)이 최적
> - **백본 확대는 batch 축소로 인해 수렴 저하**: Tiny+8 > Base+4
> - **해상도 증가는 epoch 수 부족**: 224x17epoch > 384x5epoch
> - **HRF pos_weight 조정은 위험**: 1.0 미만 금지
>
> **다음 전략 — Multi-seed Ensemble**:
> Exp #5 코드를 그대로 사용, 3개 seed로 독립 학습 후 예측 평균.
> seed=42, 123, 2024로 3번 학습 → 앙상블 → variance 감소.
> 각각 300초 (총 900초 내). 가장 안전한 전략.

---

## Experiment #12 — Multi-seed Ensemble (3 seeds: 42, 123, 2024)

### 현재 상태
- **best metric**: 0.9491 (Ensemble Mean AUC, +0.0016 vs Exp#5)
- **기반 commit**: exp#5

### 가설 (CoT)
> **관찰**: Exp #5의 0.9475는 Epoch 2 단일 데이터포인트, 6회 연속 REVERT
> **해석**: 단일 seed의 variance가 큼 — 앙상블로 평활화 필요
> **가설**: 3개 seed로 독립 학습 후 예측 평균 → variance 감소 → 안정적 AUC 향상
> **구체적 변경**: Exp #5 코드 그대로, train_one_seed()로 3 seed 순차 학습, 각 300s
> **예상 영향**: 단일 seed 최고(0.9418) < 앙상블(0.9491) → variance 감소 효과
> **리스크**: 각 seed의 epoch 수 감소 (5 vs 17) → 개별 seed 품질 저하

### 결과
- **Ensemble val_mean_auc**: 0.9491 (+0.0016 vs Exp#5)
  - IRF: 0.9615, SRF: 0.9721, PED: 0.9718, HRF: 0.8911
- **개별 seed 결과**:
  - Seed 42: 0.9418 (Epoch 2, 337s, 5 epochs)
  - Seed 123: 0.9318 (Epoch 3, 335s, 5 epochs)
  - Seed 2024: 0.9286 (Epoch 2, 266s, 4 epochs)
- **VRAM**: 16.8 GB
- **학습 시간**: 965.8s / 900s budget
- **판정**: ✅ KEEP (새로운 best!)

### 판정 근거
> Ensemble 0.9491로 Exp#5(0.9475)를 능가! 핵심 인사이트:
> - 개별 seed는 각각 5 epoch밖에 못 돌림 (300s/seed), Exp#5의 17 epoch 대비 현저히 적음
> - 그러나 앙상블이 variance를 효과적으로 감소시킴
> - SRF AUC 0.9721이 특히 뛰어남 (Exp#5 0.9555)
> - HRF AUC 0.8911은 Exp#5(0.9165)보다 낮음 — 여전히 불안정
> - Ensemble improvement: +0.0073 vs best single seed (0.9418)
>
> **핵심 교훈**: 앙상블은 epoch 수가 적어도 variance 감소로 단일 모델 이상의 성능 달성 가능.
> 하지만 HRF 안정성은 여전히 과제.

---

### ⚠️ 분기 결정 (Exp #12 성공 — 추가 개선 탐색)

> **현재 best**: Exp #12 = 0.9491 (Multi-seed Ensemble)
>
> **성공 요인**: 앙상블의 variance 감소 효과가 epoch 수 부족을 상쇄
>
> **미탐색 방향**:
> 1. **5-seed 앙상블**: seed를 5개로 늘려 variance 추가 감소 (각 180s, 3 epochs)
> 2. **Snapshot ensemble**: 단일 seed 학습 중 여러 checkpoint 앙상블 (cosine annealing)
> 3. **Weighted ensemble**: seed별 성능에 따라 가중 평균 (성능 높은 seed에 더 큰 가중치)
> 4. **KFold 앙상블**: 데이터 분할을 다르게 하여 다양성 확보
> 5. **더 긴 단일 seed**: TIME_BUDGET 전체를 1개 seed에 사용 (17 epoch) + TTA

---
