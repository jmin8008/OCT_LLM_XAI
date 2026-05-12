# Program — APTOS 2021 Task 1 Classification (IRF/SRF/PED/HRF)

## 수정 범위
- train.py — **전권 위임**. 아키텍처, 백본, MIL 방식, 옵티마이저, 하이퍼파라미터, 학습 루프,
  배치 크기, 모델 크기, 어텐션 메커니즘, 활성화 함수, 정규화, 위치 인코딩... 무엇이든 바꿀 수 있다.
  완전히 새로 작성해도 된다. 기존 코드에 얽매이지 마라.

## 금지 (hard constraint)
- prepare.py 수정 금지
- 평가 함수 조작 금지 (compute_auc, compute_mape)
- 학습 데이터를 평가에 사용 금지
- 이전 체크포인트 로드 금지 (매번 from scratch)
- TIME_BUDGET 변경 금지
- 새 conda 패키지 설치 금지 (aptos2021 환경 기존 패키지만 사용)

## 목표
**Mean AUC (IRF+SRF+PED+HRF)를 최대화하라.**
현재 베스트: 0.9491 (3-seed ensemble, ConvNeXt-Tiny + MixUp)
단일 모델 베스트: 0.9475 (Exp #5)

## 도메인 특화 제약
- 이미지: 1264x596 JPG, 우측 절반이 OCT (좌측은 fundus)
- MIL bag: 환자+주입 단위(case-level) + 이미지 단위 병합
- 클래스 불균형: HRF 95.7%, PED 18.3%
- VRAM: GPU 2 약 56GB 여유
- 실행: CUDA_VISIBLE_DEVICES=2 conda run -n aptos2021 python train.py

## 이전 실험에서 확립된 인사이트 (11+1 실험)
1. ConvNeXt-Tiny(28M) > Swin-Base(87M) — 더 많은 epoch 가능
2. MixUp(α=0.2)이 유일하게 유효한 정규화 (+0.0057 AUC, HRF +0.022)
3. Dropout 0.5는 과소적합, 0.3이 적절
4. HRF pos_weight < 1.0은 학습 붕괴 유발
5. SWA, Label Smoothing, TTA 모두 수렴 방해
6. 384x384는 epoch 수 부족으로 실패
7. Multi-seed ensemble이 단일 모델 대비 +0.0016 개선
8. 핵심 병목: HRF AUC 불안정 (0.87~0.92 범위), epoch 2-7 피크 후 과적합

## 미탐색 방향
1. 5-seed / 7-seed 앙상블 (variance 추가 감소)
2. Snapshot ensemble (cosine annealing으로 다중 checkpoint)
3. Weighted ensemble (성능 기반 seed 가중치)
4. KFold 앙상블 (데이터 분할 다양성)
5. 더 긴 단일 seed (TIME_BUDGET 전체) + TTA
6. ConvNeXt-Tiny + Attention MIL (오버헤드 감소 상태에서 재도전)
7. Multi-scale feature (FPN 스타일)
8. Bag-level augmentation
9. 학습 전략: cosine annealing with warm restarts
10. FPN-style multi-layer feature fusion

## 단순성 원칙
- 동일 성능이면 단순한 쪽이 이긴다.
- 0.001 개선인데 20줄 복잡도 추가? 가치 없다.
- 0.001 개선인데 코드가 더 깔끔해졌다면? 확실히 KEEP.

## 자율성
- NEVER STOP. 사람이 수동으로 중단할 때까지 무한 루프.
- 사람에게 "계속할까요?" 묻지 마라.
- 아이디어가 고갈되면 더 열심히 생각하라.
