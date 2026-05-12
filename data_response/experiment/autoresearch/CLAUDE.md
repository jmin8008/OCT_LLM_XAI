# APTOS 2021 Task 1 — AutoResearch Project

## 목표
Mean AUC (IRF+SRF+PED+HRF) 최대화. 베이스라인 0.9400.

## 데이터
- 221 환자, 2875 OCT 이미지
- 4-class 분류: IRF (84.1%), SRF (34.2%), PED (18.3%), HRF (95.7%)
- MIL bag: case-level (patient+injection) + image-level 병합
- 이미지: 1264x596 JPG, 우측 절반 = OCT

## 환경
- conda: aptos2021 (torch 2.5, timm 1.0.26)
- GPU: CUDA_VISIBLE_DEVICES=2 (H100 80GB)
- 실행: `conda run -n aptos2021 python train.py`

## 발견한 인사이트
- ConvNeXt-Tiny(28M)가 Swin-Base(87M)보다 우수 — 더 많은 epoch 가능
- HRF(95.7% prevalence)가 variance의 주원인, pos_weight downweight는 위험
- MixUp이 best는 개선하나 variance 증가
- 모든 실험이 Epoch 2-7에서 peak 후 과적합

## 실험 관리 규칙
1. **코드 보존**: 매 실험 완료 후 `archive/train_expN.py`로 train.py 복사
2. **로그 보존**: 매 실험 완료 후 `archive/log_expN.txt`로 로그 복사
3. **results.json**: 모든 실험 결과 누적 기록
4. **note.md**: 가설-결과-판정 기록
5. **Subagent 사용**: 실험 실행은 subagent로, 결과만 메인에서 분석
6. **순차 실행**: 동일 파일 동시 수정 방지, 한 번에 1개 실험만
