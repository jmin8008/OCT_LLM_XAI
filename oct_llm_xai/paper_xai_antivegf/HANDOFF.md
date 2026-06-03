# HANDOFF — XAI-Instill Vision Capabilities (Anti-VEGF CI)

> 작성 2026-06-03. 다음 세션 인계용. 가설 제안: 권율(조권율, Bionexus R&D).
> 기반 논문: AGE-VLM ("instill vision capabilities … look at correct regions").

## 한 줄 요약
RetinaVLM(tier3)은 유체를 **본다**(FER≈0.21). 그러나 **보는 것이 임상 결정과 인과적으로 분리(decoupling)**
되어 있고, inference-time attention steering으로는 **못 고친다**(강제하면 열화). → instill은 **training-time
(Stage 3)** 이어야 함이 실증됨. 다음 할 일은 Stage 3(LoRA) 또는 gentler-bias 스윕.

## 작업 디렉토리
`/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf/`
- `code/` — 스크립트 + 예측/분석 JSON
- `fluid_masks/` — Claude-vision 실제 마스크 + 검증 이미지
- 노트북: `../5_age_vlm_xai_instill.ipynb` (모든 결과 실행·임베드 완료)
- 문서: `추가_실험.md`(전제정정+Stage1/2 결과), `code/changelog.md`(4 엔트리), `fluid_masks/FLUID_MASKS.md`

## 실행 환경 (tier3 RetinaVLM, 중요)
```bash
source /home/ubuntu/bionexus/jgy/miniconda3/etc/profile.d/conda.sh
conda activate oct_llm
export LD_PRELOAD=/home/ubuntu/bionexus/jgy/miniconda3/envs/oct_llm/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12
export CUDA_VISIBLE_DEVICES=2          # GPU2 사용(0/1은 점유 빈번)
export PYTHONNOUSERSITE=1              # 깨진 ~/.local/sklearn 회피. env에 sklearn1.7.2·httpx 있음
cd .../paper_xai_antivegf/code && python3 -u <script>.py
```
- 분석/노트북(GPU 불필요)은 `aptos2021` env: `PYTHONNOUSERSITE=1 conda run -n aptos2021 ...`
- nohup 출력 버퍼링 이슈 → 래퍼 스크립트 방식. 성공판정은 저장 JSON을 직접 파싱해 교차검증.

## 핵심 발견 (전부 검증됨)
1. **마스크 아티팩트**: 기존 FER=0.028은 `mask[4:8,2:6]=1`(모든 eye 동일, OCT 배경) 때문. 무효.
2. **§1c (36 eye, GPU 재실행)**: 실제 마스크 FER fluid eyes **0.018→0.210 (11.4×)**. 모델은 유체를 본다.
   FER↔KG-align(Z1) Spearman **r=−0.05, p=0.77 (무상관)**.
3. **Stage 1 (token-conditioned)**: " continue" vs " stop" teacher-force paired → FER 0.2051 vs 0.2047,
   Δ=+0.0004(0.3%), Wilcoxon p<1e-4. **decision-invariant**(통계 유의하나 효과크기 무시).
   모델 ci_pred **35/35 continue**(CI-AUC≈chance).
4. **Stage 2 (attention steering)**: LLaMA `eager_attention_forward` monkeypatch로 fluid key에 +bias.
   FER 0.204→0.955(4.7×, 작동). 그러나 bias3: 21/35 uncertain, flip→stop 3건(45L/273L/345R) **전부 오답**,
   CI-AUC 0.31. bias6: 34/35 uncertain(모델 파괴). 209R(무유체)=불변(음성대조 OK).

## 데이터 / 분할
- APTOS-2021 anti-VEGF, `data.build_eye_records()` + `stratified_split(test_size=0.15, seed=42)` → **test 36 eye**
  (325L은 이미지 없음 → 실효 35). test set continue 21/35 (불균형).
- 실제 마스크: 35 eye, 6×6, `fluid_masks/masks_6x6.json` (209R=빈 마스크, 유체 음성).
- geometry: pre_len=41(상수, `<ImageHere>`가 question 앞), img_len=36(6×6 1:1 대응).

## 산출 파일 (code/)
- `gen_rollout_realmask.py` → `xai_e3_tier3_rollout_realmask.json` (36 eye raw 6×6 map + FER_real/const)
- `gen_stage1_token_attn.py` → `xai_stage1_token_attn.json` (forced continue/stop + natural decision FER)
- `gen_stage2_attn_steer.py` → `predictions_tier3_Z1_instill_b{0,3,6}.json` + `xai_stage2_fer_steer.json`
- 노트북 셀: §1b/§1c(마스크·FER), §2b(Stage1), §3b(Stage2), §5(요약 4단 결론)

## 다음 세션 할 일 (사용자 결정 대기)
**옵션 A — Stage 3 (training-time AGE-VLM, 본격):**
- LoRA on cross-attention, `L = L_LM + λ·KL(attn ‖ fluid_mask_norm)`, λ annealing.
- 학습셋: APTOS train split (~221 eye). 핵심: LM coherence loss 유지(Stage 2에서 강제주입이 문장 붕괴시킴 → 학습으로 통합 필요).
- catastrophic forgetting 방지(base frozen), 학습 후 AMD staging 유지 확인.
- 설계 상세: 노트북 §4, `추가_실험.md` Stage 3.

**옵션 B — gentler bias 스윕 (빠름, Stage 2 보강):**
- bias 0.5/1.0/1.5로 재실행(`gen_stage2_attn_steer.py`의 `BIASES` 수정) → coherence 안 깨지는 최대 steering 지점 탐색.
- bias 3에서 이미 FER 0.98이라 더 낮은 bias도 FER 충분히 올릴 가능성. uncertain 비율↓ 지점에서 결정 변화 재측정.

## 열린 한계 / 주의
- 마스크는 단일 평가자(Claude vision) coarse 6×6 — 임상의 검증 아님. 논문용은 subset 재주석 권장.
- Stage1/§1c FER은 prompt 마지막 토큰 기준(생성-시점 아님은 §1c). Stage1은 teacher-forced 결정 토큰이라 OK.
- bm_pred 파서가 RetinaVLM 서술형에서 자주 None → KG-align n이 작아짐(Stage2 bias3 KG-align n=1은 무의미).
- 모델이 거의 항상 "continue" → CI-AUC가 baseline부터 chance. 클래스 불균형 고려.

## git
- branch `main`, 커밋 안 함(사용자 요청 시에만). 변경: 노트북 + paper_xai_antivegf/ 신규 파일 다수.
