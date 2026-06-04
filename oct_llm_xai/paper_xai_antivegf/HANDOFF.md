# HANDOFF — XAI-Instill Vision Capabilities (Anti-VEGF CI)

> 갱신 2026-06-04. 다음 세션 인계용. 직전 커밋 `561206a` (main).
> 미해결 작업 = GitHub `jmin8008/OCT_LLM_XAI` 이슈 **#1~#4**.

## 한 줄 요약
RetinaVLM의 grounding⊥decision(decoupling)을 **training-time instill(LoRA)**로 고칠 수 있나
검증. 결론 윤곽: **보이는 concept(유체)는 instill 가능하나, 단일 pre-image로 anti-VEGF
continue/stop 결정을 instill하는 건 정보론적으로 불가** (라벨=치료반응 함수).

## 작업 디렉토리 / 환경
`/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf/`
- **tier3 GPU(학습/추론):** `conda activate oct_llm` + `export LD_PRELOAD=.../envs/oct_llm/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12` + `PYTHONNOUSERSITE=1` + `CUDA_VISIBLE_DEVICES=2`. nohup은 래퍼스크립트 + `python3 -u`.
- **분석(GPU불필요):** `PYTHONNOUSERSITE=1 conda run -n aptos2021 ...`
- **예외:** `occlude.py`(cv2 inpaint)는 cv2가 `~/.local`에 있어 **`PYTHONNOUSERSITE` 없이** 실행.

## 지금까지 한 것
- **fluid_masks_v2/** — 218 eye 전체, **12×12** fluid 마스크(Claude-vision subagent 16 병렬, GT mismatch 0) + 메타데이터 + continue/stop 라벨 + `occluded/`·`occluded_negctrl/`(cv2 inpaint). source-of-truth = `masks_12x12.{json,npz}`, `metadata_v2.json` (렌더 이미지는 .gitignore, 스크립트로 재생성).
- **KG v2** `code/antivegf_guideline_kg_v2.json` — 다단계 Node-Edge-Node(visual→pathophysiology→prognosis(ΔCST)→decision). prognosis는 실측 ΔCST를 **예측 타깃**으로(입력 힌트 아님).
- **SFT** `sft_data/sft_kg_cot.json` — 429행(factual 218 + counterfactual 211). divergent 77 eye 정직 렌더. 생성기 `code/gen_sft_kg_cot.py`.
- **LoRA 4-arm**: `code/train_lora_b.py`(arm B/D), `code/eval_lora_b.py`, 스캐폴드 `lora_ablation.py`/`eval_ablation.py`. 어댑터는 .gitignore(재학습 ~30s).
- **B(L_LM), D(L_LM+counterfactual) 학습+eval 완료** (RetinaVLM tier3). 결과 `sft_data/eval_{A_baseline,B_sft_text,D_counterfactual}.json`.

## 핵심 결과 (test 35)
| 지표 | A base | B (SFT) | D (CF) |
|---|---|---|---|
| continue_rate | 0.80 | 1.00 | 0.00 |
| biomarker-node acc | 0.45 | **0.81** | 0.48 |
| prognosis-node acc (vs ΔCST) | 0.06 | 0.26 | 0.14 |
| (majority) | — | 0.31 | — |
| CF flip-rate | 0.11 | 0.0 | None* |
\* D는 clean도 전부 stop이라 flip 분모=0.
- prognosis-node ≤ majority → **단일 이미지 정보론적 한계 실증**.
- decision은 라벨 prior로 collapse(B→continue, D→stop) → **결정 grounding 불가**.
- biomarker concept은 학습됨(0.45→0.81) → 보이는 것은 instill 가능.

## 다음 작업 (우선순위 = 이슈 #1~#4)
1. **(#1, 최우선) occlusion 지각가능성 체크** — RetinaVLM(base/B)이 clean vs occluded vs negctrl에서 biomarker/decision이 실제 달라지는지. 못 구분하면 arm D 신호 무효 → D collapse 해석 confound 해소. 추가 학습 불필요(`eval_lora_b.py` 생성 경로 재사용, 같은 eye 3이미지 출력 비교).
2. **(#4) counterfactual 재설계** — occlusion 강화(더 큰 영역/zero-out) + CF row class balance + collapse 시 flip-rate 측정 개선 → arm D 재학습.
3. **(#2) arm C(attention-KL) 완주** — `KL(rollout_attn‖fluid_mask)` 배선. 주의: image token grid=36(6×6)이라 12×12 마스크 → 6×6 다운샘플 필요. (예상: 실패 = attention≠explanation 확증.)
4. **(#3) 다중 백본 확장** — LLaVA-Med(tier2)·Qwen3.6(tier1c)에 동일 instill+eval로 정보론적 한계·concept-grounding 일반성 검증. HF LlavaNext는 peft 표준.

## 먼저 읽을 것
`sft_data/README.md`, `code/changelog.md`(맨 끝 2026-06-03 항목), eval JSON 3개,
`code/{train_lora_b.py, eval_lora_b.py, gen_sft_kg_cot.py, occlude.py}`.

## git
- branch `main`, 직전 커밋 `561206a`. origin=github(jmin8008/OCT_LLM_XAI). gitlab remote는 프로젝트 moved(405) → 새 경로로 갱신 전엔 push 불가.
- 무거운 산출물(가중치/렌더 이미지/heatmap)은 `.gitignore` — 스크립트로 재생성.
