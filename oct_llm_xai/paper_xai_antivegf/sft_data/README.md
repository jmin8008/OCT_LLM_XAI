# SFT dataset — multi-hop KG CoT + counterfactual pairs

> 작성 2026-06-03. Stage 3 instill 학습/평가용. 입력 = **pre-treatment 이미지 ONLY.**

## 설계 (사용자 결정 2026-06-03)
모델이 이미지만 보고 **[시각소견 → 병태생리 → 예후예측(ΔCST) → 결정]** 다단계 CoT를 출력하도록 SFT.
예후(ΔCST)는 입력 힌트가 아니라 **예측 타깃** — 단일 이미지로 학습 불가하면 그 자체가 "정보론적 한계" finding.

각 노드는 **자기 GT로 감독**: Visual=image biomarker GT, Prognosis=실측 ΔCST 범주, Decision=기록 라벨.
연결 서사는 결정론이 아니라 가이드라인-기반 → 반응/결정이 어긋나는 77 eye는 모순이 아니라 정직하게 렌더("유체 있으나 반응해 안정→stop").

## 파일
| 파일 | 내용 |
|---|---|
| `sft_kg_cot.json` | **429 rows** = factual 218 + counterfactual 211 |
| `occlusion_worklist.json` | CF 대상 211 eye (유체 양성) |

### row 스키마
`id, eye_id, split, type(factual/counterfactual), image, prompt, target, nodes_gt{biomarkers,pathophysiology,prognosis,delta_cst,decision}, guideline_suggestion, response_decision_divergent, cf_contrast`

## 분기 분포 (factual 218)
| 병태생리 → 결정 | n |
|---|---|
| active exudation → continue | 135 |
| active exudation → **stop (divergent)** | 76 |
| dry → stop / continue | 1 / 1 |
| PED-only → continue / stop | 3 / 2 |
- decision: continue 139 / stop 79. divergent(정직 서사) 77. prognosis GT: marked 70/partial 63/minimal 49/worsening 36.

## Counterfactual pairs
`occluded/<eye>.png` (유체 cv2-inpaint 제거→dry) → target "Decision: stop". 211쌍 중 **flip-contrast 의미있는 건 135**(factual=continue). 음성대조 `occluded_negctrl/<eye>.png`(비유체 영역 inpaint) → 결정 불변이어야 함.

## 생성 (재현)
1. `code/antivegf_guideline_kg_v2.json` — 다단계 Node-Edge-Node 온톨로지 + ΔCST 임계.
2. `code/gen_sft_kg_cot.py` → `sft_kg_cot.json` + `occlusion_worklist.json`.
3. `code/occlude.py` → `fluid_masks_v2/occluded/` + `occluded_negctrl/` (cv2 inpaint; ⚠️ `PYTHONNOUSERSITE` 없이 실행 — cv2가 ~/.local).

## LoRA 4-arm ablation (carts) — `code/lora_ablation.py`, `eval_ablation.py`
frozen base 1 + 어댑터 4. A=zero-shot, B=L_LM(factual), C=L_LM+λ·KL(attn‖fluid), D=L_LM(factual+CF).
**헤드라인 지표 = counterfactual FLIP-RATE**(occlude fluid→결정 flip) − negctrl flip-rate. + CI-AUC, biomarker-node acc, **prognosis-node acc(vs ΔCST)**, Text-KG align.
데이터/LoRA 배선은 `--dry-run`으로 검증됨(통과). 모델 forward·target-token CE·attn-capture는 **TODO(GPU, oct_llm env)**. RetinaVLM LoRA 가능(mini_gpt4 llama_model wrap).
