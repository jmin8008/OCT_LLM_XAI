# Changelog — paper_xai_antivegf/code

## 2026-05-30
- **Phase 0 (docs):** KG 축을 KAD/DQN 모방 → **Wang 2025 (Sensors 25:6879) neuro-symbolic 가이드라인 KG** 이식으로 전환. DESIGN/EXPERIMENTAL_PROTOCOL/PAPER_OUTLINE 3문서 갱신. ROCO(E8) 조건부/미구현 강등(clinician mask 게이트). k-fold 코드만(주석) 정책 명시. `references.bib` 생성(Wang 2025 외).
- 결정사항: KG=경량 symbolic 규칙엔진 우선(신경 인코더 stub), Tier1=Qwen3-VL-8B-Instruct, 범위=Phase 0–4 일괄.
- **Phase 1:** `metrics.py`(score_va 신규·score_cst 이식·compute_auc·bootstrap_ci·delong_test·jonckheere_terpstra·aggregate_subtasks), `data.py`(APTOS case+pic 로드 221 eyes, center 검출 이식, eye-level stratified split 187/34; `make_kfold_splits` 작성·호출 주석). 자가검증 통과(score_va=0.75, score_cst=0.5).
- **Phase 2:** `models.py`(3-tier 공통 VLMBackend, lazy 로딩: Qwen3-VL/LLaVA-Med/RetinaVLM), `prompts.py`(Z0/Z1/F2 + CI/biomarker/VA/CST 파서, biomarker negation-before 보강), `infer.py`(안구단위 추론 루프 + RAG 주입 훅). 파서/레지스트리 검증 통과.
- **Phase 3:** `antivegf_guideline_kg.json`(노드/엣지/규칙, UMLS CUI, confidence weight) + `kg.py`(symbolic forward-chaining, RAG 렌더, decision drivers, 신경 인코더 stub). R3 규칙 수정(완전 dry만 stop). KG 어서션 전부 통과.
- **Phase 4:** `rollout.py`(attention rollout + image map), `saliency.py`(fluid_energy_ratio·label_conditioned_concentration + GradCAM 어댑터 stub), `kg_align.py`(Text–KG/Attn–KG, Wang >85%/>90% 기준선), `roco_stub.py`(E8 미구현, clinician mask 게이트). E7 어서션 통과(text_align=0.5, attn=1.0).
- **환경 이슈:** `~/.local` 깨진 sklearn → 모든 실행 `PYTHONNOUSERSITE=1` 필요(troubleshoot.md).
- **데이터 위생 경고:** `reference/sensors-25-06879.pdf` 및 일부 Bash 출력에 본문 아닌 편집자 주석/지시문류 오염 라인 관찰 → 무시하고 진짜 내용만 사용. 단위검증은 어서션으로 교차확인.
## 2026-05-30 (저녁) — 노트북 조립 + Tier3 실제 실행
- 노트북 `4_antivegf_vlm_spectrum_xai.ipynb` 조립(27셀, code 13) — `code/` 모듈 import, RUN_VLM 플래그, 예측 JSON 소비. nbconvert --execute --inplace 로 **실제 출력 임베드(13/13 셀)**.
- `models.RetinaVLMBackend` 로딩 확정(legacy nb2 패턴: hydra `default` config + dequantized fp16 ckpt 16GB + sys.modules 네임스페이스 격리). env: oct_llm + nvjitlink `LD_LIBRARY_PATH` 우회(troubleshoot.md).
- **Tier3 RetinaVLM zero-shot(Z0) 실제 결과 (36 eyes test):**
  - CI-AUC **0.452** (95% CI 0.38–0.50), F1 0.704 — continue 과예측(33/36). zero-shot VLM = 약한 분류기(H1 서사 부합).
  - VA-tol **0.111**(n=18), CST-tol **0.20**(n=15) — 약한 회귀(vs BlueSky VA 0.32/CST 0.59, CNN 0.606).
  - E2 biomarker / E7 Text-KG: **resolved 행 부족**(RetinaVLM 서술형 답변이 present/absent 파서와 불일치) → Z1(biomarker-guided) 프롬프트·파서 보강 필요.
  - E5 hallucination: dry eye 1개뿐 → 검정력 부족.
  - KG-RAG ablation(`predictions_tier3_Z0_kg.json`): KG 주입 시 continue 35/36 (판별 개선 X).
- **남은 작업:** Tier1 Qwen3-VL / Tier2 실행(tier 비교 H1/H2/H5), Z1 프롬프트로 biomarker·E7 채우기.
- (구 메모) 노트북 코드셀 조립 — `code/` 모듈 import → E1~E7 + 도표를 셀에서 실행. **`experiments.py`는 만들지 않음**(사용자 결정: 실행 로직은 노트북에 집중, `code/`는 라이브러리 역할). 실제 VLM 추론은 가중치+GPU 필요 → subagent 위임.

## 2026-05-30 (심야) — Tier3 재현 + Tier1 Qwen3-VL 실행 (GPU 2, subagent)
- **models.py RetinaVLMBackend 보강(3건):** (1) `load()`에서 SpecialistVLMs `models/`·`run/`이 __init__.py 없는 namespace package라 이 프로젝트 `models.py`와 충돌 → sys.modules의 `models*`/`run*` 임시 evict + `types.ModuleType`로 namespace package 명시 등록 + `spec_from_file_location`으로 retinavlm_wrapper 직접 로드, `finally`에서 원복. (2) `generate()`/`_img_tensor()`에 `_to_gray()` 추가 — RetinaVLM ResNet 인코더는 단일채널 grayscale 요구인데 data.representative_pre_bscan은 RGB 반환 → `RuntimeError: expected input[1,3,192,192] to have 1 channels` 해결(노트북의 `convert('L')` 일치). (3) `generate()`를 `_inner.query()` → `self.model.forward()`로(forward가 이미지 변환 담당). 상수 RETINAVLM_SUBFOLDER/RETINAVLM_DEQUANT_CKPT 추가. QwenVLBackend는 무수정.
- **환경 블로커 3건(troubleshoot.md):** (a) oct_llm torch `libcusparse.so.12: undefined symbol __nvJitLinkCreate_12_8` → `LD_PRELOAD=.../oct_llm/.../nvidia/nvjitlink/lib/libnvJitLink.so.12`(해당 심볼 보유본). LD_LIBRARY_PATH로는 불충분, PRELOAD 필수. (b) `PYTHONNOUSERSITE=1`로 깨진 user-site sklearn 회피 시 env에 httpx/urllib3/annotated-doc/rich/pandas 부재 → env에 설치(transformers는 env 5.3.0 사용). (c) vllm env는 torch/transformers 전무 → `torch==2.8.0 torchvision transformers==4.57.1 accelerate qwen-vl-utils` 설치. 모두 env bin/python 직접 호출(conda run은 exit code/stdout 비신뢰).
- **Tier3 RetinaVLM (oct_llm) 재현:** smoke(3) OK → 전체 36 eyes. `predictions_tier3_Z0.json`(에러 1=325L no_pre_bscan; ci_pred: continue 33/stop 2/uncertain 1; ci_text 35/35; Z0 109s), `predictions_tier3_Z0_kg.json`(continue 35/uncertain 1; KG가 193L·273L를 stop→continue로; 102s). bm_pred는 RetinaVLM 서술형이라 concrete present/absent call 2/36(파서 한계, 로드 실패 아님).
- **Tier1 Qwen3-VL-8B-Instruct (vllm, torch 2.8.0+cu128, transformers 4.57.1):** QwenVLBackend 무수정 동작. smoke(3) OK, `predictions_tier1_Z0.json`(에러 0, ci_pred 전부 continue, ci_text 임상적으로 일관, bm_pred dict 생성, VA/CST 숫자 파싱). 가중치 최초 다운로드 포함 108s.
- Tier2 LLaVA-Med: 미실행(외부 llava lib, 지시대로 skip).
- **위생:** Bash stdout 오염·간헐 무응답·wrapper(0) vs inner(1) exit 불일치 빈번 → 모든 성공 판정은 저장 JSON을 env python으로 직접 파싱(rows/errors/필드 비어있지 않음)해 교차검증.

## 2026-06-02 — Fluid mask 재구축 (Claude-vision) + FER 아티팩트 정정
- **문제 발견:** FER 계산에 쓰이던 fluid mask가 `mask[4:8,2:6]=1`(8×8 하단-중앙) 상수 — 모든 eye 동일, OCT에서 맥락막/배경 영역. eye별 실제 유체 위치(상단/중단/망막 띠 내부) 미반영. 4_antivegf_analysis.ipynb·5 노트북 모두 이 가짜/placeholder(np.zeros) 사용. → "FER=0.028, 모델이 유체를 안 본다(text-bias)" 결론은 마스크 위치 아티팩트였음.
- **해결:** 사용자 지시("마스킹 데이터를 너가 직접 판단해서 만들어")대로 Claude-vision으로 36 test eye 중 35개(325L 이미지 없음) pre-injection 중심 B-scan을 직접 판독, **eye별 6×6 유체 마스크** 주석(saliency map_shape와 동일 프레임). SAM·CNN 미사용. 산출: `fluid_masks/{masks_6x6.json,.npz, manifest.json, annot/, overlay/, ALL_overlays_montage.png, FLUID_MASKS.md}`. fluid-present 34/35, 209R(HRF only)=빈 마스크(음성 대조), 평균 5.3 cells/eye.
- **예비 결과:** raw 6×6 attention map 보유 8 eye(xai_e3_tier3_maps.json) FER 재계산 → 상수마스크 0.016 → 실제마스크 **0.204 (평균 12.8×↑, 115R 0.42 최대)**. RetinaVLM attention은 실제 유체에 상당히 쏠려 있음. "유체를 안 본다" 결론 부분 반박.
- **노트북:** `5_age_vlm_xai_instill.ipynb` §1b 추가(실제마스크 FER 재계산, 실행·임베드 완료). Stage1/2 stub의 상수 heuristic·CNN-GradCAM 의존을 `real_masks` 로딩으로 교체(RUN_VLM=True 시 실제 마스크 사용). `추가_실험.md` §0-0 전제 정정 블록 추가.
- **다음(GPU):** tier3 rollout 재실행 시 `map_values`(6×6)를 36 eye 전부 저장 → 실제 마스크로 FER↔KG-align Spearman r 재검증(현 r=0.057은 가짜마스크 기반, 무효 가능성).

## 2026-06-02 (속편) — 36 eye rollout GPU 재실행 + FER↔KG 재검증
- **gen_rollout_realmask.py** (oct_llm env, GPU2): 36 eye 전부 RetinaVLM attention rollout 재생성(Z1, query=-1=마지막 prompt 토큰), raw 6×6 map_values 저장 → `xai_e3_tier3_rollout_realmask.json` (35/36, 325L no image). 실행: LD_PRELOAD nvjitlink + **PYTHONNOUSERSITE=1**(env에 sklearn 1.7.2·httpx 모두 존재 — 기존 "PYTHONNOUSERSITE 금지" 노트는 stale. 깨진 ~/.local sklearn(__check_build 빈 디렉토리)이 transformers import 시 circular import 유발하던 것을 회피).
- **결과(핵심):** ① FER fluid eyes const=0.018 → **real=0.210 (11.4×↑)** — 모델은 유체를 본다, "안 본다(text-bias)"는 마스크 아티팩트. ② **그러나 FER_real↔KG-align Z1 여전히 무상관** Spearman r=-0.051 p=0.77 (FER|aligned 0.198 vs misaligned 0.212, n.s.). → 시각 grounding은 존재하나 텍스트 임상 정합과 분리(decoupling). 가설 재정의: instill=보게 만들기(이미 봄)가 아니라 decision 시점 grounding↔결정 인과 결합.
- **노트북 5:** §1c 셀 추가(36-eye 상관 재검증, 실행·임베드). §1b는 8-eye 검증용 유지.
- **문서:** 추가_실험.md §0-0, fluid_masks/FLUID_MASKS.md 최종 결과로 갱신.
- **한계:** FER이 prompt 마지막 토큰 기준(생성-시점 아님). mini_gpt4.attention()은 forward-only → "continue/stop" 생성-토큰 attention은 generate 중 hook으로 별도 캡처해야(Stage 1 본실행). 단일 평가자(Claude) coarse 6×6 마스크.

## 2026-06-02 (Stage 1) — token-conditioned attention 실행
- **gen_stage1_token_attn.py** (oct_llm/GPU2): mini_gpt4.attention()이 forward-only라 answer_preamble로 결정 단어를 teacher-force하는 2-pass 기법. 35 eye 각각 " continue"/" stop"를 강제 답변 토큰으로 넣고 그 토큰의 rollout-attention FER(실제 마스크) paired 비교 + 자연 생성 결정-토큰 FER. → `xai_stage1_token_attn.json`.
- **결과:** FER@continue=0.2051 vs FER@stop=0.2047, paired delta=+0.00041 (FER의 0.31%, max 0.0016). Wilcoxon p=3.65e-5(유의)이나 효과크기 무시 수준. ci_pred 35/35 continue(stop 0개, CI-AUC~0.48). FER@natural-decision↔KG-align r=-0.063 p=0.72.
- **해석:** image attention은 거의 decision-invariant — 결정 토큰을 바꿔도 같은 곳을 봄. §2 H1("약한 vestigial grounding 존재, 결정과 거의 분리")에 정확히 해당. instill 표적: 0.3% vestigial fluid→continue 결합을 결정-구동 수준으로 증폭.
- **노트북 5:** §2b 셀 추가(실행·임베드). 기법 메모: backend.attention()은 answer_preamble 미노출 → inner.attention(img_t,[prompt],answer_preamble=[word]) 직접 호출. inputs_tokens는 image 위치=-1 인라인, decision 토큰 인덱스=T축 인덱스 그대로.

## 2026-06-02 (Stage 2) — inference-time attention steering 실행
- **gen_stage2_attn_steer.py** (oct_llm/GPU2): transformers 5.3.0 `eager_attention_forward` monkeypatch — pre-softmax 점수의 fluid image-token key 열(=pre_len(41)+fluid_idx)에 +bias 주입(실제 마스크). bias 0/3/6 스윕. 후크가 generate의 모든 디코드 스텝에 적용되도록 attention_mask 대신 점수에 직접 주입(mask가 None일 수 있어서). pre_len=41 상수(template상 <ImageHere>가 question 앞), img_len=36 확인.
- **결과:** FER bias0=0.204→bias6=0.955 (4.7×, steering 완전 작동). 그러나 임상 출력은 **열화**: bias3 resolved 14/35(21 uncertain), continue→stop flip 3건(45L/273L/345R) **전부 오답**(y_continue=1), CI-AUC 0.312(<chance). bias6 34/35 uncertain(모델 파괴). 209R(무유체 마스크)=전 bias 불변(음성대조 OK). KG-align "1.0"은 n=1 아티팩트.
- **결론:** inference-time steering으로 decoupling 해결 불가 — attention을 강제로 fluid에 몰면 틀린 결정+비문 유발(시각 grounding이 올바른 임상 논리를 담지 않음, Stage1과 일치). → instill은 training-time(Stage 3)이어야: fluid-grounded attention을 일관 추론으로 통합하도록 학습(LM coherence + attention guidance 동시).
- **산출:** predictions_tier3_Z1_instill_b{0,3,6}.json, xai_stage2_fer_steer.json. 노트북 5 §3b 추가(실행·임베드), §5 요약 4단 결론으로 갱신.

## 2026-06-03 — fluid_masks_v2: 전체 cohort 12×12 grounding+결정 데이터셋
- **배경:** attention-KL 주입 폐기("attention is not explanation"; Stage 2가 FER 0.95로 올려도 결정 열화 → decoupling은 inference로 못 고침). 마스크가 토큰격자(6×6)에 묶일 이유 소멸 → 12×12 상향. v1은 test 35 eye만 있어 학습 불가(train 183 eye grounding 라벨 부재) → 전 cohort 확장.
- **`gen_dataset_v2.py`(신규):** 218 eye(train 183/test 35, 이미지 없는 196R/215R/325L 제외) 렌더 — 2/98% 대비 스트레치 + 360px 업스케일 clean + 12×12 격자(축라벨) grid. `metadata_v2.json`(결정 라벨 continue_injection + dx/drug/age/gender/preVA/preCST/VA/CST/ΔCST/ΔVA/biomarkers/fluid_types/split) + `annotation_manifest.json`(GT fluid_types 제약).
- **주석:** 16 배치 분할 → **Claude vision subagent 16개 병렬**(`ANNOTATION_GUIDE.md`: SRF/IRF/PED 판독 정의 + GT 타입 제약 + 12×12 [row,col] 스키마). 결과 `annot_parts/batch_*.json`.
- **`assemble_v2.py`(신규):** `masks_12x12.json/.npz` + overlay/ + 몽타주 + GT 교차검증 + 메타 병합(fluid_cell_count/mask_confidence/types_seen/note).
- **결과:** 218 마스크, **GT mismatch 0**(음성대조 114R/209R 빈 마스크, 나머지 216 eye ≥1 cell). 평균 10.2 cell/eye(144 중), 범람·좌표오류 0. 신뢰도 high 37/med 142/low 39. 메타 결측 0.
- **한계:** 단일 평가자(Claude vision) coarse — 12×12는 셀당 면적↓로 개별 노이즈↑(false precision 주의). 대표 1 B-scan. 결정 라벨은 APTOS 전용 → 218이 라벨 상한. 상세 `fluid_masks_v2/README.md`.

## 2026-06-03 (이어서) — 다단계 KG CoT SFT + counterfactual + LoRA 4-arm 스캐폴드
- **방향 전환 근거:** attention-KL 주입 폐기 재확인(plausible≠faithful). KG 정적룰(유체→continue)이 라벨과 62%만 일치(stop 라벨 76개가 유체 양성) → 라벨은 pre-scan biomarker 함수가 아니라 치료반응 함수. 검증: post_cst→continue AUC 0.762, ΔCST AUC 0.681(ΔVA 무용 0.456). Wang 2025 대조: 그 논문 KG도 confidence-weighted 룰 CSV 수준 + n=10(test 2)라 우리가 빈약하지 않음. Wang의 temporal/counterfactual 정신을 명분 삼아 개정.
- **사용자 결정:** 입력=pre 이미지 ONLY, prognosis(ΔCST)는 **예측 타깃**으로 다단계 CoT 구성(예측 오르면 good, 안 오르면 "단일이미지 정보론적 한계" finding = win-win).
- **`antivegf_guideline_kg_v2.json`(신규):** 4-layer Node-Edge-Node(visual→pathophysiology→prognosis→decision) + ΔCST 임계(marked/partial/minimal/worsening) + 가이드라인룰. 각 노드 자기 GT 감독.
- **`gen_sft_kg_cot.py`(신규):** 다단계 CoT 타깃 생성. 분기는 실제 라벨, prognosis는 실측 ΔCST 범주. 연결 서사 비결정론적 → divergent 77 eye(유체→stop 76)는 모순 아닌 정직 렌더. Step4 로직 patho-aware로 정정(favorable 단정 제거, PED-only 별도 분기). → `sft_data/sft_kg_cot.json` 429행(factual 218 + CF 211), `occlusion_worklist.json`.
- **`occlude.py`(신규):** masks_12x12로 유체셀 cv2.INPAINT_TELEA 제거(검정박스 금지=지름길 차단) → `occluded/` 211. 동일연산 비유체 영역 = `occluded_negctrl/` 211(음성대조). ⚠️ cv2가 ~/.local에 있어 `PYTHONNOUSERSITE` **없이** 실행해야 함(occlude는 sklearn 불필요). QC 몽타주 자연스러움 확인.
- **`lora_ablation.py`+`eval_ablation.py`(신규, 스캐폴드):** frozen base 1 + 카트리지 4(A zero-shot/B L_LM/C +KL(attn‖fluid)/D +counterfactual). LoRA targets q_proj·v_proj. 헤드라인 지표=**counterfactual flip-rate − negctrl flip-rate**(faithfulness), + CI-AUC·biomarker-node·**prognosis-node(vs ΔCST)**·Text-KG. 데이터/LoRA배선 `--dry-run` 검증 통과(A183/B183/C183+masks/D360, eval test35+34 parser 35/35). forward·target-CE·attn-hook은 TODO(GPU oct_llm). RetinaVLM LoRA 가능 확인(mini_gpt4:323 llama_model lora-wrap).
- **남은 한계:** prognosis 노드 학습가능성 미검(GPU eval로 측정 예정). ΔCST 지름길은 입력에서 배제됨(pre-only). v1 정적KG vs v2 다단계 ablation은 미구현(요청 시).
