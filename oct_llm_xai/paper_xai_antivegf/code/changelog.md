# Changelog — paper_xai_antivegf/code

## 2026-06-06 — v0.3 전 백본×arm 전수 재실행 완료 + 매트릭스 조립 (3백본 일반성 확정)
- **실행:** `verify_v3.py` 18/18 PASS 재확인 후 tier3→tier2→tier1c 순 전수 재실행(GPU2 직렬, 래퍼+`python3 -u`, oct_llm). 산출 `eval_{tier}_{arm}{,_meta}.json` 17셀(tier1c C=N/A) 전부 v0.3. 로그 `run_{tier3,tier2,tier1c{,_fix}}_v3.log`. 매트릭스 `sft_data/matrix.{md,json}`.
- **⚠️ tier1c QLoRA OOM 버그 발견·수정(핵심):** Qwen3.6-27B B/D/B_meta 학습이 전부 CUDA OOM(~78GB). 원인 = **gradient checkpointing 무력화**: `backbones.py load()`가 모델을 `.eval()`로 두는데 HF는 `self.training=True`일 때만 checkpointing 실행 → 27B 전 레이어 activation 보관 → v0.3의 긴 CoT 타깃에서 OOM(v0.2는 타깃 짧아 우연 통과). **수정:** `QwenVL.attach_lora` quantized 경로에 `model.train()` + `config.use_cache=False` + `gradient_checkpointing_enable(use_reentrant=False)` + `enable_input_require_grads()` 추가. 재실행 결과 B/D/B_meta **skip=0, OOM 0, loss 1.1→0.03** 정상 수렴(B_meta도 통과 — 더는 env-limited 아님). tier2/tier3 무영향(quantized 분기 한정).
- **⚠️ 부수 버그 2건:** ① 1차 래퍼의 `exit=$?`가 `$(date)` 뒤 평가돼 python 아닌 date 종료코드(0) 보고 → 재실행 래퍼는 `rc=$?`를 date 전에 캡처. ② OOM으로 어댑터 미저장 시 B/D eval이 **구식 v0.2 어댑터** 로드 → 무효결과. 해당 `eval_tier1c_{B,D}.json`은 `.stale_v2adapter_*`로 백업 후 재학습 어댑터로 재eval. ③ `assemble_matrix.py`가 tier3 A/B/C/D를 **레거시 v0.2 파일명**(eval_A_baseline 등)에 매핑하고 있어 stale 유입 → `eval_tier3_{A,B,C,D}.json` 균일 네이밍으로 교정 + response 지표(resp3/rMaj/rGPbal) 매트릭스에 추가.
- **헤드라인(3백본 일치, v0.3):** ① **보이는 concept(biomarker)는 SFT로 instill — 백본 무관**(tier3 0.32→0.81, tier2 →0.75, tier1c 0.80→0.83; tier1c A_meta zero-shot 1.0). ② **continue/stop 결정은 collapse — 백본 무관**(B/C continue_rate≈1.0[tier3 0.94/tier2 1.0/tier1c 1.0], D→stop[tier3·tier2 0.0]; decision balAcc≈0.5). ③ **종합 responder(good/poor)는 이미지-only로 chance 초과가 tier3 B(0.613)뿐**, 나머지 0.5. ④ **메타데이터가 핵심 레버**: tier2 **A_meta가 balAcc 0.619·rGPbal 0.758·resp3 0.824**로 도약(pre-meta에 responder 신호 존재=한계 일부는 정보부족). 단 **B_meta(SFT+meta)는 0.5로 도로 collapse** → SFT text-prior가 메타 신호를 덮음(주목 finding). ⑤ tier1c 특이: D가 all-stop이 아닌 continue 1.0(cfFlip 1.0/faithGap +0.118), B_meta continue 0.714(부분 비collapse) — 백본별 prior 차이.
- **분석 전문:** `sft_data/MATRIX_ANALYSIS_v0.3.md`(라벨분포 n=35, 확고결과 A/B/C, 메타×SFT D/E, tier1c 이상치, over-claim 주의, 다음작업 5건).
- **상태:** v0.3 전수 매트릭스 완성 + 분석 문서화. **다음작업(우선순위):** ①[최우선] A_meta>B_meta 역전(E) tier2 case-level diff + preCST-only vs 전체 메타 ablation ②Qwen perceptibility_check 재실행(tier1c D 신호 검증) ③메타-천장 대비(meta_ceiling responder) ④핵심셀 bootstrap CI/permutation(n=35 n.s. 표기) ⑤paper.tex 표/서사 갱신.

## 2026-06-04 — v0.3 Gemini 검토 반영(조정1) + 무결성 18/18 + RetinaVLM sanity-check 가동
- **Gemini 검토 결정:** composite OR조건 유지·Step2 독립 유지·dry=observation 유지(모두 확정). **조정1만 반영**: divergent stop+good의 가정법("if given would respond")이 "좋아질 건데 왜 중단?" 모순 → **decision-aware 서사로 교정**. continue=예측형("Under continued anti-VEGF the expected course is..."), stop+good=회고형("Cumulative anti-VEGF achieved a good response — ... — so therapy was stopped, stable prognosis"), stop+poor("response was poor ... stopped given limited benefit; guarded prognosis"). build() Step4 분기 수정.
- **fluid_resolution 버그 수정:** pre-fluid 없는 dry eye가 "resolved"로 오기록 → has_fluid 게이트 추가해 "na". (resolved 13→10, dry 전부 na.)
- **재생성·검증:** sft_kg_cot.json v0.3 (429행, response good150/poor66/no_active2, decision 139/79, divergent77). `verify_v3.py` **18 PASS/0 FAIL**(CoT순서·KG일관·decision→response·lineage·누설0·CF짝·split). parse_ci 218/218, parse_prognosis 197/218.
- **sanity-check 가동:** `harness.py --tier tier3` v0.3 데이터로 B 학습 → A/B eval (eval_tier3_{A,B}.json). 목적: 인과 재정렬된 사슬에서 "결정 분류 가능 vs 종합반응 한계"가 분리되는지 1백본 확인 후 확장. 로그 tier3_v3_sanity.log.

## 2026-06-04 — CoT/KG v0.3 인과 재설계 (사용자 명령: decision↔prognosis 선후 교정)
- **배경:** v0.2는 [visual→prognosis→decision]로 "미래 예후를 점친 뒤 결정"하는 인과 역전 구조. 사용자(+Gemini 검토) 지적: 임상 현실은 [현재상태→결정→그 결정의 예상반응]. 또 (a) 입력에 진단명을 주면 Step2(pathophysiology)가 눈속임(shortcut), (b) 신생혈관 반응은 CST 단독이 아니라 ΔCST+ΔVA+fluid 종합으로 판단해야 함.
- **명령 구현 4건:**
  1. **입력 프롬프트에서 진단명 제거** — Step2가 이미지만으로 활성도 추론(진짜 inference).
  2. **CoT 순서 뒤집기**: Step3=**임상결정**(guideline, image+meta→continue/stop) → Step4=**종합 치료반응**(ΔCST+ΔVA+fluid → good/poor responder).
  3. **KG v0.3 엣지 재설계**: layers [visual→pathophysiology→**decision→response**], 엣지 `decision→expected_response`(continue→good/poor_responder, stop→no_active_disease). `response_definition`에 composite rule 명시.
  4. **`gen_sft_kg_cot.py` build() 전면 수정** — 결정 서사는 사후정보 누출 제거(divergent는 "clinical factors beyond this single scan"로 정직 렌더), Step4는 anatomic(ΔCST bucket)+functional(ΔVA dir)+fluid(pre→post 해소) 종합.
- **신규 데이터원:** post-injection fluid를 pic CSV(train_anno_pic.csv, post 1374장)에서 집계 → fluid_resolution(resolved/persistent/na). composite_responder = (ΔCST≤−25 OR ΔVA≥+0.1)?good:poor.
- **재생성 결과(`sft_kg_cot.json` v0.3, 429행):** response GT good150/poor66/no_active2, decision continue139/stop79, divergent 77. 백업 `*.bak_v0.2.*`.
- **eval 파서 보강:** `prompts.parse_ci`가 추론 중 반대단어가 먼저 나와도(예: divergent "favor continuation ... was to stop") **마지막 `Decision:` 태그를 권위있게 채택** → 타깃↔GT decision 일치 142→**218/218**. parse_prognosis(ΔCST bucket) 199/218 유지.
- **⚠️ downstream (미실행, 사용자 확인 후):** ① 전 arm/백본 **재학습 필요**(v0.2 가중치는 구 데이터). 기존 11셀+메타 매트릭스는 v0.2 결과. ② eval에 **response-node(composite responder) 채점 추가** 필요(현재 prognosis=ΔCST만). ③ harness는 row['prompt'] 사용이라 새 프롬프트 자동 반영, meta augment도 "Reason step by step:" 앵커 유지돼 호환.

## 2026-06-04 — 메타데이터 ablation 계획 확정 + Part A(정보 천장) 실행
- **사용자 확정 스코프:** Part A(천장 베이스라인) 포함 + Part B = A_meta·B_meta × 3 백본(Qwen3.6 B_meta QLoRA 포함). C/D_meta 제외.
- **Part A — `meta_ceiling.py`(신규, CPU/aptos2021):** 메타만으로(age/gender/drug/preVA/preCST, 이미지 없음) 결정·예후 예측 상한. 동일 split(train183/test35). 결과 `sft_data/meta_ceiling.json`:
  - **결정(continue/stop):** logreg balAcc **0.536**/AUC **0.588**, gbm 0.524/0.578 (chance 0.50). coef: preCST 0.281·drug_avastin 0.236·drug_steroid −0.261·age 0.129. → 메타만으로도 결정은 거의 chance = **결정은 baseline 정보로 근본적 예측난(치료반응 함수)**, 메타 추가해도 결정 collapse 지속 예상.
  - **예후(ΔCST 4-class):** logreg acc **0.457** vs majority 0.314 (gbm 0.343). → 메타(특히 preCST)는 예후 신호 보유, **이미지-VLM(~0.26)·majority보다 높음**. 즉 천장이 ~0.46. **B_meta가 VLM에게 이 신호를 쓰게 하는지가 Part B 핵심.**
- **해석 프레임:** Part A 예후천장(0.46)>VLM(0.26)이므로, B_meta로 예후가 0.46 근처로 오르면 "VLM이 텍스트 메타 활용"=한계 일부는 정보부족; 안 오르면 "멀티모달 융합 실패 또는 정보론적 한계 강건". 결정은 천장 자체가 낮아(0.54) 메타로도 개선 어려울 것으로 예측.
- **Part B 실행:** `harness.py --meta`로 A_meta(eval)+B_meta(train+eval) × tier2/tier3/tier1c. `meta_ablation.log`(좀비-가드 제거, GPU 직접). 어댑터/결과 `_meta` 접미사. assemble_matrix가 A vs A_meta, B vs B_meta 자동 비교.

## 2026-06-04 — 메타데이터 ablation 추가 (사용자 요청)
- **질문 확인:** 현재 프롬프트는 진단명+이미지만. 가용 메타데이터(metadata_v2.json, 218 eyes 결측 0): age(24-95)·gender(M148/F70)·**drug(Avastin168/Tricort22/Razumab11/Accentrix9/Eylea3/기타; ⚠️Tricort·Ozurdex는 스테로이드=라벨노이즈)**·preVA(0.01-1.0)·preCST(190-1081). 원본 APTOS case CSV 10컬럼 확인 → 숨은 추가필드 없음(이전주사횟수·visit간격·시계열 부재). post VA/CST·ΔCST·ΔVA는 **outcome=라벨누설**이라 입력 금지.
- **`harness.py --meta`:** pre-treatment 메타(age/gender/drug/preVA/preCST)만 "Patient context:" 절로 프롬프트에 주입(`augment_prompt`/`apply_meta`), outcome 누설 없음. 어댑터/결과는 `_meta` 접미사(`{tier}_B_meta`, `eval_{tier}_{A,B}_meta.json`). train/eval 양쪽 적용.
- **ablation 범위:** A_meta(zero-shot)·B_meta(SFT) × 3 백본. RetinaVLM·LLaVA-Med(빠름) 먼저, Qwen3.6 A_meta(eval)·B_meta(QLoRA). `assemble_matrix.py`에 _meta 행 추가(A vs A_meta, B vs B_meta 비교). 가설: 메타 추가로 결정/예후 collapse가 완화되면 "한계=정보부족", 불변이면 "정보론적 한계 강건" — 양쪽 다 finding.
- **큐잉:** `meta_ablation.log` — 현재 Qwen3.6 B/D(PID 3756760) 종료 대기 후 자동 시작(GPU2 직렬).

## 2026-06-04 — Issue #3: 다중 백본 하네스 — LLaVA-Med 4-arm 완주 + 매트릭스 + Qwen3.6 진입
- **`backbones.py`(신규):** 백본 어댑터 추상화(load/attach_lora/enable_eager/lm_loss/attn_loss/generate/save·load_adapter). RetinaVLM(tier3)·LLaVA-Med(tier2)·Qwen(tier1c) 단일 인터페이스. arm C는 attn_kl 공통 6×6 비교격자.
- **`harness.py`(신규):** `--tier --arm --mode{train,eval}` 드라이버. eval 점수 = eval_lora_b 스키마. 어댑터 `lora_adapters/{tier}_{arm}`, 결과 `sft_data/eval_{tier}_{arm}.json`.
- **`assemble_matrix.py`(신규):** 모든 eval 셀 → 비교 매트릭스(`sft_data/matrix.{json,md}`). RetinaVLM 레거시 네이밍 + 신규 네이밍 모두 처리.
- **학습 안정화 2건:** (a) **grad-finite guard**(grad norm 비유한 시 step 건너뜀 → NaN 가중치 오염 방지, skip 카운트). (b) LLaVA-Med forward를 **bf16 autocast**로(fp16 장시퀀스 eager attention 불안정 → arm C 학습이 step30부터 NaN이던 것 해소; 재학습 결과 skip=0, KL 2.22→0.95 정상).
- **LLaVA-Med(tier2) 4-arm 완주(test 35):** A cont1.0·bm없음(base 파싱불가)·prog0; **B cont1.0·bm0.848·prog0.257(≤maj0.314)**; **C cont1.0·bm0.848·prog0.257**(KL 2.2→0.95=attention 유체로, 그러나 결정/예후 불변); **D cont0.0(→stop)·bm0.819·prog0.229**. → **RetinaVLM 패턴 완전 재현**(SFT가 biomarker는 학습, 결정은 B→continue/D→stop collapse, 예후는 majority, attention 정렬해도 불변).
- **매트릭스 헤드라인(2백본 일치):** "보이는 concept(biomarker)는 SFT로 instill되나, 단일 pre-image로 continue/stop 결정·ΔCST 예후는 정보론적 한계로 collapse/majority — 백본 무관 일반성." arm C(attention rollout)도 FER↑(학습KL↓)이지만 결정/예후 불변 = attention≠explanation 일반성.
- **Qwen3.6-27B(tier1c) 제약 발견 + 사용자 결정:** ① arch=Qwen3_5(hybrid **linear-attention**: layer_types 대부분 linear_attention/gated-delta-net + 일부 full_attention) → **attention rollout(arm C) 적용 불가**. ② 27B(54GB)+학습 activation > 단일 80GB(타 GPU 점유) → fp16 학습 OOM. **사용자 결정: Qwen3.6-27B 그대로 진행, rollout은 유연하게.** → 계획: arm A(zero-shot, full bf16 추론) + **B/D는 4-bit QLoRA(bnb 0.49.2, nf4, grad-checkpoint)**, arm C는 linear-attention이라 N/A(메서드 일반성 한계 finding). 현재 arm A eval 실행중(linear-attn torch fallback이라 느림).

## 2026-06-04 — Issue #3: 다중 백본 하네스 (de-risk 단계) — 환경/자산 확인 + attn_kl 안정화 + LLaVA-Med 경로 실증
- **자산/환경 확인:** 3 백본 전부 **oct_llm 단일 env**(LD_PRELOAD nvjitlink + PYTHONNOUSERSITE=1 + peft 0.18.1 + tf 5.3.0)에서 적재 가능 확인. RetinaVLM(dequant ckpt)✅, LLaVA-Med v1.5(saved_models 15G + LLaVA-Med fork lib)✅, **Qwen3.6-27B(HF_HUB_CACHE 52G, 15/15 shard 완전, arch Qwen3_5ForConditionalGeneration — tf5.3.0에 클래스 존재)**✅. (사용자 확인: Qwen3.6는 커스텀 HF 캐시 경로에 받아둠.)
- **사용자 결정:** 매트릭스 = 3 백본 × **4 arm(A/B/C/D 포함)** = 3×4(D도 포함).
- **`attn_kl.py` 안정화(공용 비교격자):** 미세격자(LLaVA-Med 576토큰=24×24)에서 forward KL의 `-q/p`가 p≈0에서 폭주 → grad NaN. 해결: native attention map과 마스크를 **공통 coarse 비교격자(compare_hw=6×6)로 average-pool 후 KL** + p에 uniform smoothing(1e-3) floor. 부수효과: 백본별 native 토큰해상도(36/576/dynamic)와 무관하게 **동일 6×6 해상도에서 grounding 비교**(매트릭스 공정성). RetinaVLM(native 6×6)은 사실상 무변(6→6 항등 + 미세 smoothing).
- **`probe_llavamed.py`(신규, tier2 de-risk):** 1회 GPU 적재로 LLaVA-Med arm-C 전 경로 실증 — ① LoRA(q/v_proj 8.4M) ② `output_attentions=True`(eager) forward → L_LM 1.84 + grad 유지 attention(32층, T=870) ③ image 토큰 **576개=24×24, img_start=5**(`-200` IMAGE_TOKEN 위치) ④ answer 토큰 위치(이미지 확장분 +575 shift 보정) ⑤ attn_kl KL=1.99 ⑥ backward **grad_norm=3.07(유한, NaN 해소)** ⑦ generate `'Decision: continue.'`.
- **버그 2건 수정:** (a) tf5.3.0 generate가 `cache_position`/`logits_to_keep`를 LLaVA-Med fork(4.36-era) forward에 전달 → TypeError. 런타임 monkeypatch로 흡수(학습 forward는 직접호출이라 무관, eval generate만 영향). (b) fork generate가 inputs_embeds 기반이라 tf5.3.0이 **생성 토큰만 반환** → `models.py`의 `out_ids[:, input_len:]` 슬라이스가 전부 잘라 빈 문자열 → 길이 비교 후 전체 디코드로 수정.
- **상태:** RetinaVLM 4-arm 완료, LLaVA-Med 학습+생성 경로 실증(다음=B/C/D 학습+A/B/C/D eval), Qwen는 적재 가능하나 27B output_attentions(arm C) OOM 리스크 미검증. 다음: `backbones.py`(어댑터 추상화) + 하네스 드라이버 작성 → tier2 4-arm 실행 → Qwen 시도.

## 2026-06-04 — Issue #2: Arm C (attention-rollout KL) 학습 배선 완료 + 학습/eval 완주
- **목적:** arm C 핵심 = `L = L_LM + λ·KL(rollout_attn ‖ fluid_mask)`. attention을 유체로 끌어와도 결정/예후 grounding이 고쳐지는지 training-time 검증(핸드오프 예상: FER↑ but decision collapse = attention≠explanation 확증).
- **`attn_kl.py`(신규, 재사용 코어):** 순수 torch·미분가능. ① `torch_attention_rollout`(rollout.py numpy 레시피를 torch로: head-mean → 0.5A+0.5I residual → row-norm → L층 곱, grad 유지) ② `downsample_mask_to_grid`(12×12 유체 마스크 → 이미지 토큰 격자로 **average pooling**; `F.adaptive_avg_pool2d`로 6×6 외 비정수배 격자도 지원 — 다중 백본 대비) ③ `image_attn_distribution`(answer 토큰 행들의 image-token 열 평균 → 분포) ④ `attn_kl_loss`(forward KL(q_fluid‖p_attn), 빈 마스크 eye는 None 반환=skip). **backend-agnostic**(attention 텐서+image slice+grid만 받음 → task #3 하네스 공용). CPU 단위테스트 통과(rollout shape, 12→6 mass 보존, KL 유한, **grad-flow** 확인).
- **`train_lora_b.py` 확장(arm C 추가, B/D 경로 무변):** `forward_with_attn` — `form_input`으로 [prompt|image|answer] 시퀀스 구성 후 `llama_model(..., output_attentions=True)`(eager)로 **미분가능 attention** 확보. image 토큰 위치는 `inputs_tokens==-1` 센티넬로 자동 탐지(하드코딩 X, prompt_wrap이 image 위치에 -1 기록), answer 토큰은 `targets!=-100`. `set_eager_attention`(PEFT-wrap 유무 모두 32층에 eager 설정). 마스크 비어있으면 L_LM만. `--attn-lambda`(기본 0.5).
- **`lora_ablation.py` 배선:** `compute_losses`/`attach_lora`를 train_lora_b로 위임(스캐폴드↔트레이너 단일 경로). NotImplementedError 제거. STATUS 갱신.
- **GPU smoke(2 step):** `lm=2.82 kl=2.79 grad_norm=5.74 finite=True` → attention 경로로 **grad가 LoRA q_proj/v_proj까지 흐름** 실증.
- **full train(183행×3ep, ~80s):** **KL 2.09 → ~0.9 (절반)** = attention rollout이 유체 마스크로 끌려옴(=FER↑의 학습시점 proxy). 어댑터 `lora_adapters/C_attn_guide` 저장.
- **eval(test 35, `eval_C_attn_guide.json`):** biomarker_node **0.838**(B 0.81↗), 그러나 **continue_rate 1.00(전부 continue로 collapse), prognosis_node 0.257(=B, ≤majority 0.314), cf_flip 0.0, faithfulness_gap 0.0.** → **attention을 유체로 정렬해도 결정/예후 grounding 불변** = 핸드오프 예측 확증(attention≠explanation, training-time에서도). Stage 2(inference-time steering FER 0.95에도 결정 열화)와 일관.
- **RetinaVLM 비교 매트릭스(test 35):** continue_rate A0.80/B1.00/**C1.00**/D0.00 · biomarker A0.45/B0.81/**C0.84**/D0.48 · prognosis(maj0.314) A0.06/B0.257/**C0.257**/D0.14 · cf_flip A0.11/B0/**C0**/D—.
- **다음:** (#3) attn_kl 코어를 LLaVA-Med·Qwen 하네스로 확장해 3×3(A/B/C) 매트릭스 도출. C의 eval-time rollout FER 직접측정(B 대비 FER↑ 명시)은 #3 매트릭스에 포함.
- **실행:** oct_llm + LD_PRELOAD nvjitlink + PYTHONNOUSERSITE=1 + CUDA_VISIBLE_DEVICES=2, 래퍼+`python3 -u`. 로그 `code/train_c{,_smoke}.log`, `code/eval_c.log`.

## 2026-06-04 — Issue #1: occlusion perceptibility check (D-collapse confound RESOLVED)
- **목적:** arm D는 (occluded→"Decision: stop") counterfactual 쌍으로 학습. 만약 모델이 cv2-inpaint occlusion을 **지각조차 못 하면** D의 신호는 픽셀-분리된 라벨 prior일 뿐 → "D가 전부 stop으로 collapse"는 grounding 결과가 아니라 confound. 이를 직접 측정.
- **`perceptibility_check.py`(신규):** 같은 프롬프트로 동일 eye의 clean / occluded / occluded_negctrl 3이미지를 greedy 생성(mini_gpt4 do_sample=False → 텍스트 차이는 전부 이미지 탓)하여 ① 출력 텍스트 동일성(byte-identical + difflib ratio) ② fluid biomarker(IRF/SRF/PED) present→absent 드롭률(occ vs negctrl) ③ continue→stop 결정 flip률을 arm A(base)·B(SFT 어댑터)에서 측정. 34 test fluid eye(325L 이미지 없음). 산출 `sft_data/perceptibility_check.json`.
- **결과 (핵심):**
  - **arm B(SFT, = D가 올라탄 substrate): 사실상 지각 불가.** clean vs occluded 출력이 **28/34(82.4%) byte-identical**, mean difflib sim **0.9953**, fluid biomarker 드롭 **0/34**, 결정 flip **0/34**. 예: 5L은 유체를 inpaint로 지운 뒤에도 글자까지 똑같이 "IRF, SRF, PED present" 출력. 나머지 6 eye도 sim≥0.978·드롭0·flip0(사소한 변화).
  - **arm A(base): 지각은 되나 fluid-비특이적.** 출력은 clean과 **0/34 byte-identical**(항상 변함, mean sim 0.56)이나, fluid biomarker 드롭률 occ=neg=0.118(gap_bm **0.0**), 결정 flip occ 3/27 vs neg 1/27(gap_dec **0.074**, n 극소). 즉 generic 픽셀 변화엔 반응하나 유체 제거에 특이적으로 반응하지 않음. (base는 concrete present/absent call이 6/34뿐 — 파서 한계로 biomarker test 저검정력.)
- **결론 (issue #1 종결):** **arm D의 counterfactual 신호는 무효(void).** D가 학습하는 occluded 이미지는 모델이 clean과 구별 못 하는(특히 SFT B는 82% 글자까지 동일) 입력인데 라벨만 stop으로 뒤집힘 → D가 배울 수 있는 건 라벨 prior뿐 → **all-stop collapse는 grounding/faithfulness 결과가 아니라 confound(=핸드오프가 예측한 최대 리스크 확증).** D collapse 해석에서 "결정 grounding 불가"는 유지하되, "CF flip으로 grounding 입증 실패"는 *occlusion 자체가 지각 불가*가 원인임을 명시해야 함.
- **이슈 #4 직접 동기화:** 지각 불가 원인 = (a) 12×12 셀 cv2 INPAINT_TELEA는 192×192 grayscale ResNet 인코더가 거의 무시할 미세 변화, (b) SFT가 모델을 더 text-prior화(canned CoT)해 픽셀 민감도↓. → #4 재설계 필수: occlusion 강화(더 큰 영역/zero-out/대비 파괴) + 지각가능성을 먼저 perceptibility_check로 재검증한 뒤에야 D 재학습.
- **실행:** oct_llm + LD_PRELOAD nvjitlink + PYTHONNOUSERSITE=1 + CUDA_VISIBLE_DEVICES=2, 래퍼+`python3 -u`(`/tmp/run_perceptibility.sh`), 로그 `code/perceptibility.log`. 추가 학습 없음(생성 경로만 재사용).

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
