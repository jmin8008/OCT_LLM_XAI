# 실험 프로토콜 (Experimental Protocol)

> 설계 문서. 실제 코드/학습은 승인 후 진행. 본 문서는 *무엇을, 어떤 입력/프롬프트/메트릭으로* 측정할지 확정한다.

---

## 1. 모델 스펙트럼 (general → specialist)

| Tier | 모델 | 백본 | 전문화 수준 | repo 자산 / 비고 | 역할 |
|------|------|------|-------------|------------------|------|
| **1a** | **LLaVA-v1.6-mistral-7b** | **Mistral-7B** | 없음 (자연영상) | HF `llava-hf/llava-v1.6-mistral-7b-hf` | 일반 VLM floor + **forgetting 기준선** (Tier2와 동일 백본 → 직접 비교) |
| **1b** | **Qwen3.6-27B** | Qwen3 (MoE) | 없음 (자연영상) | 외부 가중치, SOTA instruction follower | 일반 VLM 두 번째 기준선; H7에서 tier1b ≥ tier1a 예상 |
| **2** | **LLaVA-Med-v1.5-mistral-7b** | **Mistral-7B** | 의료-일반(PubMed, Mistral FT) | `SpecialistVLMs/configs/pretrained_models/llava_med_{192,384}px.yaml`, `models/llava_med.py` | 중간; **Tier1a↔Tier2 쌍이 catastrophic forgetting을 직접 정량화** |
| **3** | **RetinaVLM-Specialist** | LLaMA-3-8B | 망막-OCT 특화 | `models/retinavlm_wrapper.py`, `load_method1_save_dequantized.py`, HF `RobbieHolland/RetinaVLM` | 천장(ceiling); 훈련 내 태스크 specialist |
| **ref** | CNN/앙상블 (기존 Task 1–3) | — | 지도학습·과제특화 | `autoresearch_task{1,2,3}/` 결과 인용 | 비-VLM 상한 앵커 |

**핵심 설계 논리 — Forgetting 정량화:**
- Tier1a(LLaVA-v1.6-mistral) → Tier2(LLaVA-Med-mistral): **동일 Mistral-7B 백본**, 유일한 차이 = medical PubMed fine-tuning. 이 쌍의 성능 갭이 *의료 FT에 의한 catastrophic forgetting*을 직접 측정한다.
- Tier2 → Tier3: 추가 OCT 도메인 특화. forgetting 2단계 누적.
- **망각 축(Forgetting axes):**
  - 일반 instruction-following 능력: Tier1b ≈ Tier1a > Tier2 > Tier3 (novel task, E0b)
  - 도메인 특화 능력: Tier3 > Tier2 > Tier1 (AMD staging, E0a)
- **KG 보상:** Z2_KG_COT가 Tier1b에 임상 규칙을 주입 → 전문화 없이 forgetting 보상(E0c/H8).

**아키텍처 메모:**
- RetinaVLM = **MiniGPT-4 계열** (vision encoder → **Perceiver connector(projection)** → LLaMA-3-8B). LLaVA 계열 아님.
- LLaVA-v1.6-mistral/LLaVA-Med-mistral = LLaVA 계열 (vision encoder → **MLP connector** → Mistral-7B).
- connector/projection은 모달리티 추가 모듈(망각과 무관); LoRA는 가중치 미세조정(망각 발생).
- 공정성: 동일 입력 영상·동일 전처리·동일 파싱 규칙. 모델별 권장 해상도 차이는 명시.

---

## 2. 예측 과제 (VLM은 프롬프트→텍스트→파싱으로 라벨화)

### 2.1 1차: `continue injection` (이진 반응)
- **입력:** 환자(안구)의 대표 B-scan (center-image 검출은 `autoresearch_task2/prepare.py`의 G-channel row-sum 휴리스틱 재사용). pre/post 중 **pre-injection** 영상 + 임상 메타(연령·진단·기저 VA/CST)를 텍스트로.
- **프롬프트(4종 변형으로 robustness 점검):**
  - Z0 (zero-shot, 직접): *"This is a pre-treatment macular OCT B-scan of a patient with {diagnosis}. Based on the retinal fluid and structure, will this patient need continued anti-VEGF injections? Answer 'continue' or 'stop' and explain."*
  - Z1 (바이오마커 유도): 위 + *"First state whether IRF, SRF, PED are present, then decide."*
  - **Z2_KG_COT (KG-guided chain-of-thought, 신규):** 3단계 구조화 프롬프트. KG 규칙을 명시적으로 주입하여 임상 추론 과정을 강제.
    - **Step 1 — Biomarker extraction:** *"Step 1: Extract the retinal biomarkers visible in this OCT image. Return as JSON: {\"IRF\": true/false, \"SRF\": true/false, \"PED\": true/false, \"HRF\": true/false, \"dry_macula\": true/false}."*
    - **Step 2 — KG rule application:** *"Step 2: Apply the following clinical rules to the biomarkers you identified: Rule A: SRF present → recommend 'continue' (confidence 0.85). Rule B: IRF present → recommend 'continue' (confidence 0.80). Rule C: dry macula (no IRF/SRF) → recommend 'stop' (confidence 0.85). Rule D: PED only without fluid → 'case-dependent' (confidence 0.45). State which rules apply and their combined recommendation."*
    - **Step 3 — Decision with rule citation:** *"Step 3: State your final decision ('continue' or 'stop') and cite the specific rule(s) that led to this decision."*
    - **목적:** Z2_KG_COT는 neuro-symbolic KG가 일반 VLM의 forgetting을 보상할 수 있는지 직접 검증(H8). KG 규칙을 프롬프트에 명시 주입 → 임상 논리를 모델이 학습하지 않아도 추론 단계에 강제.
    - **파싱:** JSON 추출 → Step 3 결정 텍스트 파싱. JSON 파싱 실패율(robustness metric)을 별도 집계.
  - F2 (few-shot): 2–4개 라벨된 예시(영상+정답) 제시 후 질의.
- **파싱:** 정규식으로 continue/stop 추출 → 이진 라벨. 파싱 실패 시 "uncertain"으로 분류하고 별도 집계.

### 2.2 2차: 바이오마커 존재 (IRF/SRF/PED/HRF)
- 영상 단위로 VLM에 4개 바이오마커 존재 여부를 질의 → 멀티라벨. 이는 (a) 그 자체 sub-task이자 (b) **XAI 정렬 앵커**.

### 2.3 반응성(magnitude): VA / CST 예측 — **APTOS-2021 공식 채점식 채택**
- VLM에 치료 후 **수치 추정**을 요청(예: *"Estimate the post-treatment central subfield thickness in microns"*, *"Estimate the post-treatment visual acuity (decimal)"*). 정성 추론(악화/호전/유지)은 보조.
- **점수는 APTOS-2021 공식 tolerance 방식**(아래 §3.1)으로 계산 → BlueSky·기존 CNN ref와 직접 비교 가능.
- 솔직성: VLM은 약한 회귀기라 tolerance 점수가 낮을 것으로 예상. 이는 *결함이 아니라 발견*("VLM은 두께를 정밀 회귀하지 못한다")이며 CNN ref(CST-tol 0.606)와의 격차로 서사화. **leaderboard 우승 주장은 하지 않음.**

---

## 3. 데이터 분할 & 평가

- **분할:** 학습 221 안구를 **patient(eye)-level stratified split (85/15)**, `diagnosis`×반응으로 층화 — 선행 Task와 동일 규약(`EXPERIMENT_LIMITATIONS.md` 참조). 누수 방지 위해 같은 안구의 pre/post는 같은 fold.
- **검정력 보강(코드만, 기본 비활성):** 테스트 ~33 안구로 통계 검정력이 제한적이므로, **k-fold 교차검증 함수(`data.make_kfold_splits`)를 코드에 작성하되 호출부는 주석 처리**한다. 기본은 단일 split, 필요 시 주석 해제로 k-fold 활성화(리뷰 대응용).
- **검증셋 342 안구:** 라벨 없음 → 정량 평가 제외, 정성 사례/일반화 시연용으로만.

### 3.1 APTOS-2021 공식 채점식 채택 (선행작업·BlueSky와 직접 비교)
출처: `data_response/experiment/aptos_scoring_formula.md` (arXiv:2505.05768 §4.4) — 이 파일은 **의사코드**다. 구현 현황(검증됨):
- `score_cst` ≈ **이미 구현**: `data_response/experiment/task2_v2/prepare.py:120` `score_cst_tolerance(y_true,y_pred,tol=0.075)` → **참고/적응**.
- `score_va` = **미구현(의사코드만)** → 위 의사코드(0.05 / 7.5% 이중 임계)로 **신규 작성 필요**.
- CST 보정(선택): `olives_validation/train_option_a.py:197–251`의 pw3/isotonic 보정 **참고**(VLM 회귀에는 적용 여부 미정).

| Subtask | 임상 의미 | 공식 채점 | 비교 앵커 (BlueSky / 본 repo CNN) |
|---------|-----------|-----------|-----------------------------------|
| **CI** (continue injection) | **치료 지속 여부 결정** | **AUC** | BlueSky 0.7026(S1)/0.7828(S2); CNN ref CI-AUC 0.68(E2E)~0.81(oracle) |
| **VA** (post 시력) | 반응성 | tolerance: `\|ŷ−y\|≤0.05` (y<1) / `≤7.5%` (y≥1) | BlueSky 0.3216(S1) |
| **CST** (post 두께 μm) | 반응성 | `±7.5% tolerance` | BlueSky 0.5906(S1); CNN ref cal_tol 0.606 |
| IRF/SRF/HRF/PED | 바이오마커 | AUC | BlueSky 0.85~0.93; CNN ref mean AUC 0.9636 |

- **주의:** 공식 final score는 Stage1+Stage2 14개 subtask 평균이지만, 우리는 **라벨 있는 내부 split만** 평가하므로 14-subtask 종합점수가 아니라 **CI-AUC / VA-tol / CST-tol / biomarker-AUC를 개별 보고**한다(이 점 명시).

### 3.2 임상 보조 메트릭 (불균형 대응)
- CI는 클래스 불균형(64:36) → **balanced accuracy, F1, sensitivity/specificity**를 AUC와 병기. 모두 **bootstrap 95% CI**(1,000 resample).
- 바이오마커: per-class AUC/F1 + mean AUC.

### 3.3 층화·통계
- **층화 보고:** DME vs CNVM 별도, 약물(Avastin vs 기타) 편향 caveat.
- **통계 검정:** tier 간 AUC차 DeLong test, tolerance 점수차 bootstrap CI. H1/H2 단조 증가는 Jonckheere–Terpstra trend test.

---

## 4. 통합 XAI 프로토콜 (기존 스택 + VLM-네이티브 기법 + KG 정량 검증)

> 설계 철학: 단일 히트맵을 넘어 **(i) 공간적 근거(saliency)**, **(ii) 트랜스포머 내부 텍스트-이미지 연결(attention)**, **(iii) 언어-논리 정합(neuro-symbolic KG)** 의 핵심 3축 + **(iv) 인과적 의존성(perturbation, 조건부)** 으로 해석가능성을 입체화한다. KG 축은 **Wang 2025(Sensors 25:6879, *Explainable AI Framework for Predicting Treatment Outcomes in AMD*)** 의 neuro-symbolic + LLM 프레임워크(온톨로지 KG + symbolic 규칙 + RAG-LLM 내러티브, >85% 규칙-지지 추론·>90% 바이오마커 인용·AUROC 0.94)를 직접 이식한다. anti-VEGF 투여결정은 규칙기반이라 **DQN 강화학습 모방(KAD)보다 규칙기반 neuro-symbolic 이 적합**하다.

### 4.1 재사용 스택 (코드: `experiments/`, exp1–exp3)
| 기법 | 대상 | 코드 위치 |
|------|------|-----------|
| GradCAM (+GradCAM++/Eigen/LayerCAM) | vision encoder layer4 | nb1, `exp1_outputs/` |
| LLM per-token attention | 키워드("fluid","continue") 조건부 6×6 그리드 | nb2/3 §7.5 |
| VL-saliency | 키워드 logit→영상 역전파 | nb2/3 §7.6 |
| Setting A vs B | 원본 vs XAI-overlay 입력 | nb3 |
| Hallucination detection | 키워드 기반 | nb2/3 |

### 4.2 신규 XAI ①: Cross-Attention Rollout (트랜스포머 내부 뷰 — 수학적 직관)
- **무엇:** VLM의 마지막 레이어 cross-attention을 입력단까지 **rollout**(레이어별 attention 행렬 곱 + residual 보정)하여, 특정 **출력 토큰(예: "continue", "fluid") 생성 순간** 어떤 **이미지 토큰**을 강하게 참조했는지 추적.
- **왜:** post-hoc gradient(saliency)와 달리 **forward-pass 내재 메커니즘** 뷰 → KAD의 DQN-attention과 동일 철학(Clever Hans 완화).
- **임상 서사:** "모델이 *치료 지속*을 말하는 그 순간 SRF(망막하액) 토큰을 얼마나 봤는가" — 단어↔이미지 연결 고리.
- **메트릭:** rollout 질량의 바이오마커-영역 집중도(§4.4 KG 규칙과 교차검증), tier별 비교.
- **구현 현황:** 현 `inner.attention()`은 *평균* text→image attention만 반환(스칼라급) → **layer/head rollout은 신규 작성**. KAD의 DQN cross-attention(`experiments_product/5_kad_*.ipynb`, `exp4_outputs/cross_attention_maps/`)을 *아키텍처 참고*로 활용.

### 4.3 신규 정량 검증 (핵심): Neuro-Symbolic AntiVEGF-Guideline-KG — **Wang 2025 이식**
기존 약점("이미지 단위 라벨뿐, 마스크 없음 → 충실도 측정 간접적")을 **언어-논리 정량 검증**으로 보완. 픽셀 마스크 없이도 평가 가능. **방법론 출처: Wang 2025(Sensors 25:6879) neuro-symbolic + LLM 프레임워크.**
- **KG 형식(Wang 2025식):** directed labeled graph `G=(V,E)`. 노드 `V` = {biomarker(IRF/SRF/PED/HRF), diagnosis(DME/CNVM/PCV), drug, outcome(continue/stop)}; 엣지 `r=(h,t,ρ)` + **confidence weight `w(r)∈[0,1]`**; **온톨로지 매핑 필드(SNOMED-CT/ICD-10)** 포함. 예:
  - `(SRF, clinical_implication, Continue, w=0.85)`
  - `(IRF, clinical_implication, Continue, w=0.80)`
  - `(no fluid / dry macula, clinical_implication, Stop, w=0.85)`
  - `(PED only, weak_implication, case-dependent, w=0.45)`
  - 출처: anti-VEGF 임상 가이드라인 + APTOS 라벨 정의(`APTOS-2021_analysis.md`). 트리플렛 ~수십 개, 공개·재현. `antivegf_guideline_kg.json`.
- **추론 엔진(경량 우선):** **symbolic forward-chaining** 규칙엔진(IF biomarker-set THEN continue/stop) + confidence-weighted fusion `rule_confidence = Π(premise_conf) × rule_weight`. **신경 entity encoder(PubMedBERT InfoNCE, KAD 이식)는 stub+주석으로만** 남겨 향후 full neuro-symbolic 확장점으로 둔다.
- **RAG 주입:** 관련 triple/가이드라인 snippet 을 VLM 프롬프트 컨텍스트로 주입(Wang 2025 RAG) → 환각↓. `ablation_no_context/` 와 연결한 **with/without KG ablation**.
- **(a) Text–KG Alignment:** 각 tier VLM 생성 문장을 파싱(entity·relation)하여 **KG 인과 경로와 일치하는 비율** = Wang 의 ">85% rule-supported reasoning" 에 대응.
- **(b) Attention–KG Consistency:** §4.2 rollout 이 가리키는 영역이 KG 결정-바이오마커와 일치 = Wang 의 ">90% biomarker-citation accuracy" 에 대응.
- **메트릭:** alignment ratio(%), tier별. **예상:** Tier1 ~40% → Tier3 ≥90% 단조 증가(H5).
- **런타임 제약(future):** KG 를 디코딩 가드레일로 사용해 가이드라인 위배 답변 플래그. 본 논문은 *평가용* 우선.

### 4.4 조건부 XAI: Perturbation-based ROCO (인과 — 마스크 가용 시에만)
> **상태: 미구현(조건부).** 유체 영역 마스크가 필요하나 본 데이터셋은 이미지 단위 라벨뿐. **임상의 제공 마스크 입수 시에만 진행**하며, 그 전까지는 인터페이스 stub(`roco_stub.py`)만 유지. region heuristic 기반 근사는 신뢰도가 낮아 본 논문 본문에서 제외.
- **무엇(가용 시):** 마스크된 유체 영역을 **occlusion/Gaussian blur**로 지운 뒤 **CI logit 하락폭** 측정(ROCO = Remove-and-Observe Causal Outcome).
- **메트릭(가용 시):** ΔlogitCI(유체 occlusion) vs Δlogit(무관 영역) — **causal specificity**; tier별(H6).

### 4.5 통합: 스펙트럼-해석가능성 & 결정 연결
- **Spectrum-of-interpretability:** §4.2–4.4 지표(saliency 충실도·rollout 집중·causal specificity·Text-KG alignment)를 tier1→3로 비교 → H2/H5 검정. 성능 추세(H1)와 페어 플롯.
- **Clinical Alignment(핵심 서사):** 전문화될수록 *의사 사고흐름(KG 인과)* 과 *AI 내부 연산(attention·saliency·causal)* 이 정렬됨을 4축 동시 입증.
- **Decision linkage(H3):** 유체 saliency/attention 집중도 → `continue injection` 예측력(단변량 AUC, mediation식 서사).

---

## 5. 실험 매트릭스 (요약)

| 실험 | 산출 | 뒷받침 가설 | 도표 |
|------|------|-------------|------|
| **E0a_forgetting** AMD staging (4 tier) | CI-AUC / ACC, Tier3>Tier2>Tier1 예상 | **H7** (specialist advantage on trained task) | Table 0a, Fig 0a |
| **E0b_forgetting** 일반 VQA (4 tier) | ACC / BLEU, Tier1b≈Tier1a>Tier2>Tier3 예상 | **H7** (forgetting of general ability) | Table 0b, Fig 0b |
| **E0c_KG_compensation** Qwen+Z2_KG_COT vs RetinaVLM (anti-VEGF CI) | CI-AUC 직접 비교 | **H8** | Table 0c |
| E1 치료결정 CI(4 tier × 4 prompt: Z0/Z1/Z2_KG_COT/F2) | **CI-AUC**(공식)+BA/F1, bootstrap CI | H1, H8 | Table 1, Fig 2 |
| E1b 반응성 VA/CST(4 tier) | **VA-tol / CST-tol**(APTOS 공식) vs BlueSky·CNN ref | H1 | Table 1b |
| E2 바이오마커 분류(4 tier) | per-class/mean **AUC**(공식) | (anchor) | Table 2 |
| E3 XAI 충실도(4 tier × saliency 지표 a/b) | 정렬도 막대/추세 | H2 | Fig 3 |
| **E3b Cross-Attention Rollout**(단어↔이미지) | rollout 집중도 tier별 | H2 | Fig 3b |
| E4 결정 연결(saliency/attn → CI) | 단변량 AUC | H3 | Fig 4 |
| E5 hallucination(정상/무소견) | 허위생성률 | H4 | Fig 5 |
| E6 사례 연구(질적) | overlay 패널 | 서사 | Fig 6 |
| **E7 KG-Fidelity**(neuro-symbolic Text–KG / Attn–KG alignment, Wang 2025) | alignment %(tier1~40%→tier3≥90%) | **H5** | Table 3, Fig 7 |
| **E3c Forgetting XAI Probe** | attention entropy(tier별); fluid-energy ratio(fluid eyes) | **H7** (mechanistic) | Fig 3c |
| **E3d Token-Logit Attribution** | VL-saliency on continue token (tier별 attribution map 비교) | **H7** (visual) | Fig 3d |
| ~~E8 Perturbation-ROCO~~ **(조건부/미구현)** | clinician mask 입수 시에만 | (H6) | (Fig 8) |

**E0 실험 상세 — Forgetting Curve:**
- **E0a (AMD staging):** APTOS CNVM 안구(diagnosis=CNVM) + OCT2017 이미지를 사용. 태스크: *"Is there choroidal neovascularization present? Stage as: no AMD / early AMD / neovascular AMD."* RetinaVLM의 훈련 분포에 가장 근접한 태스크. **예상: Tier3 > Tier2 > Tier1** (specialist advantage).
- **E0b (일반 VQA):** OCT 이미지 또는 일반 자연영상 대상 *"What type of image is this? What structures do you see? Describe in detail."* 영상 유형 식별 ACC + 구조 설명 BLEU/ROUGE. **예상: Tier1b ≈ Tier1a > Tier2 > Tier3** (forgetting of general instruction-following).
- **E0c (KG compensation):** anti-VEGF CI 태스크에서 Qwen(Tier1b)+Z2_KG_COT vs RetinaVLM(Tier3)+Z0. 동일 테스트셋. **예상: KG+일반모델 CI-AUC ≥ 전문모델 zero-shot** (H8 검증).
- E0 세 실험이 함께 *전문화-일반화 tradeoff 곡선*과 *KG 보상 효과*를 정의한다.

---

## 6. 자산 맵 — **참고(reference) vs 신규 작성(build)** (코드 탐색 검증 결과, 2026-05-30)

> 솔직성: 기존 코드는 대부분 **drop-in 재사용이 아니라 참고/적응** 대상이다. 특히 (1) 기존 anti-VEGF 작업은 전부 **CNN/앙상블**(ConvNeXt/ResNet101/sklearn)이고 **VLM 추론은 0** → 본 논문의 VLM 파이프라인은 사실상 신규. (2) 단, **KG·cross-attention은 KAD 쪽에 작동 구현이 존재**해 E7/E3b의 from-scratch 부담이 크게 줄어든다.

### 6.1 참고 자산 (검증된 경로)
| 용도 | 경로 | 상태 |
|------|------|------|
| RetinaVLM 로딩/추론 | `SpecialistVLMs/models/retinavlm_wrapper.py`, `load_method*.py`; `inner.query()`/`inner.attention()`/`inner.softmax_logits()` | **참고**(라벨/프롬프트만 교체) |
| GradCAM/VL-saliency(키워드) | `experiments/_test_vl_saliency.py`(L80–189), `_test_exp3.py`(L298–343), nb1–3 | **참고/적응** |
| Setting A/B·overlay·hallucination | `experiments/3_*.ipynb`, `_test_exp3.py`; `exp1_outputs…exp3_outputs/` | **참고** |
| **KG(OCT) + Knowledge Encoder + DQN cross-attn** | **`experiments_product/5_kad_oct_diagnosis_executed.ipynb`**(OCT-KG 24 entities/40 relations, PubMedBERT+InfoNCE, DQN), `experiments/exp4_outputs/`(cross_attention_maps, kg_results), `exp5_outputs/`(checkpoints) | **참고(E7/E3b의 핵심 출발점)** |
| center B-scan 검출 | `autoresearch_task2/prepare.py:159` `detect_center_image_fundus()` + `extract_macular_oct_crop()` | **재사용 가능** |
| CST tolerance 채점 | `task2_v2/prepare.py:120` `score_cst_tolerance()` | **재사용 가능** |
| mean AUC·EarlyStopping·logger | `shared/training_utils.py`(L62–386) | **재사용 가능** |
| pw3/isotonic 보정 | `olives_validation/train_option_a.py:197–251` | **참고**(적용 여부 미정) |
| 데이터 사전 | `anti-vegf-dataset/APTOS-2021_analysis.md` | 참고 |

### 6.2 신규 작성 목록 (build — `paper_xai_antivegf/code/`)
- **`metrics.py`** — `score_va()`(신규, `aptos_scoring_formula.md` 의사코드) + `score_cst_tolerance()`(이식) + subtask 집계 + DeLong/bootstrap/Jonckheere–Terpstra.
- **`data.py`** — center 검출 이식, APTOS CSV(case+pic) 로드, eye-level stratified split. **`make_kfold_splits()` 작성하되 호출부 주석**. E0a용 CNVM 필터 + OCT2017 로더.
- **`models.py`/`prompts.py`/`infer.py`** — **4-tier 로더**(LLaVA-v1.6-mistral-7b / **Qwen3-VL-8B-Instruct** / LLaVA-Med-v1.5-mistral-7b / RetinaVLM), **Z0/Z1/Z2_KG_COT/F2** 프롬프트·파싱, 안구단위 추론. Z2_KG_COT 3단계 파싱(JSON 추출 → Step3 결정) 포함.
- **`forgetting.py`** — E0a(AMD staging) + E0b(일반 VQA) + E0c(KG compensation) 실험 스크립트. forgetting 곡선 시각화(tier × task 히트맵).
- **`kg.py` + `antivegf_guideline_kg.json`** — **Wang 2025 이식** neuro-symbolic 가이드라인 KG. symbolic forward-chaining 규칙엔진 우선; PubMedBERT InfoNCE 인코더는 stub+주석. **Z2_KG_COT 프롬프트 생성 함수**(KG 규칙 → 프롬프트 텍스트 직렬화) 포함.
- **`kg_align.py`** — Text–KG / Attn–KG alignment 채점기(답변 파싱 → KG 경로 일치율).
- **`rollout.py`** — Cross-Attention Rollout. `mini_gpt4.attention()` 이 **전체 attention 텐서를 반환**하므로 layer×head rollout 구현 가능(설계 추정과 달리 평균값만이 아님).
- **`saliency.py`** — `_test_vl_saliency.py` 키워드조건 GradCAM 적응.
- **`roco_stub.py`** — Perturbation-ROCO **미구현 stub**(clinician mask 게이트).
- **`references.bib`** — Wang 2025, McCloskey & Cohen 1989(forgetting), Kirkpatrick 2017(EWC) 등 인용.

---

## 7. 향후 과제 — Projection(connector) vs LoRA (맨 마지막에만 기록)

- **개념 구분(의미적으로 다름, 직교적):**
  - *Projection/connector* (RetinaVLM=Perceiver, LLaVA=MLP, BLIP-2=Q-Former): vision feature를 LLM 토큰 임베딩 공간으로 **번역하는 새 모듈 → 모달리티를 추가**. 없으면 LLM이 영상을 못 읽음.
  - *LoRA*: 기존 가중치 행렬에 low-rank delta 주입 → **행동을 미세조정**(PEFT). 모달리티 추가 아님.
- **본 논문 범위 밖.** 향후: **connector-only vs connector+LoRA vs LoRA-only** 전문화 레버가 anti-VEGF 반응 성능·XAI 충실도에 미치는 영향을 ablation. (설계 메모일 뿐, 본 논문에서 학습하지 않음.)
