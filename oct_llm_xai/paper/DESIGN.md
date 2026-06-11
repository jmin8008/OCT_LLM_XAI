# 논문 설계: 단일 이미지 CDSS의 환상 — anti-VEGF 결정은 instill되지 않는다

> 상태: **결과 확정(v0.3 전수 재실행 완료, 2026-06-06) → 서사 재정렬 단계.**
> 본 폴더(`oct_llm_xai/paper/`)의 3개 .md(DESIGN/EXPERIMENTAL_PROTOCOL/PAPER_OUTLINE)는
> 논문 설계·서술 골격이며, 실제 실험·매트릭스는 `paper_xai_antivegf/`의 `code/` +
> `sft_data/matrix.{md,json}` + `MATRIX_ANALYSIS_v0.3.md`에 있다.
>
> **⚠️ 서사 전환(2026-06-10):** 본 문서의 이전 버전은 "4-tier 전문화 스펙트럼 → 성능·XAI
> 단조 상승 + catastrophic forgetting + KG 보상(H1~H8)"라는 *zero/few-shot 프롬프트(Part A)*
> 서사였다. 그러나 실제로 완주된 실험은 **LoRA SFT 4-arm × meta × 3-backbone instill
> 매트릭스(Part B)**이고, 그 결과는 *전혀 다른 논문*을 가리킨다. 본 문서는 **Part B의
> negative/mechanistic 결과를 패러다임 비판으로 격상한 새 spine**으로 재작성됐다.
>
> **경로 규약:** 문서는 `oct_llm_xai/paper/`에 모여 있고, **코드/데이터/결과 경로
> (`code/`, `sft_data/`, `fluid_masks_v2/`)는 모두 `oct_llm_xai/paper_xai_antivegf/` 기준
> 상대경로**다(문서만 paper/로 분리).

---

## 1. 한 줄 요약 (Thesis)

> **단일 pre-treatment OCT 영상 한 장으로 anti-VEGF "continue/stop" 치료결정을 모델에
> instill할 수 있는가?** 우리는 강한 training-time 개입 — 지도미세조정(SFT), counterfactual
> 증강, 그리고 attention을 유체 병변에 강제 정렬하는 attention-KL — 을 3개의 이질적
> VLM 백본에 전수 적용했고, **답이 "불가능"임을 일관되게 보인다.** 보이는 개념(유체
> 바이오마커)은 백본 무관하게 instill되지만(perceptual), continue/stop 결정은 모든 arm·
> 백본에서 상수로 collapse하고, 예후·종합반응도 단일 pre-image의 *정보론적 천장* 탓에 대부분
> majority 이하에 머문다(약한 부분 신호는 §6 (B) 참조). 막힌 신호의 일부는 pre-treatment 임상
> 메타데이터에 존재하나 — **정형 chain-of-thought SFT가 베이스 모델의 그 메타 판별력마저
> 덮어버린다(SFT의 역설).**

핵심 재프레이밍: 이것은 "우리 모델이 잘 안 됐다"가 아니라, **"단일 cross-sectional 영상으로
치료결정·예후를 예측하려는 기존 medical-AI 패러다임"과 "SFT/instruction-tuning이면 임상
추론을 instill할 수 있다는 만능주의"라는 두 가정의 한계를, 촘촘한 ablation(arm A~D + meta)으로
최초 실증**한 연구다. grounding(보는 곳)과 decision(판단)의 **decoupling**이 training-time
개입으로도 풀리지 않음을 인과적으로 보인다.

---

## 2. 동기 (Motivation)

- **임상 니즈:** Anti-VEGF 주사(Avastin/Eylea 등)는 nAMD·DME·PCV의 1차 치료지만 "주사
  지속/중단" 결정은 OCT 유체 소견(IRF/SRF/PED)과 시력·두께 변화에 의존한다. 자동 보조의
  가치는 크다.
- **패러다임 가정 ①(단일 이미지 환상):** 기존 anti-VEGF 예측 연구는 사실상 전부 단일
  cross-sectional 영상 기반 CNN/앙상블이다(VLM 추론 전무). 그러나 "continue/stop"은
  *치료 후 미래 결과의 함수*다 — 단일 pre-image에는 그 답을 결정할 정보가 원리적으로
  없다. 이 가정은 거의 검증된 적이 없다.
- **패러다임 가정 ②(SFT 만능주의):** 의료 VLM 연구는 "도메인 데이터로 SFT/instruction-tuning
  하면 임상 추론이 모델에 들어간다"고 암묵 전제한다. 우리는 *무엇이 instill되고 무엇이
  안 되는지, 그리고 SFT가 베이스 능력을 오히려 파괴할 수 있는지*를 분해한다.
- **해석가능성 가정 ③(attention=explanation):** XAI 관행은 "모델이 본 영역=판단 근거"로
  읽는다. 우리는 attention을 임상 병변에 *강제 정렬(attention-KL)* 했을 때도 결정이 불변임을
  보여, attention 정렬이 grounding을 고치지 못함을 인과적으로 입증한다.
- **기회:** APTOS-2021은 이미지 단위 바이오마커 라벨 + pre/post 페어 + continue/stop +
  ΔCST/ΔVA를 함께 제공해, "무엇이 학습 가능한 신호이고 무엇이 정보론적 천장인가"를 분해할
  드문 자원이다.

---

## 3. 데이터셋 (Dataset)

**원천:** `data_response/anti-vegf-dataset/APTOS-2021/Final Datasets/` (사전: `APTOS-2021_analysis.md`)
**가공물(학습 직접 입력):** `sft_data/sft_kg_cot.json`, 유체 마스크 `fluid_masks_v2/`.

| 항목 | 내용 |
|------|------|
| 모달리티 | OCT B-scan, pre/post 페어, 안구(eye) 단위 |
| 사용 단위 | 유체 마스크가 작업된 **218 eye** (train 183 / test 35, eye-level split) |
| 1차 라벨 | **`continue injection`** (continue/stop). test 분포 continue 21 / stop 14 (prior 0.60) |
| 종합반응(2차) | **composite responder** good/poor (`ΔCST≤−25` OR `ΔVA≥0.1`). test good 24 / poor 10 (prior 0.71) |
| 예후(2차) | ΔCST 4-bucket(marked/partial/minimal/worsening). test majority 0.314 |
| 이미지 바이오마커 | IRF/SRF/PED (12×12 마스크, Claude-vision subagent, GT mismatch 0) |
| meta arm 주입 5필드 | **age, gender, drug, preVA, preCST** (미래 함수 아님 → 입력 허용). 진단명은 기록돼 있으나 Step2를 진짜 추론으로 만들기 위해 프롬프트에서 **고의 제외** |
| post/Δ outcome | post_cst/va, ΔCST/ΔVA — **라벨이므로 입력 금지(누설 방지)** |

**핵심 한계(논문에 명시):**
- **test n=35(소표본)** → 셀 간 소수점 차이 대부분 통계적 유의성 없음. 신뢰할 것은
  **3개 백본에 반복되는 질적 패턴**.
- 바이오마커는 이미지 단위 라벨(세그멘테이션 마스크 아님) → Attn-KG·perturbation 인과
  XAI는 조건부/미가동.
- 시점 pre/post 2개뿐 → longitudinal 모델링 불가(이것이 본 논문의 핵심 한계이자 결론).
- 약물·진단 편향(Avastin/DME 우세).

---

## 4. 선행 작업과의 차별성 (Positioning)

| 선행 라인 | 무엇 | 본 논문과의 관계 |
|-----------|------|------------------|
| anti-VEGF/OCT 반응 예측 (BlueSky APTOS-2021, CNN/앙상블) | 단일 이미지 기반 분류·회귀 | **본 논문이 반증하는 패러다임의 대표** — 전부 단일 cross-sectional, VLM 추론 전무 |
| **Attention is not Explanation** (Jain&Wallace 2019; Wiegreffe&Pinter 2019 반론) | attention 가중치≠근거 | 본 논문은 attention을 **강제 정렬(attention-KL)** 하고도 결정이 불변임을 보여 *training-time*으로 명제를 확장 |
| **Shortcut learning** (Geirhos 2020; DeGrave 2021 COVID X-ray) | 모델이 인과 아닌 상관/숏컷 학습 | 결정 collapse를 "라벨 prior 숏컷"으로 해석하는 이론적 배경 |
| **SFT/medical-FT forgetting** (McCloskey&Cohen 1989; Kirkpatrick 2017; LLaVA-Med 평가) | 미세조정이 기존 능력 소실 유발 | (E) SFT 역설 — 정형 CoT SFT가 zero-shot 메타 판별력을 net-negative로 덮음의 이론적 닻 |
| **Neuro-symbolic + LLM** (Wang 2025, Sensors 25:6879) | 온톨로지 KG + symbolic 규칙 + RAG-LLM | 본 논문 KG의 *방법론 출처*. 단 본 연구에서 KG는 **성능 보상기가 아니라 답변 자기정합성(self-consistency) 채점기**로 격하 |
| KAD (Zhang 2023, UMLS KG + DQN) | KG 데이터구조/인코더 | 코드 스캐폴딩 참고만 (DQN 미채택) |

→ **본 논문의 빈자리(niche):** (a) anti-VEGF 결정 instill을 **3개 이질적 VLM 백본**(RetinaVLM
OCT특화 / LLaVA-Med 의료 / Qwen3.6 일반)에 전수 적용해 **결과의 백본 불변성**을 입증 —
즉 이것은 모델 결함이 아니라 *과제의 성질*임을 보임. (b) instill 가능/불가능을 **arm A~D +
meta ablation으로 분해**(perceptual은 가능, decision/prognosis는 정보론적 천장). (c) attention-KL로
"attention is not explanation"을 *training-time*에서 인과 검증. (d) **SFT의 역설**(구조 instill ≠
판별능력 instill, net-negative 가능)을 메타 조건화의 역전으로 최초 실증.

> **백본 명명:** 본 연구는 "tier(전문화 등급)" 대신 **백본 다양성 축**으로 읽는다.
> tier3=RetinaVLM, tier2=LLaVA-Med, tier1c=Qwen3.6-27B는 코드/매트릭스 호환을 위해 라벨만
> 유지하되, 서사상 의미는 "스펙트럼 위치"가 아니라 "독립적 3 백본에서의 재현".

---

## 5. 기여 (Contributions)

1. **C1 — Grounding⊥Decision decoupling이 training-time 개입으로도 풀리지 않음(핵심).**
   [findings (B)+(C)] SFT(arm B)·counterfactual(arm D)·attention-KL(arm C) 어느 것으로도
   anti-VEGF continue/stop 결정 grounding이 학습되지 않고 상수로 collapse(balAcc≈chance)함을
   **3개 백본에서 일관**되게 입증. 특히 attention을 유체에 강제 정렬(KL 절반↓)해도 결정 불변
   → *training-time* "attention is not explanation".
2. **C2 — 단일 pre-treatment 이미지의 정보론적 천장.** [findings (A) vs (B)] 보이는 개념(유체
   바이오마커)은 백본 무관하게 instill되나(0.32→0.81 등), 결정·예후·종합반응은 *치료 후 미래
   결과의 함수*이므로 단일 pre-image로 ≤majority. "보이는 것은 instill 가능, 미래를 결정하는
   것은 불가"의 경계를 실증.
3. **C3 — SFT의 역설(가장 novel).** [findings (D)+(E)] 막힌 신호의 일부는 pre-treatment
   메타데이터에 있고 zero-shot 모델은 이를 융합(tier2 A_meta rGPbal 0.758)하지만, **정형 CoT
   SFT를 거치면 모델이 시각패턴 낭독기로 변해 그 메타 판별력을 상실(B_meta 0.500)** 한다.
   "구조 instill ≠ 결정능력 instill, SFT가 net-negative일 수 있음"을 메타 조건화 역전으로 입증.

(부차) KG는 답변 자기정합성(Text–KG self-consistency) 채점기로 사용; connector vs LoRA,
multi-modal/longitudinal CDSS는 향후 과제.

---

## 6. 핵심 발견 (Findings) — matrix 라벨 (A)–(E)

> 이전 H1~H8(전문화 스펙트럼/forgetting/KG 보상)은 **폐기**. 발견 라벨은 `sft_data/matrix.md`·
> `MATRIX_ANALYSIS_v0.3.md`의 (A)–(E)와 **1:1 일치**(전 문서 공용).

- **(A) 보이는 biomarker는 SFT로 instill ✓ (백본무관) — 유일한 명백 positive.**
  tier3 0.317→0.810, tier2 (파싱불가)→0.752, tier1c 0.800→0.829; arm C도 동급(0.829).
- **(B) continue/stop 결정 collapse ✓ (백본무관) — 중심 negative.** 결정 grounding은
  instill(B)·counterfactual(D)·attention-KL(C) 어느 것으로도 학습되지 않고 상수로 붕괴
  (B/C→continue cont 0.94–1.0, D→stop; balAcc 0.42–0.50=chance, 예외 없음). **예후(ΔCST
  4-class)·종합반응도 단일 pre-image로 대부분 ≤majority**(예후 ≈0.257<0.314)=정보론적 천장.
  (부분 예외: tier3 B responder balAcc 0.613 — 약하고 n.s., 단일백본.)
- **(C) attention≠explanation ✓.** attention-KL이 학습 KL 2.09→0.9로 attention을 유체에
  끌어와도(FER↑) 결정/예후/responder 전부 불변 → 정렬해도 grounding 안 고쳐짐(training-time).
- **(D) 메타 신호 일부 존재 부분 ✓ (백본의존).** 막힌 신호 일부는 pre-treatment 메타에. tier2
  A_meta balAcc 0.619 / rGPbal 0.758 / resp3 0.824로 image-only(chance)에서 도약. 단 tier3·
  tier1c는 미동 → *지시따르기 능력*에 의존.
- **(E) ★ SFT의 역설 ✓ (tier2 한정) — 가장 novel.** 정형 CoT SFT가 그 메타 판별력을 덮음.
  tier2 A_meta rGPbal 0.758 → B_meta 0.500 역전(cont 1.0). net-negative SFT의 증거.

> **arm D 설계 의도 (반증 프레이밍).** arm D는 유체 픽셀을 counterfactual로 제거(occlusion)해
> **"유체(단일 영상) 소견만으로 continue/stop을 판단할 수 있다"는 암묵 가정을 직접 반증**하기
> 위한 실험이다 — 유체를 지웠을 때 결정이 인과적으로 따라 변하는지를 본다. 결과는 결정이
> 유체 제거를 따라가지 않고 라벨 prior로 collapse → **유체만으로는 결정이 grounded되지 않음**을
> 보인다. (단 occlusion이 저해상 grayscale 백본에 *지각 불가*인 confound가 겹쳐, 인과 결론은
> perceptibility 재검 뒤로 한정 — §8.) ⚠️ 실제 dry-eye의 임상 결정 분포는 arm D의 *반증 논리와
> 별개*이며 arm D 라벨의 정당성 근거로 쓰지 않는다(기껏해야 (B)/(C2)의 부수적 방증).

---

## 7. 산출물 (Deliverables)

| 파일 | 내용 |
|------|------|
| `DESIGN.md` (본 파일) | 동기·논지·기여·포지셔닝·데이터·가설 |
| `EXPERIMENTAL_PROTOCOL.md` | 백본·instill arm(A/B/C/D+meta)·분할·메트릭·XAI/KG·실험매트릭스 |
| `PAPER_OUTLINE.md` | 섹션 골격 + claim↔도표 매핑 |
| `sft_data/matrix.{md,json}`, `MATRIX_ANALYSIS_v0.3.md` | 전수 결과 원자료 + 분석 전문 |
| `code/harness.py` 외 | 실제 train/eval 하네스, attn-KL, SFT 생성기 |

---

## 8. 위험 요소 (Risks)

- **Negative-result framing 수용성:** 핵심 결과가 negative이므로, "모델 결함" 인상을 피하고
  *패러다임 한계의 인과 실증*으로 당당히 프레이밍해야 함(§1·§5). target venue는 position/
  findings를 받는 곳(예: findings track, clinically-oriented workshop) 우선 검토.
- **검정력(eye-level):** 결정 N이 eye 수 제한(현 test 35 → 재실행 test≈70, 최소포함조건).
  셀 차이 대부분 n.s. → 핵심 셀 bootstrap CI/permutation으로 명시; 질적 패턴 *3백본 반복*을
  신뢰 근거로. 특히 **dry eye는 전체 7개뿐**이라 quota(≥2)로 붕괴만 막을 뿐 근본적으로 약함.
- **(E) 메타 역전의 단일백본 한정성:** A_meta>B_meta 역전은 tier2에서만 관측(A_meta가 애초
  높았던 유일 백본). "메타가 zero-shot 우위를 준 곳에서 SFT가 지웠다"로 *한정 서술*.
- **occlusion 지각불가 confound:** arm D all-stop collapse는 occlusion이 192×192 grayscale
  백본에 지각 불가(이슈#1 확정)인 confound이지 grounding 결과 아님. tier1c D
  cfFlip1.0/faithGap+0.118만 예외 → Qwen perceptibility_check 재실행으로 검증 필요(과대해석 금지).
- **KG 트리플렛 수작업(주관성):** Wang 2025 온톨로지 방식 차용·공개·소규모로 완화. 단 KG는
  self-consistency 채점 용도로 격하했으므로 비중 낮음.
