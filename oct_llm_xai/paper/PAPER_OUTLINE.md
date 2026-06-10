# 논문 골격 (Paper Outline)

> 각 섹션 = [뒷받침할 주장] + [채울 도표/실험]. 가제:
> **"The Single-Image Illusion: Why Supervised Fine-Tuning Cannot Instill
> Anti-VEGF Treatment Decisions in Medical Vision-Language Models"**
> 부제 후보: *Grounding⊥Decision decoupling, the informational ceiling of a single
> pre-treatment scan, and the SFT paradox.*
>
> (구 가제 보존: *"The Cost of Specialization: Catastrophic Forgetting … KG
> Compensation"* — forgetting/KG-보상 포지셔닝, 폐기.)

대상 형식: 의료 AI findings/position 수용 venue(예: MICCAI/MIDL findings, clinical-AI
workshop) 우선. 8–9p 본문 + 부록. **포지셔닝: 우리 모델 결함이 아니라 *패러다임 한계의 인과
실증*.**

> **경로 규약:** 문서는 `oct_llm_xai/paper/`. 코드/데이터 경로는 `paper_xai_antivegf/` 기준.
> 발견 라벨 (A)~(E)는 DESIGN §6·`sft_data/matrix.md`와 1:1.

---

## Abstract (3단계 서사)
- **문제(단일 이미지 환상):** anti-VEGF "continue/stop" 결정은 임상적으로 단일 OCT 유체
  소견에 의존한다고 여겨지나, 결정은 본질적으로 *치료 후 미래 결과의 함수*다. 기존 연구는
  단일 cross-sectional 영상으로 이를 예측하려 했고(전부 CNN), "SFT면 임상 추론을 instill
  할 수 있다"는 가정도 검증된 적 없다.
- **방법:** anti-VEGF 결정 instill을 **3개 이질적 VLM 백본**(RetinaVLM OCT특화 / LLaVA-Med
  의료 / Qwen3.6 일반)에 **4-arm × meta ablation**으로 전수 적용 — A(zero-shot)·B(SFT)·
  C(attention-KL)·D(counterfactual) + 메타 프롬프트. 채점: decision·biomarker·prognosis·
  composite responder.
- **결과:** (A) 보이는 biomarker는 백본 무관 instill(0.32→0.81); (B) **결정은 모든 arm·백본
  에서 상수 collapse**(balAcc≈chance); (C) attention을 유체에 강제 정렬해도 결정 불변
  (training-time "attention≠explanation"); (D) 막힌 신호 일부는 pre-treatment 메타에 존재
  (tier2 A_meta rGPbal 0.758); (E) **정형 CoT SFT가 그 메타 판별력을 덮음**(B_meta 0.500
  역전, SFT의 역설).
- **의의:** 단일 이미지 기반 CDSS와 SFT 만능주의의 한계를 촘촘한 ablation으로 최초 실증.
  신뢰가능 CDSS는 multi-modal/longitudinal 임상정보 통합으로 가야 하며, SFT는 지각 하위
  태스크에 국한해야 함을 제언. **⚠️ test n=35 소표본 — 신뢰 근거는 3백본 반복 질적 패턴.**

## 1. Introduction
- **주장:** anti-VEGF 결정 보조는 중요하나, 단일 영상 + SFT로 결정을 instill하려는 지배적
  접근은 정보론적·기제적으로 막혀 있다.
- 임상 배경(결정은 유체 소견 의존이라는 통념) → 두 패러다임 가정(① 단일 이미지 환상,
  ② SFT 만능주의) → 본 논문의 질문(무엇이 instill되고 무엇이 막히는가; SFT가 베이스 능력을
  파괴하는가).
- 기여 3개 bullet (DESIGN §5: C1 decoupling, C2 정보론적 천장, C3 SFT 역설).

## 2. Related Work
- (a) anti-VEGF/OCT 반응 예측(BlueSky APTOS-2021, CNN/앙상블) — **본 논문이 반증하는 단일
  이미지 패러다임의 대표**.
- (b) 의료 VLM(LLaVA-Med, RetinaVLM, Med-Flamingo) + SFT/instruction-tuning 관행.
- (c) **Attention is not Explanation**(Jain&Wallace 2019; Wiegreffe&Pinter 2019 반론) —
  본 논문은 attention-KL 강제 정렬로 *training-time*에서 명제 확장.
- (d) **Shortcut learning**(Geirhos 2020; DeGrave 2021 COVID X-ray) — 결정 collapse=라벨
  prior 숏컷 해석의 배경.
- (e) **Catastrophic forgetting / SFT 부작용**(McCloskey&Cohen 1989; Kirkpatrick 2017 EWC;
  LLaVA-Med 평가) — SFT 역설(C3)의 이론적 닻.
- (f) Neuro-symbolic + LLM 의료 AI(Wang 2025) — 본 KG의 방법론 출처(단 self-consistency
  채점으로 격하). KAD(Zhang 2023)는 스캐폴딩 참고.
- **주장:** 결정 instill을 3백본 × arm ablation으로 분해해 "무엇이 막히는지"와 "SFT가
  메타 판별력을 덮는지"를 동시에 인과 규명한 연구는 없다.

## 3. Dataset
- APTOS-2021 OCT, pre/post, `continue injection`/composite responder/ΔCST, 이미지 단위
  바이오마커(12×12 마스크). 218 eye(train 183 / test 35, eye-level).
- 분포·편향(DME/Avastin)·한계(2시점→longitudinal 불가, 마스크=이미지 단위, n=35).
- **도표:** Table(코호트 통계 + test 라벨 분포), Fig 1(데이터/4-step CoT 태스크 개요).

## 4. Methods
- 4.1 **백본 다양성 축(3 backbone):** RetinaVLM/LLaVA-Med/Qwen3.6 — "스펙트럼"이 아니라
  *결과 백본 불변성* 입증용. 아키텍처·QLoRA·arm C N/A(tier1c linear-attn) 명시.
- 4.2 **Instill arm(A/B/C/D + meta):** A=zero-shot, B=factual SFT(LM loss), C=B+attention-KL
  (rollout→6×6 pool→forward KL, λ=0.5), D=factual+counterfactual(occluded, 라벨 뒤집기),
  meta=프롬프트에 age/gender/drug/preVA/preCST 주입(post/Δ 금지). `harness.py` 배선 그대로.
- 4.3 **4-step CoT 데이터:** Visual→Pathophysiology→Decision→Response(인과 순서, 진단명
  미제공). divergent 77 eye 정직 렌더. 생성기 `gen_sft_kg_cot.py`.
- 4.4 **채점·통계:** 4 과제 자동 파싱, balanced accuracy, majority baseline 병기, bootstrap
  CI/permutation으로 핵심 셀 n.s. 명시.
- 4.5 **XAI/KG(보조):** Text–KG self-consistency 채점(Wang 2025 규칙엔진); attention rollout은
  arm C 정렬 증거(FER↑)로만; Attn–KG/perturbation은 마스크 부재로 조건부.
- **도표:** Fig(파이프라인 — 4-step CoT + arm A/B/C/D + meta).

## 5. Experiments & Results
- **E-A instill 매트릭스(3 백본 × arm) → Table 1(matrix.md 치환). 발견 라벨 (A)~(E)는
  DESIGN §6·matrix.md와 1:1:**
  - **(A) biomarker instill** → Fig 2a: A→B 상승 tier3 0.317→0.810 / tier2 →0.752 /
    tier1c 0.800→0.829 (백본무관).
  - **(B) decision collapse + 정보론적 천장** → Fig 2b: B/C cont 0.94–1.0, D cont 0.0,
    balAcc 0.42–0.50 (백본무관); 예후 ≈0.257 ≤ majority 0.314 (Table 1).
  - **(C) attention≠explanation** → Fig 3: arm C 학습 KL 2.09→0.9·FER↑ vs 결정/예후/
    responder 불변.
  - **(D) meta signal** → Fig 4a: tier2 A_meta balAcc 0.619/rGPbal 0.758/resp3 0.824 vs
    image-only chance (백본의존).
  - **(E) ★SFT 역설** → Fig 4b(핵심): tier2 A_meta 0.758 → B_meta 0.500 역전.
- **E-B occlusion perceptibility → Fig 5:** RetinaVLM 지각불가(arm D all-stop=confound 확정);
  Qwen 재실행 결과(tier1c D cfFlip1.0/faithGap+0.118 검증).
- **E-C meta-ceiling 정렬 → Table 2:** tier2 A_meta 0.758 vs Part A logreg responder 천장
  (천장 이하면 정당 신호).
- **E-D 역전 case-level → Fig 6(핵심 분석):** tier2 zero-shot 적중 vs SFT 오답 eye diff +
  preCST-only ablation → "SFT가 무슨 신호를 덮나".
- 통계: bootstrap CI / permutation, n.s. 명시. 3백본 반복을 신뢰 근거로.

## 6. Discussion
- **핵심 서사 — 단일 이미지 환상의 해부(C1·C2):** 결정·예후·반응은 미래 결과의 함수 →
  단일 pre-image로는 *어떤* training-time 개입(SFT/CF/attention-KL)으로도 instill 불가.
  3백본 반복 = 과제의 성질이지 모델 결함이 아님.
- **Training-time attention≠explanation(C1):** attention을 병변에 강제 정렬해도 결정 불변
  → "본다"와 "판단한다"의 decoupling. 정렬형 XAI/instill의 한계.
- **SFT의 역설(C3, 가장 novel):** 정형 CoT SFT가 모델을 시각패턴 낭독기로 만들어 zero-shot이
  쓰던 메타 조건화를 파괴. "구조 instill ≠ 결정능력 instill, net-negative 가능."
- **임상 함의 / 제언:** (i) SFT는 지각 하위태스크(병변 검출)에 국한, 최종 의사결정에 맹목
  적용 시 임상 맹점; (ii) 신뢰가능 CDSS는 **multi-modal/longitudinal**(기저질환·과거력·
  복약·시계열) 통합 구조; (iii) attention 정렬은 신뢰의 충분조건 아님.

## 7. Limitations
- **test n=35(검정력):** 셀 차이 대부분 n.s. → bootstrap CI 명시, 3백본 질적 반복으로 방어.
- **(E) 메타 역전 단일백본 한정:** tier2에서만(A_meta가 높았던 유일 백본) — 한정 서술.
- **occlusion 지각불가 confound:** arm D all-stop은 grounding 아님(이슈#1); tier1c D 예외는
  Qwen perceptibility 재검 필요.
- 이미지 단위 라벨(마스크 아님) → Attn-KG/perturbation 조건부; 2시점·약물/진단 편향.
- **프레이밍:** 위 한계는 *기존 단일 이미지 패러다임의 구조적 한계*로 서술(우리 모델 결함 아님).

## 8. Conclusion
- 단일 pre-treatment 영상으로 anti-VEGF 결정을 instill하려는 시도는 — SFT·counterfactual·
  attention-KL 어느 것으로도 — 정보론적·기제적으로 막혀 있고(3백본 확증), 정형 CoT SFT는
  오히려 베이스의 메타 판별력을 덮는다. 신뢰가능 CDSS는 단일 이미지를 넘어선 다중·시계열
  임상정보 통합으로 가야 한다.

## 9. Future Work
- multi-modal/longitudinal CDSS 구조; connector(projection) vs LoRA 전문화 레버 ablation;
  KG 디코딩 가드레일(self-consistency 채점 너머).

---

## 도표 인덱스 (claim ↔ figure/table)
| ID | 산출 | 뒷받침 | 출처 실험 |
|----|------|--------|-----------|
| Fig 1 | 데이터/4-step CoT 태스크 개요 | 문제정의 | §3 |
| **Table 1** | **instill 매트릭스(3 백본 × arm × 12 메트릭)** | (A)~(E) / C1·C2·C3 | E-A (matrix.md) |
| Fig 2a | biomarker instill A→B (3백본) | (A) | E-A1 |
| Fig 2b | decision collapse (cont·balAcc, 3백본) | (B) | E-A2 |
| Fig 3 | attention-KL: KL↓·FER↑ vs 결정 불변 | (C) | E-A3 |
| Fig 4a | meta signal (tier2 A_meta 도약) | (D) | E-A5 |
| **Fig 4b** | **★SFT 역설 A_meta 0.758→B_meta 0.500** | (E) | E-A6 |
| Fig 5 | occlusion perceptibility (confound) | arm D 판정 (cf. (B)) | E-B |
| Table 2 | meta-ceiling 정렬(A_meta vs logreg 천장) | (D) 정당성 | E-C |
| **Fig 6** | **역전 case-level diff + preCST ablation** | (E) / C3 핵심 | E-D |
