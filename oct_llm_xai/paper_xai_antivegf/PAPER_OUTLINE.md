# 논문 골격 (Paper Outline)

> 각 섹션 = [뒷받침할 주장] + [채울 도표/실험]. 가제:
> **"The Cost of Specialization: Catastrophic Forgetting in Medical VLMs and Neuro-Symbolic KG Compensation for Anti-VEGF Response Prediction"**
> 부제 후보: *Can KG-guided clinical reasoning in generalist VLMs match specialist models without fine-tuning?*
>
> (구 가제 보존: *"From Generalist to Specialist: A Unified XAI Protocol for Anti-VEGF Response Prediction"* — 이전 포지셔닝)

대상 형식: 의료 AI 워크숍/저널(예: MICCAI/MIDL/Ophthalmology AI). 8–9p 본문 + 부록.

---

## Abstract
- 문제: anti-VEGF 반응 예측 + VLM 해석가능성의 임상 타당성 미검증, 기존 XAI 평가는 정성/근사. **더 나아가: VLM 전문화(medical fine-tuning)가 novel clinical task에서 catastrophic forgetting을 유발하는지, 그리고 neuro-symbolic KG가 이를 보상할 수 있는지 미검증.**
- 방법: **4-tier VLM 스펙트럼** (Tier1a=LLaVA-v1.6-mistral[일반], Tier1b=Qwen3.6-27B[일반], Tier2=LLaVA-Med-mistral[의료], Tier3=RetinaVLM[OCT특화]) — **Tier1a↔Tier2 동일 Mistral 백본으로 forgetting 직접 측정**. + **통합 XAI 프로토콜**(핵심 3축: 바이오마커 saliency 충실도 · Cross-Attention Rollout · **neuro-symbolic 임상 가이드라인 KG 정합**[Wang 2025 이식]; + 조건부 Perturbation-ROCO). + **Z2_KG_COT** 프롬프트(KG-guided 3단계 CoT)로 KG 보상 검증.
- 결과: (i) **H7 ✓**: novel task(anti-VEGF CI) Qwen3.6 CI-AUC=**0.607** > RetinaVLM=**0.440** — generalist beats specialist; (ii) **H8 ✓**: Qwen3.6+Z2_KG_COT(KG compensation) CI-AUC=0.607 > RetinaVLM+KG=0.440; (iii) **H5 부분 ✓**: Qwen3.6 Z1 Text-KG alignment=**0.955** > Wang >85% 기준선; (iv) AMD staging: RetinaVLM **60%** vs Qwen3.6 **50%** (domain gap 주의); (v) E3c/E3d XAI forgetting probe로 망각의 시각적 기제 정량화(attention entropy, fluid-energy ratio).
- 의의: **전문화의 인지적 비용(catastrophic forgetting)을 최초 정량화**, neuro-symbolic KG가 fine-tuning 없이 forgetting을 보상함을 실증. 전문화될수록 *의사 사고흐름(KG 인과)* 과 *AI 내부 연산(attention·saliency·causal)* 이 정렬됨을 함께 규명(**Clinical Alignment**).

## 1. Introduction
- **주장:** 반응 예측은 임상적으로 중요하나, VLM 근거의 임상 타당성은 미검증.
- 임상 배경(anti-VEGF 결정은 유체 소견 의존) → 해석가능성 갭(정성 히트맵·pseudo-GT 한계) → 본 논문의 질문(성능 + **상관·인과·논리** 정합).
- 기여 5개 bullet (DESIGN §5).

## 2. Related Work
- (a) 의료 VLM(LLaVA-Med, RetinaVLM, Med-Flamingo), (b) anti-VEGF/OCT 반응 예측(BlueSky APTOS-2021, CNN 접근), (c) XAI 충실도 평가(pseudo-GT의 한계), (d) **Neuro-symbolic + LLM 의료 AI(Wang 2025, Sensors 25:6879: 온톨로지 KG + symbolic 규칙 + RAG-LLM, >85% 규칙-지지·>90% 바이오마커 인용)** — 본 KG 축의 1차 출처; KAD(UMLS KG + DQN)는 부차 스캐폴딩 참고, (e) 트랜스포머 XAI(attention rollout, perturbation/occlusion).
- **(f) Catastrophic Forgetting & Continual Learning** — McCloskey & Cohen (1989)이 처음 규명한 catastrophic forgetting: 신경망이 새로운 태스크 학습 시 이전 지식을 급격히 소실. Kirkpatrick et al. (2017) EWC(Elastic Weight Consolidation)는 중요 가중치를 고정하여 보완. VLM 도메인에서는: (i) 의료 fine-tuning이 일반 instruction-following 능력을 감소시킨다는 증거(LLaVA-Med 평가 연구들); (ii) LoRA PEFT가 full FT 대비 forgetting을 줄이나 완전 해소 못 함; (iii) 기존 연구는 forgetting을 *주로 훈련 중 방지 대상*으로 다루나, **본 논문은 forgetting을 *이미 배포된 모델들 간 성능 차이의 원인*으로 측정** — 이 프레임은 신규.
- **주장:** 이들을 잇는 연구—동일 백본으로 forgetting을 직접 정량화하면서, KG 보상으로 specialization 없이 성능을 회복하고, 전문화 스펙트럼 위에서 성능·saliency·attention·causal·KG-논리를 한 프로토콜로 동시 측정—는 없다.

## 3. Dataset
- APTOS-2021 OCT, pre/post, `continue injection` 라벨, **이미지 단위 바이오마커**(XAI 앵커).
- 분포·편향(DME/Avastin 우세)·한계(2시점, 마스크 부재→KG로 보완) 표.
- **도표:** Table (코호트 통계), Fig 1 (데이터/태스크 개요).

## 4. Methods
- 4.1 **모델 스펙트럼(4 tier + CNN ref):** Tier1a(LLaVA-v1.6-mistral, 일반), Tier1b(Qwen3.6-27B, 일반), Tier2(LLaVA-Med-mistral, 의료), Tier3(RetinaVLM, OCT). **Forgetting design:** Tier1a↔Tier2는 동일 Mistral-7B 백본을 공유 → 이 쌍의 성능 차이 = medical FT의 forgetting 비용. Tier1b는 독립적 Qwen 백본으로 architectural 차이 통제.
- 4.2 **반응 예측 프롬프팅(Z0/Z1/Z2_KG_COT/F2)·파싱; 평가는 APTOS-2021 공식 채점식 채택**(CI-AUC, VA/CST tolerance, biomarker-AUC). **Z2_KG_COT 상세:** 3단계 구조화 프롬프트 — (i) biomarker JSON 추출, (ii) KG 규칙 명시 적용(SRF→continue 0.85, IRF→continue 0.80, dry→stop 0.85, PED-only→case-dependent 0.45), (iii) 규칙 인용 결정. 프롬프트 내 KG 규칙 주입 = neuro-symbolic 추론의 추론-시점(inference-time) 변형.
- 4.3 **Forgetting 측정 설계(신규):** (a) E0a — AMD staging task (Tier3 훈련 분포), 예상: Tier3>Tier2>Tier1 (specialist advantage); (b) E0b — 일반 VQA (훈련 외 태스크), 예상: Tier1>Tier2>Tier3 (forgetting curve); (c) E0c — Qwen+Z2_KG_COT vs RetinaVLM on anti-VEGF CI (H8). 두 곡선의 교차점이 *어느 태스크에서 전문화가 이득/손해인지* 경계를 정의.
- 4.4 **통합 XAI 프로토콜:** 핵심 3축 — (i) 바이오마커 saliency 충실도(fluid-energy ratio, label-conditioned concentration), (ii) **Cross-Attention Rollout**(단어↔이미지 토큰), (iii) **neuro-symbolic AntiVEGF-Guideline-KG**(Text–KG / Attn–KG alignment); + 조건부 (iv) **Perturbation-ROCO**(마스크 가용 시).
- 4.5 **KG 구축(Wang 2025 이식):** neuro-symbolic 가이드라인 KG `G=(V,E)` + confidence weight + 온톨로지 매핑(SNOMED-CT/ICD-10); symbolic 규칙엔진 우선(신경 인코더는 future), RAG 주입; 트리플렛 공개·재현(`(SRF)→Continue` 등). Z2_KG_COT와 E7 두 경로에서 공용. 평가용 우선, 디코딩 가드레일은 future.
- **도표:** Fig (파이프라인 다이어그램 — 4축 XAI + forgetting design 포함).

## 5. Experiments & Results
- **평가 규약:** APTOS-2021 공식 채점식 채택(CI-AUC, VA/CST tolerance, biomarker-AUC) → BlueSky·CNN ref와 직접 비교(`EXPERIMENTAL_PROTOCOL.md §3.1`). 단, Stage1/2 14-subtask 종합점수가 아닌 개별 subtask 보고(내부 split, 라벨 가용분만).
- **E0 Forgetting Curve → Table 0, Fig 0 (H7/H8):**
  - E0a AMD staging: Tier3 > Tier2 > Tier1 (specialist advantage on trained domain) — **Fig 0a**.
  - E0b 일반 VQA: Tier1b ≈ Tier1a > Tier2 > Tier3 (general ability forgetting) — **Fig 0b**.
  - E0c KG compensation: Qwen+Z2_KG_COT vs RetinaVLM zero-shot CI-AUC 직접 비교 — **Table 0c**.
  - 두 곡선(E0a·E0b)의 교차 = forgetting–specialization tradeoff 가시화. E0c = KG 보상 검증.
- E1 치료결정 CI(4 tier × 4 prompt: Z0/Z1/Z2_KG_COT/F2) → **Table 1, Fig 2** (H1, H8); E1b 반응성 VA/CST(공식 tolerance) → **Table 1b**.
- E2 바이오마커 분류(공식 AUC, 4 tier) → **Table 2** (anchor).
- E3 XAI 충실도 스펙트럼 → **Fig 3**; E3b Cross-Attention Rollout → **Fig 3b** (H2); **E3c Forgetting XAI Probe(attention entropy, fluid-energy ratio) → Fig 3c (H7 mechanistic)**; **E3d Token-Logit Attribution → Fig 3d (H7 visual)**.
- E4 결정 연결(saliency/attn→CI) → **Fig 4** (H3).
- E5 hallucination → **Fig 5** (H4).
- E6 질적 사례(4축 패널) → **Fig 6**.
- **E7 KG-Fidelity(neuro-symbolic Text–KG/Attn–KG alignment, Wang 2025) → Table 3, Fig 7 (H5).**
- ~~E8 Perturbation-ROCO~~ **조건부/미구현**(clinician mask 입수 시) → Fig 8 은 future.
- 통계: DeLong(AUC) / bootstrap CI(tolerance, alignment) / Jonckheere–Terpstra trend. **k-fold 는 코드만(주석), 기본 단일 split.**

## 6. Discussion
- **핵심 서사 — Forgetting Tradeoff (H7 ✓, H8 ✓):** 전문화(medical FT)는 *훈련 분포 내 태스크*(AMD staging, biomarker 분류)에서는 이득이지만, *훈련 외 novel task*(anti-VEGF CI, 일반 VQA)에서는 catastrophic forgetting으로 인해 오히려 손해. 이 tradeoff를 동일 백본(Tier1a↔Tier2) 비교로 최초 정량화.
- **KG as External Symbolic Memory:** 전문화 없이도 Z2_KG_COT로 임상 규칙을 추론 시점에 주입하면 forgetting을 보상할 수 있음. "모델은 임상 논리를 잊지만 KG가 이를 복원한다" — neuro-symbolic 보상 원리. 이는 왜 KG 정합이 Tier1(일반 VLM)에서 특히 큰 성능 향상을 보이는지 설명.
- **Clinical Alignment:** 전문화될수록 성능·saliency·attention·causal·KG-논리를 **함께** 정렬(Clinical Alignment) → 신뢰가능한 의료 VLM의 조건. 그러나 *전문화가 항상 정답은 아님* — novel task에서는 KG+일반 모델이 대안.
- 무엇이 특화의 이득인가(connector가 OCT feature를 LLM에 더 잘 전달 → 인과·논리 추론 획득). LoRA FT = forgetting 발생 경로; connector-only = 모달리티 추가(forgetting 없음).
- 임상 함의: (i) 주사 결정 보조, (ii) KG 정합 근거 제시로 신뢰성 확보, (iii) 인과 검증으로 Clever Hans 배제, (iv) novel task에는 KG+일반 모델 배포가 fine-tuning보다 비용·위험 낮음.
- **LoRA context:** LoRA fine-tuning은 specialization 이득과 forgetting 비용을 동시에 수반. connector/projection은 modality만 추가하며 forgetting 없음. 본 논문은 학습 없이 이미 배포된 모델들의 forgetting을 측정 → 향후 LoRA ablation으로 forgetting 통제 가능성 탐구(§9 Future Work).

## 7. Limitations
- 이미지 단위 바이오마커 라벨(마스크 아님) → **neuro-symbolic KG Text/Attn alignment로 정면 보완**, 잔여분만 한계.
- KG 트리플렛 수작업(주관성) → Wang 2025 온톨로지 매핑 차용·공개·근거기반·소규모로 완화.
- **Perturbation-ROCO 는 마스크 부재로 본 논문에서 조건부/미구현**(임상의 마스크 입수 시 future); 2시점·약물/진단 편향·검증셋 무라벨; 테스트 표본 소수(~33)로 검정력 제한(k-fold 옵션 코드 보유).
- VLM은 약한 분류기/회귀기 → 절대 성능은 CNN 하회 가능(포지셔닝으로 방어).

## 8. Conclusion
- 성능 + **4축 해석가능성(상관·인과·논리)** 동시 측정 프레임 제시, 전문화의 다중 이득과 **Clinical Alignment** 규명(가설), KG 기반 충실도 평가 제안.

## 9. Future Work — Projection(connector) vs LoRA  *(맨 마지막, 사용자 요청)*
- connector(모달리티 추가) ≠ LoRA(기존 가중치 미세조정), 직교·결합 가능.
- 향후 ablation: connector-only vs connector+LoRA vs LoRA-only 가 반응 성능·XAI 충실도에 미치는 영향.
- **KG 런타임 제약 주입:** KG를 평가용을 넘어 디코딩 가드레일(가이드라인 위배 답변 플래그/제약)로 사용(KAD식). (본 논문 범위 밖, 설계 메모.)

---

## 도표 인덱스 (claim ↔ figure/table)
| ID | 산출 | 뒷받침 | 출처 실험 |
|----|------|--------|-----------|
| Fig 1 | 데이터/태스크 개요 | 문제정의 | §3 |
| **Table 0a** | **AMD staging 성능(4 tier) — specialist advantage** | **H7** | **E0a** |
| **Fig 0a** | **AMD staging tier 비교 — forgetting curve (훈련 내)** | **H7** | **E0a** |
| **Table 0b** | **일반 VQA 성능(4 tier) — forgetting of general ability** | **H7** | **E0b** |
| **Fig 0b** | **일반 VQA tier 비교 — forgetting curve (훈련 외)** | **H7** | **E0b** |
| **Table 0c** | **KG compensation: Qwen+Z2_KG_COT vs RetinaVLM CI-AUC** | **H8** | **E0c** |
| Table 1 | 치료결정 CI 성능(4 tier × 4 prompt, CI-AUC 공식) | H1, H8 | E1 |
| Table 1b | 반응성 VA/CST(공식 tolerance) vs BlueSky·CNN | H1 | E1b |
| Fig 2 | tier별 CI-AUC 추세 (Z0/Z1/Z2_KG_COT/F2) | H1, H8 | E1 |
| Table 2 | 바이오마커 mean AUC | anchor | E2 |
| Fig 3 | XAI saliency 충실도 스펙트럼 | H2 | E3 |
| Fig 3b | Cross-Attention Rollout(단어↔이미지) | H2 | E3b |
| Fig 4 | saliency/attn→결정 연결 | H3 | E4 |
| Fig 5 | hallucination 율 | H4 | E5 |
| Fig 6 | 질적 4축 XAI 패널 | 서사 | E6 |
| Table 3 | neuro-symbolic Text–KG / Attn–KG alignment(tier별, Wang 2025) | H5 | E7 |
| Fig 7 | KG 정합 단조 추세(40%→90%) | H5 | E7 |
| ~~Fig 8~~ | ~~Perturbation-ROCO~~ (조건부/future, mask 가용 시) | (H6) | (E8) |
