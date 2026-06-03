# 논문 설계: Anti-VEGF 반응 예측을 위한 VLM 스펙트럼 + 임상 근거 기반 XAI

> 상태: **설계(Design) 단계 — 학습/구현 없음.** 본 폴더의 3개 .md + `oct_llm_xai/4_antivegf_vlm_spectrum_xai.ipynb`(텍스트 전용)는 논문 골격이며, 실제 실험은 승인 후 별도 단계에서 진행한다.
>
> **작업 폴더:** `oct_llm_xai/` (설계 산출물은 `oct_llm_xai/paper_xai_antivegf/`). 본 문서의 모든 경로는 **repo 루트 `OCT_LLM_XAI/` 기준 상대경로**다.

---

## 1. 한 줄 요약 (Thesis)

> VLM이 **일반 → 의료-일반 → 망막-OCT 특화**로 전문화될수록 anti-VEGF 치료반응 예측 성능이 좋아지는가, 그리고 그 XAI가 **임상적으로 더 충실(faithful)** 해지는가 — 즉 임상의가 치료 지속을 결정할 때 보는 **유체(fluid) 바이오마커**에 모델이 실제로 주목하는가?

성능(performance)과 해석가능성(interpretability)을 **하나의 축(전문화 정도)** 위에서 동시에 측정하는 것이 핵심 차별점이다.

> **핵심 재프레이밍:** 단순히 "전문화가 성능을 올리는가"가 아니라, **"전문화의 인지적 비용(catastrophic forgetting)은 무엇이며, neuro-symbolic KG 유도로 이를 보상할 수 있는가?"** 가 본 논문의 중심 질문이다. Tier1a↔Tier2 동일 Mistral 백본 비교가 망각을 직접 정량화하고, Z2_KG_COT 프롬프트가 KG 보상 경로를 검증한다.

---

## 2. 동기 (Motivation)

- **임상 니즈:** Anti-VEGF 주사(Avastin/Eylea 등)는 nAMD·DME·PCV 치료의 1차 선택이지만, 환자마다 반응이 다르고 "주사 지속/중단" 결정은 OCT 상의 유체 소견(IRF/SRF/PED)과 시력 변화에 의존한다. 반응을 사전·조기에 예측하면 불필요한 주사와 비용을 줄일 수 있다.
- **해석가능성 갭:** 의료 VLM은 그럴듯한 보고서를 생성하지만 *근거가 임상적으로 타당한지*는 검증되지 않는다. 기존 XAI 평가는 pseudo-GT(GradCAM)와의 IoU 같은 약한 기준에 머물러 있다 (본 repo `experiments/exp3_outputs/iou_pointing_game_results.json`에서 IoU≈0.0007로 사실상 무의미).
- **기회:** APTOS-2021 데이터셋은 **이미지 단위 바이오마커 라벨(IRF/SRF/PED/HRF)** 을 제공한다. 이는 XAI saliency를 *실제 임상 소견*과 정량적으로 대조할 수 있게 해주는 드문 자원이다.

---

## 3. 데이터셋 (Dataset)

**경로:** `data_response/anti-vegf-dataset/APTOS-2021/Final Datasets/`
**데이터 사전:** `data_response/anti-vegf-dataset/APTOS-2021_analysis.md`

| 항목 | 내용 |
|------|------|
| 모달리티 | **OCT B-scan** (1264×596 JPG), pre/post-injection 페어, 안구(eye) 단위 |
| 규모 | 학습 221 안구(라벨 有, 2,864 이미지) / 검증 342 안구(라벨 無, 4,492 이미지) |
| 1차 반응 라벨 | **`continue injection`** (0=중단, 1=지속) — 142:79 ≈ 64%/36% |
| 2차 회귀 타깃 | **VA**(치료 후 시력), **CST**(치료 후 중심망막두께 μm); 기저값 `preVA`,`preCST` 제공 |
| 이미지 바이오마커 | **IRF 84% / SRF 34% / PED 18% / HRF 96%** (이미지 단위 이진 라벨) |
| 임상 메타 | age, gender, diagnosis(DME 63% / CNVM 31% / PCV 6%), anti-VEGF drug(Avastin 77% 우세) |

**주의/한계 (논문에 명시):**
- 바이오마커는 **이미지 단위 이진 라벨**이며 **세그멘테이션 마스크가 아님** → XAI 정렬은 *근사*이다.
- 시점이 **pre/post 2개뿐**, 명시적 follow-up 기간 없음 → 장기 progression 모델링 불가.
- **약물·진단 편향**(Avastin 77%, DME 63%) → 약물·진단별 세부 분석은 통계력 제한.
- 검증셋 342 안구는 라벨이 없으므로 내부 평가는 학습 221 안구의 patient-level 분할로 수행.

---

## 4. 선행 작업과의 차별성 (Positioning)

| 선행 자산 | 무엇을 했나 | 본 논문과의 관계 |
|-----------|-------------|------------------|
| `autoresearch_task1/2/3` | CNN/앙상블로 바이오마커 분류(AUC 0.9636), CST 회귀(cal_tol 0.606), VA/CI 예측 | **VLM 아님, XAI-임상 연결 없음.** 본 논문의 *비-VLM 상한 앵커(reference)* 로만 인용 |
| `paper_ai4research/` | AutoResearch 파이프라인(메타-방법론) 논문 | VLM·해석가능성 주제 아님 → **주제 비중복** |
| `experiments/` nb 1–3 | XAI 스택 구축(GradCAM×4, LLM attention, VL-saliency, Setting A/B, hallucination) — OCT2017 NORMAL vs CNV | **참고할 도구**(drop-in 아님, 적응 필요). 본 논문은 *반응 예측 + 바이오마커 정렬*로 과제 전환 |
| **Wang 2025 (Sensors 25:6879)** | **Neuro-symbolic + LLM** AMD 예후 프레임워크: 온톨로지 KG(SNOMED-CT/ICD-10) + symbolic 규칙 + RAG-LLM 내러티브; >85% 규칙-지지 추론, >90% 바이오마커 인용, AUROC 0.94 | **E7 KG 축의 1차 방법론 출처** — 본 논문의 AntiVEGF-Guideline-KG·alignment 지표가 이를 직접 이식 |
| `experiments_product/5_kad_*.ipynb`, `exp4/exp5_outputs/` | **KAD 구현**: OCT-KG(24 entities/40 relations) + Knowledge Encoder(PubMedBERT) + DQN cross-attention | **부차 아키텍처 참고(스캐폴딩)** — KG 데이터구조/인코더 코드 참고용. DQN 강화학습 모방은 본 논문 채택 안 함 |

→ **본 논문의 빈자리(niche):** (a) anti-VEGF 반응을 **generalist→specialist VLM 스펙트럼**으로 최초 벤치마크(※ 기존 anti-VEGF 작업은 전부 CNN/앙상블, **VLM 추론 전무** → 본 파이프라인 자체가 신규), (b) XAI를 정성적 히트맵에서 **임상 바이오마커 기반 정량 충실도 지표**로 격상, (c) **neuro-symbolic 임상 가이드라인 KG**(Wang 2025 이식)로 언어-논리 정합을 정량 검증하여 마스크 부재 약점을 우회, (d) **동일 Mistral 백본(Tier1a=LLaVA-v1.6-mistral, Tier2=LLaVA-Med-mistral) 쌍**으로 medical fine-tuning에 의한 catastrophic forgetting을 직접 정량화(기존 연구는 backbone이 달라 forgetting과 architecture 차이가 혼재).

**LLaVA-v1.6-mistral-7b(Tier1a)** 추가 배경: 기존 설계는 Tier1을 단일 일반 VLM으로 두었으나, Tier1a(LLaVA-v1.6-mistral)와 Tier2(LLaVA-Med-mistral)가 **동일 Mistral-7B 백본**을 공유한다는 사실이 핵심 실험 레버를 제공한다. LLaVA-v1.6-mistral은 HuggingFace에서 `llava-hf/llava-v1.6-mistral-7b-hf`로 즉시 접근 가능하며, LLaVA-Med v1.5도 동일 Mistral 기반(공식 릴리스 확인). Tier1c(Qwen3.6-27B)는 별도 Qwen 백본의 SOTA instruction follower로 일반 VLM 두 번째 기준선 역할.

**융합 출처:** **Wang 2025(Sensors 25:6879)의 neuro-symbolic 가이드라인 KG**(온톨로지 노드 + confidence-weighted 인과 엣지 + symbolic 규칙 + RAG-LLM 내러티브, >85%/>90% 정량 해석가능성)를 본 스펙트럼 연구의 KG 축으로 이식한다. anti-VEGF 투여결정은 본질적으로 규칙(IF biomarker THEN continue/stop)이므로 **강화학습(DQN) 모방보다 규칙기반 neuro-symbolic 이 자연스럽다.** KAD(UMLS KG + DQN-attention) 구현은 KG 데이터구조·Knowledge Encoder *코드 스캐폴딩 참고*로만 활용(`experiments_product/`).

**자산 현실(코드 탐색 검증, 2026-05-30):** 대부분 **참고/적응** 대상이며 drop-in 재사용은 일부(center 검출, `score_cst_tolerance`, training_utils)뿐. **신규 작성 필요:** `score_va`, cross-attention rollout, **AntiVEGF-Guideline-KG(Wang 2025 기반, symbolic 규칙 엔진 우선)**, Text/Attn–KG alignment, anti-VEGF VLM 추론 파이프라인 (→ `EXPERIMENTAL_PROTOCOL.md §6.2`). Perturbation-ROCO 는 임상의 제공 마스크 입수 시에만 진행하는 **조건부 항목**으로 강등.

---

## 5. 기여 (Contributions)

1. **VLM 스펙트럼 벤치마크** — anti-VEGF 치료결정(`continue injection`)·반응성(VA/CST)을 zero-/few-shot으로 4개 모델(Tier1a/1b 일반·Tier2 의료-일반·Tier3 OCT-특화)에서 측정. **APTOS-2021 공식 채점식(CI-AUC, VA/CST tolerance, biomarker-AUC) 채택**으로 BlueSky(1위) 및 기존 CNN 레퍼런스와 **직접 비교**(`EXPERIMENTAL_PROTOCOL.md §3.1`).
2. **통합 XAI 프로토콜(3+1축)** — (i) 바이오마커 기반 saliency 충실도, (ii) **Cross-Attention Rollout**(단어↔이미지 토큰 연결, forward-pass 내재), (iii) **neuro-symbolic KG 정합**을 핵심 3축으로 결합 + (iv) **Perturbation-ROCO**(유체 occlusion 인과 의존성)는 *조건부*(임상의 마스크 입수 시). 정성 히트맵을 넘어선 *상관+논리(+조건부 인과)* 입체 해석.
3. **Neuro-symbolic KG 기반 정량 검증** — **Wang 2025(Sensors 25:6879)** 의 neuro-symbolic 가이드라인 KG 를 이식한 경량 **AntiVEGF-Guideline-KG**(symbolic 규칙 엔진 우선)로 VLM 답변의 **Text–KG alignment**(언어-논리, Wang 의 >85% rule-supported 기준선)와 **Attn–KG consistency**(시각-논리, >90% biomarker-citation 기준선)를 측정. **마스크 부재라는 기존 약점을 정면 보완**, 전문화에 따른 의학적 인과 추론 획득을 정량화.
4. **Forgetting 정량화(신규)** — **Tier1a(LLaVA-v1.6-mistral, 일반)↔Tier2(LLaVA-Med-mistral, 의료 FT)** 동일 Mistral-7B 백본 쌍으로 medical fine-tuning에 의한 **catastrophic forgetting**을 직접 측정. Tier3(OCT FT)까지 포함하면 전문화-일반화 tradeoff 곡선이 완성된다. 이를 통해 *"전문화가 비-훈련-도메인 태스크에서 오히려 해롭다"*는 forgetting 가설(H7)을 최초 검증.
5. **KG as Forgetting Compensation(신규)** — **Z2_KG_COT** 프롬프트(3단계 KG 가이드 CoT: biomarker JSON 추출 → KG 규칙 적용 → 규칙 인용 결정)로, 일반 VLM(Qwen+KG)이 fine-tuning 없이 OCT 특화 모델에 버금가는 CI-AUC를 달성할 수 있는지 검증. **"모델은 임상 논리를 잊지만 KG가 이를 복원한다"** 는 neuro-symbolic 보상 원리를 정량 입증(H8).
6. **Clinical Alignment 규명(핵심 서사)** — 전문화될수록 *의사 사고흐름(가이드라인 인과)* 과 *AI 내부 연산(attention·saliency·causal)* 이 정렬됨을 4축 동시 입증. "모델이 보는 곳↔치료 결정" 연결 + hallucination/실패 특성화 포함.
7. **향후 과제(맨 마지막):** connector/projection vs LoRA 전문화 레버, KG 런타임 제약 주입 (→ `EXPERIMENTAL_PROTOCOL.md §4.4·§7`, `PAPER_OUTLINE.md` 마지막 절).

---

## 6. 핵심 가설 (Hypotheses)

- **H1 (성능):** 반응 예측 성능은 tier1a/1b < tier2 < tier3 순으로 증가한다 — **실험 결과: CI task에서 기각** (Qwen3.6>RetinaVLM), AMD staging에서 부분 지지(도메인 갭 주의).
- **H2 (충실도):** XAI-바이오마커 정렬도 또한 tier1 < tier2 < tier3로 증가한다.
- **H3 (연결):** 유체 영역에 대한 saliency/attention 집중도가 `continue injection` 결정과 연관된다(모델이 임상 논리를 따르는가).
- **H4 (hallucination):** 전문화가 진행될수록 무소견(정상) 영상에서의 허위 병변 생성률이 감소한다(nb2의 NORMAL 100% hallucination 대비).
- **H5 (KG 정합):** 답변의 Text–KG alignment가 tier1(~40%) → tier3(≥90%)로 단조 증가한다(전문화가 의학적 인과 추론을 부여). **부분 지지**: Qwen3.6 Z1=0.955(Wang >85% 기준선 달성), RetinaVLM Z1=0.571.
- **H6 (인과):** 유체 영역 perturbation 시 CI logit 하락폭(causal specificity)이 전문화될수록 커진다(판단이 해당 소견에 인과 종속).
- **H7 (forgetting) ✓ 지지**: 훈련 분포 외 태스크(anti-VEGF CI)에서는 **tier1b ≥ tier1a > tier2 > tier3** — 즉 전문화될수록 일반 추론이 더 많이 망각되어 novel task 성능이 역전된다. AMD staging 같은 *훈련 내 태스크*에서는 반대(tier3 > tier2 > tier1)로, 두 곡선의 교차가 forgetting–specialization tradeoff를 가시화한다.
- **H8 (KG 보상) ✓ 지지**: Qwen3.6+Z2_KG_COT CI-AUC=0.607 > RetinaVLM+KG=0.440. Qwen(Tier1b)+Z2_KG_COT의 CI-AUC가 RetinaVLM(Tier3) zero-shot CI-AUC 이상 — fine-tuning 없이 KG 가이드 CoT만으로 specialist 격차를 좁히거나 역전한다.

---

## 7. 산출물 (Deliverables)

| 파일 | 내용 |
|------|------|
| `DESIGN.md` (본 파일) | 동기·논지·기여·포지셔닝·데이터·가설 |
| `EXPERIMENTAL_PROTOCOL.md` | 모델·프롬프트·분할·메트릭·XAI 방법·통계·ablation |
| `PAPER_OUTLINE.md` | 섹션별 골격 + 각 섹션이 뒷받침할 주장 + 도표 매핑 |
| `oct_llm_xai/4_antivegf_vlm_spectrum_xai.ipynb` | 텍스트 전용 스토리보드 노트북 (nb2 구조 모사) |

---

## 8. 위험 요소 (Risks)

- VLM은 본질적으로 약한 분류기/회귀기 → 반응 예측 절대 성능이 CNN보다 낮을 수 있음. **포지셔닝을 "해석가능성·일반화 연구"로 잡아 절대 leaderboard 경쟁을 피한다.**
- 일반 VLM이 OCT를 거의 못 읽어 tier1이 바닥일 위험 → 그 자체가 H1의 증거이자 "도메인 특화의 가치" 서사. 또한 E0b(일반 VQA)에서 tier1이 tier3을 앞서는 것은 **H7(forgetting)의 증거**로 서사화.
- 이미지 단위 라벨의 근사성 → **neuro-symbolic KG 기반 Text/Attn alignment(§4.4)로 정면 보완**(픽셀 마스크 없이 언어-논리 정량 검증). 잔여 한계로 명시하고 향후 세그멘테이션 마스크를 추가 보강책으로 제안.
- KG 트리플렛이 수작업 구축 → 주관성 위험. 완화: **Wang 2025 의 온톨로지 매핑(SNOMED-CT/ICD-10) 방식 차용**, 공개 가이드라인·APTOS 라벨 정의에 근거, 트리플렛 전체 공개로 재현성 확보, 규칙 수를 작게(수십 개) 유지.
- **Perturbation-ROCO(E8)는 유체 마스크가 필요한데 본 데이터셋은 이미지 단위 라벨뿐 → 임상의 제공 마스크 입수 시에만 진행하는 조건부 실험으로 강등**(미구현 stub 만 유지). region heuristic 기반 근사는 causal specificity 신뢰도가 낮아 본 논문 본문에서 제외.
- **Forgetting 측정의 confound 위험:** Tier1a↔Tier2 비교에서 forgetting 외에 instruction-tuning 품질 차이가 혼입될 수 있음. 완화: E0b(순수 일반 VQA, OCT 무관)에서 두 모델의 기초 능력 비교로 통제. Tier1a·Tier2 모두 instruction-tuned 버전 사용.
- **Z2_KG_COT 파싱 복잡도:** 3단계 JSON 출력 요구가 LLM 버전에 따라 불안정. 완화: JSON 파싱 실패 시 "uncertain" 집계 + 파싱 성공률 별도 보고(robustness metric).
