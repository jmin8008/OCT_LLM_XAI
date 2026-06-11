# 실험 프로토콜 (Experimental Protocol)

> **상태: 실험 완주(v0.3 전수, 2026-06-06).** 본 문서는 *무엇을 어떤 입력/arm/메트릭으로*
> 측정했는지를 확정 기술한다. 원자료: `sft_data/matrix.{md,json}`, 셀별
> `eval_{tier}_{arm}{,_meta}.json`, 분석 `MATRIX_ANALYSIS_v0.3.md`. 실제 코드: `code/harness.py`.
>
> **⚠️ 이전 버전(zero/few-shot 프롬프트 4-tier 스펙트럼 + forgetting curve E0a/b/c +
> APTOS 공식채점 + Perturbation-ROCO)은 폐기.** 본 논문의 실험은 **LoRA SFT instill 매트릭스
> (arm A/B/C/D + meta) × 3 백본**이다.
>
> **경로 규약:** 문서는 `oct_llm_xai/paper/`. 코드/데이터 경로(`code/`, `sft_data/`,
> `fluid_masks_v2/`)는 `oct_llm_xai/paper_xai_antivegf/` 기준 상대경로.

---

## 1. 백본 (backbone diversity axis)

전문화 "스펙트럼"이 아니라 **결과의 백본 불변성(backbone-invariance)** 을 보기 위한 3개의
이질적 VLM. 코드/매트릭스 호환을 위해 tier 라벨만 유지.

| 라벨 | 모델 | 아키텍처 | 성격 | instill arm |
|------|------|----------|------|-------------|
| **tier3** | RetinaVLM-Specialist | vision enc → Perceiver → LLaMA-3-8B (MiniGPT-4 계열) | OCT-특화 | A/B/C/D + meta |
| **tier2** | LLaVA-Med-v1.5-mistral-7b | vision enc → MLP → Mistral-7B (LLaVA 계열) | 의료-일반 | A/B/C/D + meta |
| **tier1c** | Qwen3.6-27B | Qwen3 MoE, gated-delta-net(linear-attn) | 일반 SOTA | A/B/**C=N/A**/D + meta |

- **공정성:** 동일 입력 영상·전처리·프롬프트·파싱 규칙. 학습 ~3 epoch, LoRA, AdamW lr 1e-4.
- **tier1c 제약:** (i) 27B → 학습 시 자동 **4-bit QLoRA**(`bb.quantized=True`), (ii) **arm C는
  N/A** — linear-attention(gated-delta-net)이라 attention rollout 불가, (iii) B_meta는 긴
  프롬프트로 일부 OOM(env-limited). ⚠️ flash-linear-attention(fla) 설치 금지(tf 5.3.0 비호환).
- **런타임 패치:** LLaVA-Med fork forward를 cache_position/logits_to_keep 흡수하도록 패치
  (`backbones.py` 반영). arm C 학습은 eager attention + bf16 autocast + grad-finite guard.

---

## 2. 예측 과제 (자동 채점 4종 — `harness.py:evaluate`)

모델은 4-step CoT 텍스트를 생성하고, 정규식 파서(`prompts.py`)가 노드별로 라벨화한다.

| 과제 | 노드 | 파서 | GT 출처 | 성격 |
|------|------|------|---------|------|
| **decision** | Step 3 continue/stop | `parse_ci` | APTOS `continue_injection` | 미래 함수 |
| **biomarker** | Step 1 IRF/SRF/PED | `parse_biomarkers` | 12×12 마스크 GT | 보이는 것(perceptual) |
| **prognosis** | ΔCST 4-bucket | `parse_prognosis` | 실측 ΔCST | 미래 함수 |
| **composite responder** | Step 4 good/poor/no_active | `parse_response` | ΔCST≤−25 OR ΔVA≥0.1 | 미래 함수 |

- decision은 클래스 불균형(prior 0.60) → **balanced accuracy**(sens/spec 평균) + continue_rate.
- responder는 good/poor 이진 **balanced accuracy**(no_active 제외, n=34) + 3-class acc(resp3).
- prognosis는 **majority baseline(0.314)** 과 병기 — 단일이미지 천장 검증용.

---

## 3. Instill 실험 설계 (arm A/B/C/D + meta) — 핵심

### 3.1 학습 데이터 (`sft_data/sft_kg_cot.json`, 생성기 `gen_sft_kg_cot.py`)
- **429행** = factual 218 + counterfactual 211. **v0.3 완주 split: train 360행(183 eye) /
  test 69행(35 eye)** (test는 factual 35 평가). ⚠️ **재실행 시 §4.2 최소포함조건 분할
  (test≈70)로 교체** — 본 절 수치는 현재까지 완주된 run 기준.
- 각 행: pre-treatment 이미지(clean) + 4-step CoT 프롬프트(진단명 미제공) + target CoT 텍스트
  + `nodes_gt`(채점 라벨). 4-step = Visual → Pathophysiology → **Decision** → **Response**
  (인과 순서; v0.2의 진단명·역순 제거).
- **divergent 77 eye**(가이드라인 vs 실제 결정 갈림)는 거짓 합리화 없이 정직 렌더.

### 3.2 arm 정의 (`harness.py:train`)
| arm | 학습 데이터 | loss | 의도 |
|-----|-------------|------|------|
| **A** | 없음(eval만) | — | zero-shot 천장 baseline |
| **B** | factual 183 (clean) | LM next-token (`lm_loss`) | "정형 CoT 추론을 텍스트로 instill" |
| **C** | factual 183 + 12×12 마스크 | LM + λ·KL (`attn_loss`, λ=0.5) | "attention을 유체에 정렬하면 결정이 고쳐지나" |
| **D** | factual 183 + counterfactual 177 (clean+occluded) | LM | "유체 픽셀에 결정이 인과 의존하도록" |

- **counterfactual 행**(arm D): occluded 이미지(유체 cv2 inpaint 제거) + 뒤집힌 라벨
  (IRF/SRF=0, dry_macula, **decision=stop**, no_active). `gen_sft_kg_cot.py:219–230`.
- **attention-KL**(arm C, `attn_kl.py`): 미분가능 torch rollout → 12×12를 6×6로 average-pool
  → 유체 마스크 union에 forward KL. `bb.enable_eager()`로 attention 텐서 노출.

### 3.3 meta 변형 (별개 arm 아님, 프롬프트 ablation 플래그 `--meta`)
- `augment_prompt`(`harness.py:29-38`)가 프롬프트에 **pre-treatment 임상 메타 5필드** 주입:
  `age, gender, drug, preVA, preCST`. **post/Δ(미래 결과)는 절대 금지(누설 방지).**
- **A_meta** = zero-shot + 메타(eval만). **B_meta** = factual SFT + 메타.
- 목적: (i) 막힌 결정/반응 신호가 pre-treatment 메타에 있는지(A_meta), (ii) SFT가 그
  메타 조건화를 보존/파괴하는지(A_meta vs B_meta) 분리.

### 3.4 재현
- 학습: `python3 -u harness.py --tier <t> --arm <B|C|D> --mode train [--meta]`
- 평가: `python3 -u harness.py --tier <t> --arm <A|B|C|D> --mode eval [--meta]`
- 매트릭스 조립: `python3 assemble_matrix.py` → `sft_data/matrix.{md,json}`.
- 무결성: `verify_v3.py` 18/18.

---

## 4. 데이터 분할 & 통계 — 두 평가 단위(per-slice / eye-level) 분리

본 논문은 **평가 단위가 다른 두 regime**을 명시적으로 분리한다(보고도 §4.4에서 별도 표):

| regime | 단위 | N | 라벨 출처 | 대상 finding |
|--------|------|---|-----------|--------------|
| **Per-slice Biomarker** | B-scan 슬라이스 | ~1501 pre-slice | `train_anno_pic.csv` per-image IRF/SRF/PED | (A) |
| **Eye-level CDSS** | 안구(eye) | 218 (CV/split) | eye-level decision·ΔCST·ΔVA | (B)(C)(D)(E) |

### 4.1 Per-slice Biomarker regime (지각 태스크)
- 슬라이스마다 유체 양상이 실제로 다름(eye 내 라벨 변동 IRF 22% / SRF 36% / PED 16%) →
  **per-slice presence 라벨은 진짜 신호**(broadcast 아님). 전체 pre-slice 풀에서 absent
  (negative)가 폭증: IRF 21→**197**, SRF 91→909, PED 149→1213(eye→slice).
- **누수 방지:** instill 모델(B/C/D)은 train-eye 대표슬라이스로 학습 → biomarker 평가는
  **§4.2의 test-eye에 속한 슬라이스만**(train-eye 슬라이스 평가 금지). test≈70 eye면 평가
  슬라이스 ≈ 480장(test-eye × ~6.9 pre-slice), 대표 1슬라이스(35장)보다 균형 negative 대폭 확보.
- **독립 per-biomarker 평가**: IRF/SRF/PED **각각** AUC/F1/balAcc를 따로 보고("biomarker를
  잘 찾는다"는 집계 금지). **IRF는 present-편향(0.87) caveat 명시**, 균형인 **SRF·PED가 진짜
  instill 증거**.
- **상관 보정:** 슬라이스는 eye 내 상관 → eye-clustered bootstrap CI(또는 eye당 1슬라이스
  서브샘플 robustness)로 보고.
- **마스크 불필요:** presence 라벨만 사용(12×12 마스크는 arm C attention-KL에만, 대표 슬라이스 유지).

### 4.2 Eye-level CDSS regime — 분할 & 최소포함조건(★신규)
- **분할:** 218 eye를 **eye-level**로 분할(같은 안구 pre/post 동일 fold; 슬라이스 증강은
  *학습 입력*에만 허용, **독립 test 표본으로 카운트 금지**=pseudo-replication 방지).
- **희소 negative 문제:** dry eye 7 / IRF-absent 21 (eye-level)뿐 → 무작위 분할 시 일부
  fold/test에 0개가 들어가 통계 신뢰도 붕괴.
- **★ 최소포함조건(minimum-inclusion) 검증 — 분할 채택 전 필수 게이트 (Core-only Quota):**
  층화 키 = **decision × responder × has_fluid**. 분할(또는 각 CV fold)의 **test 세트가
  아래 정족수를 모두 만족할 때만 채택**, 실패 시 seed 변경 재추출:
  - **dry eye(has_fluid=0) ≥ 2 — 유일한 핵심 제약(필수).** 7개뿐이라 test가 0~1로 떨어지면
    dry/responder=no_active 칸이 수학적으로 붕괴 → 최소 2 강제. (단 전체 7개라 채워도 검정력은
    약함 → Limitations에 정직 명시.)
  - **stop ≥ 1, poor responder ≥ 1 — 형식적 안전장치.** 두 클래스는 prior 30~36%라 test≈70에
    20여 개 자동 확보됨; "아예 0이 되지 않는다" 수준의 상징적 floor로만 두어 `verify_v3.py`가
    불필요한 정족수 미달로 멈추는 것 방지 + train 손실 최소화.
  - **IRF-absent quota 제거.** biomarker(A)는 §4.1 per-slice(IRF-absent 197장)로 평가되므로
    eye-level IRF-absent 제약은 불필요 → 둠.
  - 검증 스크립트: `data.assert_min_inclusion(split)` (신규; `verify_v3.py`에 게이트로 배선).
- **권장 구성(스코프·검정력 균형):** 단일 stratified hold-out **test ≈ 70 / train ≈ 148**
  + 위 정족수. (3-fold CV는 fold당 dry≈2로 빠듯 → *옵션*; 채택 시 동일 게이트를 fold별로 통과
  해야 하고, 실패하면 fold 수를 줄이거나 정족수를 낮춰 재설계.)
- **메트릭(셀별):** decision_balanced_acc, continue_rate, prognosis_node_acc + majority,
  response_goodpoor_balacc(rGPbal) + resp3, cf_flip_rate, negctrl_flip_rate,
  faithfulness_gap, text_kg_align.

### 4.3 통계
- **유의성:** test 소표본 → 핵심 셀(tier3 B responder rGPbal, tier2 A_meta rGPbal)에
  **bootstrap 95% CI + permutation test**를 붙여 표에 **n.s. 명시**. 셀 간 단순 비교는 질적
  패턴(3백본 반복)으로만 주장.

### 4.4 보고 양식 — 두 표 분리(★ matrix.md / 논문 공용)
`assemble_matrix.py`는 **두 개의 별도 표**를 emit한다(현 단일 표 폐기). 원자료
`sft_data/matrix.{md,json}`(데이터 폴더 `paper_xai_antivegf/sft_data/`에 위치):

- **Table A [Per-slice Biomarker]** — 행: backbone × arm(A/B/C/D). 열: IRF / SRF / PED
  각각의 (balAcc 또는 AUC) + n_present/n_absent. IRF 편향 각주.
- **Table B [Eye-level CDSS]** — 행: backbone × arm(A/B/C/D) × meta(±). 열: decision balAcc·
  continue_rate · responder rGPbal·resp3 · prognosis vs majority · cf_flip·negctrl·faithGap ·
  text_kg_align. 핵심 셀 bootstrap CI.

---

## 5. XAI / KG (격하된 보조 축)

> 이전 4축(saliency 충실도·rollout·KG 정합·perturbation-ROCO)에서, **마스크 부재**로 인해
> 인과/시각 축은 조건부/미가동임이 확정됐다. 본 논문에서 XAI는 *결정 collapse의 기제 설명*에
> 종속된다.

- **Text–KG self-consistency (가동):** `kg_align.text_kg_aligned` — VLM이 보고한 biomarker를
  symbolic 규칙엔진(`kg.py`, Wang 2025 이식)에 통과시켜 *기대 결정*을 도출, VLM의 실제 결정과
  **자기정합성**을 채점(정답 대비 아님). harness eval에 상시 배선. Wang 기준선 >85% 참조.
- **attention rollout (arm C 증거로만):** 학습 KL 감소(2.09→0.9)로 attention이 유체로
  끌려옴(FER↑)을 보이는 데 사용. 결정 불변과 대비 → "attention≠explanation".
- **Attn–KG consistency (조건부/미가동):** 임상의 픽셀 마스크 필요 → 없으면 `None`(조작 안 함).
- **Perturbation-ROCO (미구현 stub `roco_stub.py`):** 마스크 가용 시에만. 본 논문 본문 제외.
- **occlusion perceptibility (`perceptibility_check.py`):** arm D occlusion이 백본에 *지각
  가능한가*를 검증 — RetinaVLM엔 지각불가(confound 확정). **Qwen 재실행 필요**(tier1c D 예외 검증).

---

## 6. 실험 매트릭스 (요약)

| 실험 | 산출 | 뒷받침 | 산출물 |
|------|------|--------|--------|
| **E-A instill 매트릭스** (3 백본 × arm A/B/C/D + meta A/B) | 셀별 12 메트릭 표 | (A)~(E) 전체 | `matrix.{md,json}` |
| **E-A1 biomarker instill** | A→B 상승, **per-slice·per-biomarker(IRF/SRF/PED 분리)**; SRF·PED(균형)가 핵심, IRF 편향 caveat | (A) | Table A |
| **E-A2 decision collapse** | balAcc≈chance, cont 상수(3백본) | (B) | matrix |
| **E-A3 attention-KL** | KL↓·FER↑ but 결정 불변 (tier3/tier2 C) | (C) | `attn_kl`, eval_C |
| **E-A4 prognosis ceiling** | prognosis ≤ majority | (B) 정보론적 천장 | matrix |
| **E-A5 meta signal** | A_meta balAcc/rGPbal 도약(tier2) | (D) | eval_*_A_meta |
| **E-A6 SFT 역설** | A_meta 0.758→B_meta 0.500 역전(tier2) | (E) | eval_tier2_{A,B}_meta |
| **E-B perceptibility** | occlusion 지각가능성(RetinaVLM 불가; Qwen 재실행 예정) | arm D confound 판정 (cf. (B)) | `perceptibility_check.json` |
| **E-C meta-ceiling** | A_meta vs Part A logreg responder 천장 정렬 | (D) 정당성 | `meta_ceiling.py` |
| **E-D 역전 case-level** | tier2 zero-shot 적중 vs SFT 오답 eye diff + preCST-only ablation | (E) 핵심 figure | (분석 필요, HANDOFF #1) |

> **남은 분석(논문 작성 전 권장, HANDOFF 다음작업):** ① A_meta>B_meta case-level diff(C3
> 핵심 figure) ② Qwen perceptibility_check 재실행(D 예외 검증) ③ meta-ceiling 정렬 ④ 핵심
> 셀 bootstrap CI/permutation.

---

## 7. 자산 맵 (`code/`)

| 용도 | 파일 |
|------|------|
| train/eval 하네스(arm A/B/C/D + meta) | `harness.py` |
| 백본 로더/LoRA/attn_loss/generate | `backbones.py` |
| attention-KL (rollout + pool + forward KL) | `attn_kl.py` |
| SFT 데이터 생성(4-step CoT + counterfactual) | `gen_sft_kg_cot.py` |
| occlusion(cv2 inpaint) | `occlude.py` (cv2가 ~/.local → PYTHONNOUSERSITE 없이 실행) |
| KG 규칙엔진 / self-consistency 채점 | `kg.py`, `kg_align.py`, `antivegf_guideline_kg_v2.json` |
| occlusion 지각가능성 | `perceptibility_check.py` |
| 메타 천장(logreg) | `meta_ceiling.py` |
| 매트릭스 조립 / 무결성 | `assemble_matrix.py`, `verify_v3.py` |
| 프롬프트/파서 | `prompts.py` |

---

## 8. 향후 과제 (맨 마지막)

- **multi-modal / longitudinal CDSS:** 단일 pre-image의 정보론적 천장 → 환자 기저질환·과거력·
  복약·시계열 OCT를 통합하는 구조가 본질적 해법(본 논문의 핵심 제언).
- **connector(projection) vs LoRA:** connector=모달리티 추가(forgetting 없음) vs LoRA=가중치
  미세조정. SFT 역설이 LoRA 경로에서 발생 → connector-only 전문화 레버 ablation.
- **KG 런타임 가드레일:** self-consistency 채점을 넘어 디코딩 제약으로(KAD식). 본 논문 범위 밖.
