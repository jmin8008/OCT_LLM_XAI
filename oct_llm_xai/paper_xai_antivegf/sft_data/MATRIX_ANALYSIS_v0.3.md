# v0.3 매트릭스 분석 (2026-06-06)

원자료: `sft_data/matrix.{md,json}`, 셀별 `eval_{tier}_{arm}{,_meta}.json`.
재현: `verify_v3.py` 18/18 → tier3·tier2·tier1c 17셀(tier1c arm C=N/A) 전수 v0.3.

## 테스트셋 라벨 분포 (n=35 eyes)
- **결정**: continue 21 / stop 14 → continue prior **0.60**.
- **responder**: good 24 / poor 10 / no_active 1 → binary good/poor n=34, good prior **0.71**, balAcc chance 0.5.
- **예후(ΔCST 4-class)**: minimal 11 / marked 10 / partial 9 / worsening 5 → majority **0.314** (class당 ~9, worsening 5).
- ⚠️ n이 작아 셀 간 소수점 차이는 대부분 비유의. 신뢰할 건 **3백본에 반복되는 질적 패턴**.

## 1. 확고한 결과 (backbone-general)
- **(A) 보이는 concept(biomarker)는 SFT로 instill됨 — 유일한 명백 positive.** tier3 0.317→0.810, tier2 (파싱불가)→0.752, tier1c 0.800→0.829. arm C(attn-KL)도 B와 동급(tier3 0.829). 단서: biomarker GT가 SFT 타깃에 포함 → 상승 일부는 "정형 CoT 낭독" 학습일 수 있으나 held-out present/absent 적중은 진짜.
- **(B) continue/stop 결정 collapse — 중심 negative.** 모든 학습 arm이 상수 결정으로 붕괴: B/C→continue(cont 0.94–1.0, prior 0.60 초과), D→stop(cont 0.0). balAcc 0.42–0.50=chance, 예외 없음. 라벨=치료반응(미래) 함수 → 단일 pre-image로 도출 불가, **3백본 재현**.
- **(C) attention≠explanation 일반성.** tier3 C·tier2 C가 B와 사실상 동일. 학습 KL 절반↓(attention은 유체로 끌려옴=FER↑)인데 결정/예후/responder 전부 불변 → attention 정렬해도 grounding 안 고쳐짐(training-time, 백본 무관).

## 2. 가장 흥미로운 부분 — 메타데이터 × SFT
- **(D) 메타에 사라진 신호 일부 존재(한계=부분적 정보부족).** tier2 **A_meta: balAcc 0.619, rGPbal 0.758, resp3 0.824** — 이미지-only(chance)에서 도약. age/gender/drug/preVA/preCST 5필드만으로 zero-shot LLaVA-Med가 responder 적중. 누설 아님(post/Δ 금지 verify). preCST↔ΔCST 여지 상관=정당 신호(Part A 메타-천장 logreg 예후 0.457과 일관). **백본 의존**: tier3 A_meta 미동(0.435), tier1c A_meta responder 파싱불가(cont 0) → 메타 활용은 *지시따르기 능력*에 의존.
- **(E) ★ SFT가 그 메타 신호를 덮음 (v0.3 새 메시지·가장 novel).** tier2 **A_meta rGPbal 0.758 → B_meta 0.500**(cont 1.0, resp3 0.686=majority). 정형 CoT SFT가 모델을 text-prior 낭독기로 만들어 zero-shot의 메타 조건화를 중단시킴. 함의: "추론 구조 instill" SFT가 베이스의 *판별 능력을 파괴*할 수 있음(구조 instill ≠ 결정능력 instill, net-negative 가능). ⚠️ 역전은 tier2 한정(A_meta가 애초 높았던 유일 백본) → "메타가 zero-shot 우위를 준 곳에서 SFT가 지웠다"로 한정 서술.

## 3. 조사 필요한 이상치 (tier1c)
- **tier1c D**: cont **1.0**(all-stop 아님), cfFlip **1.0**, faithGap **+0.118**. Qwen은 occlusion에 continue→stop flip, 유체-occluded가 negctrl보다 +0.118 더 flip = 유일한 양의 faithfulness 신호. 단 이슈#1에서 occlusion은 RetinaVLM엔 지각불가(confound)였음 → Qwen은 비전 강해 지각될 수도/generic 픽셀반응일 수도. **Qwen perceptibility_check 재실행이 선결.**
- **tier1c B_meta**: cont 0.714(부분 비collapse), cfFlip 0.458 — Qwen은 메타+SFT에도 완전붕괴 안 함(rGPbal 0.479=chance).

## 4. 해석 주의 (over-claim 금지)
1. 검정력: n=35(responder 34). tier3 B rGPbal 0.613, tier2 A_meta 0.758 모두 수 eye 차 → 개별 유의성 없음. "약함/단일백본/n.s." 명시.
2. cfFlip·faithGap 대부분 무의미: 상수붕괴 시 flip 불가 → tier3/tier2 0값 정보없음. D occlusion은 이슈#1에서 지각불가 confound. 의미있는 건 tier1c뿐(검증 필요).
3. 예후: 대부분 0.257 < majority 0.314. 단일이미지 예후도 사실상 학습불가(4-class, class당 ~9 노이즈 큼).
4. D arm: tier3/tier2 all-stop은 grounding 아니라 지각불가 occlusion 라벨 prior(이슈#1 확정) — 과대해석 금지.

## 한 줄 요약
보이는 건 백본 무관 instill되나, 단일 pre-image로 continue/stop·예후·종합반응 instill은 정보론적으로 막혀 있고(3백본 확증) — 막힌 신호 일부는 pre-treatment 메타에 있으나 **정형 CoT SFT가 그 메타 판별력을 덮어버린다(tier2 A_meta→B_meta 역전)**가 v0.3의 새 메시지.

## 다음 작업 (우선순위)
1. **[최우선] A_meta>B_meta 역전 정밀 분석** — tier2에서 zero-shot은 맞고 SFT는 틀리는 eye case-level diff + preCST 의존도(메타 ablation: preCST-only vs 전체) → "SFT가 무슨 신호를 덮나" 규명. (E)를 논문 핵심 figure로.
2. **Qwen perceptibility_check 재실행** — tier1c D cfFlip 1.0/faithGap +0.118이 진짜 유체-grounding인지 generic 픽셀반응인지. 이슈#1을 강한 비전 백본으로 재실행(`code/perceptibility_check.py`를 tier1c로).
3. **메타-천장 대비 정렬** — tier2 A_meta 0.758을 Part A logreg responder 천장과 직접 비교(천장 초과면 의심, 아래면 정당). `code/meta_ceiling.py` responder 버전.
4. **유의성 표기** — 핵심 셀(tier3 B resp, tier2 A_meta)에 bootstrap CI/permutation test → 표에 n.s. 명시.
5. **논문 표/서사 갱신** — matrix.md → paper.tex 결과표 교체, 서사를 (A)(B)(C) + (E) 중심으로 재구성.
