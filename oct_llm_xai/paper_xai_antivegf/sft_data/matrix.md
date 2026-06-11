# Multi-backbone instill matrix — v0.4 (STRUCTURE / template)

> ⚠️ **구조만 정의한 스켈레톤.** 셀 값은 Issue 2.3 GPU 재실행(v0.4 split, train 128 / test 69)
> 후 `assemble_matrix.py`가 채운다. v0.3 원본(test 35)은 `matrix.bak_v0.3.{md,json}`에 보존.
>
> 발견 라벨 (A)~(E) = `../../paper/DESIGN.md` §6. **두 평가 단위 분리** = `../../paper/EXPERIMENTAL_PROTOCOL.md` §4.4.
> 셀 표기: `—` = 미측정(TBD), `N/A` = 구조상 불가(tier1c arm C = linear-attn).

## Test set (v0.4, n=69 eyes / 470 pre-slices)
- **decision**: continue 44 / stop 25 — prior **0.638**
- **responder**: good 48 / poor 20 / no_active 1 — good/poor n=68, prior **0.706**
- **prognosis(ΔCST 4-class)**: marked 25 / partial 20 / minimal 13 / worsening 11 — majority **0.362**
- **per-slice biomarker (test-eye, 470 slice)**: IRF present 419 / absent 51 (absent 0.11 ⚠️ present-편향) ·
  SRF 184 / 286 · PED 79 / 391

---

## Table A — Per-slice Biomarker (지각 regime)
> per-slice, **per-biomarker 독립** (집계 금지). 셀 = balanced accuracy `[eye-clustered 95% CI]`.
> 누수방지: test-eye 슬라이스만. 균형인 **SRF·PED가 핵심 instill 증거**, IRF는 present-편향 caveat.
> 채점: `perslice_biomarker.py --tier <t> --arm <A|B|C|D>` → `perslice_{tier}_{arm}.json`.

| backbone             | arm | IRF balAcc[CI] | SRF balAcc[CI] | PED balAcc[CI] |
|----------------------|-----|----------------|----------------|----------------|
| RetinaVLM (tier3)    | A   | —              | —              | —              |
| RetinaVLM (tier3)    | B   | —              | —              | —              |
| RetinaVLM (tier3)    | C   | —              | —              | —              |
| RetinaVLM (tier3)    | D   | —              | —              | —              |
| LLaVA-Med (tier2)    | A   | —              | —              | —              |
| LLaVA-Med (tier2)    | B   | —              | —              | —              |
| LLaVA-Med (tier2)    | C   | —              | —              | —              |
| LLaVA-Med (tier2)    | D   | —              | —              | —              |
| Qwen3.6-27B (tier1c) | A   | —              | —              | —              |
| Qwen3.6-27B (tier1c) | B   | —              | —              | —              |
| Qwen3.6-27B (tier1c) | C   | N/A            | N/A            | N/A            |
| Qwen3.6-27B (tier1c) | D   | —              | —              | —              |

→ finding **(A)** 보이는 biomarker는 SFT로 instill (A→B 상승; SRF·PED 기준). 백본무관 기대.

---

## Table B — Eye-level CDSS (결정 regime)
> eye-level (n=69). 핵심 셀(decision balAcc, responder rGPbal)에 bootstrap 95% CI.
> 열: cont=continue_rate · prog=prognosis_node_acc · maj=prognosis majority(0.362) ·
> resp3=response 3-class acc · rMaj=responder majority · rGPbal=good/poor balAcc ·
> balAcc=decision balanced acc · cfFlip=cf flip-rate · faithGap=faithfulness gap · textKG=Text–KG self-consistency.
> 채점: `harness.py --tier <t> --arm <A|B|C|D> --mode eval [--meta]`.

| backbone             | arm    | cont | prog | maj   | resp3 | rMaj | rGPbal | balAcc | cfFlip | faithGap | textKG |
|----------------------|--------|------|------|-------|-------|------|--------|--------|--------|----------|--------|
| RetinaVLM (tier3)    | A      | —    | —    | 0.362 | —     | —    | —      | —      | —      | —        | —      |
| RetinaVLM (tier3)    | A_meta | —    | —    | 0.362 | —     | —    | —      | —      | —      | —        | —      |
| RetinaVLM (tier3)    | B      | —    | —    | 0.362 | —     | —    | —      | —      | —      | —        | —      |
| RetinaVLM (tier3)    | B_meta | —    | —    | 0.362 | —     | —    | —      | —      | —      | —        | —      |
| RetinaVLM (tier3)    | C      | —    | —    | 0.362 | —     | —    | —      | —      | —      | —        | —      |
| RetinaVLM (tier3)    | D      | —    | —    | 0.362 | —     | —    | —      | —      | —      | —        | —      |
| LLaVA-Med (tier2)    | A      | —    | —    | 0.362 | —     | —    | —      | —      | —      | —        | —      |
| LLaVA-Med (tier2)    | A_meta | —    | —    | 0.362 | —     | —    | —      | —      | —      | —        | —      |
| LLaVA-Med (tier2)    | B      | —    | —    | 0.362 | —     | —    | —      | —      | —      | —        | —      |
| LLaVA-Med (tier2)    | B_meta | —    | —    | 0.362 | —     | —    | —      | —      | —      | —        | —      |
| LLaVA-Med (tier2)    | C      | —    | —    | 0.362 | —     | —    | —      | —      | —      | —        | —      |
| LLaVA-Med (tier2)    | D      | —    | —    | 0.362 | —     | —    | —      | —      | —      | —        | —      |
| Qwen3.6-27B (tier1c) | A      | —    | —    | 0.362 | —     | —    | —      | —      | —      | —        | —      |
| Qwen3.6-27B (tier1c) | A_meta | —    | —    | 0.362 | —     | —    | —      | —      | —      | —        | —      |
| Qwen3.6-27B (tier1c) | B      | —    | —    | 0.362 | —     | —    | —      | —      | —      | —        | —      |
| Qwen3.6-27B (tier1c) | B_meta | —    | —    | 0.362 | —     | —    | —      | —      | —      | —        | —      |
| Qwen3.6-27B (tier1c) | D      | —    | —    | 0.362 | —     | —    | —      | —      | —      | —        | —      |

→ findings **(B)** 결정 collapse·예후 ≤majority · **(C)** attn-KL(arm C)도 불변 · **(D)** 메타 신호(A_meta) ·
   **(E)** SFT 역설(A_meta→B_meta 역전). v0.3에서 관측, v0.4로 재확인 예정.
