# HANDOFF — XAI-Instill Vision Capabilities (Anti-VEGF CI)

> 갱신 2026-06-11. 다음 세션 인계용.

## ▶▶ 다음 세션 시작점 (2026-06-11)
**지금까지(GPU-free 준비 전부 완료):** spine 재정렬+`../paper/` 이동 → v0.4 split(train128/
val21/test69, Core-quota) → SFT 재생성 → per-slice 채점기 빌드·검증 → matrix.md를 v0.4
2-table 템플릿으로 교체(v0.3=`matrix.bak_v0.3.{md,json}`). **남은 건 거의 GPU 추론.**

**바로 할 일 (우선순위):**
1. **[GPU-free] Issue 2.3a — `assemble_matrix.py` 2-table emit 수정.** Table A(per-slice:
   `perslice_{tier}_{arm}.json`) + Table B(eye-level: `eval_{tier}_{arm}{,_meta}.json`)을
   `sft_data/matrix.md`(이미 v0.4 구조 정의됨)에 채우게. + `verify_v3.py`에
   `data.assert_min_inclusion` 게이트 배선.
2. **[GPU] Issue 2.3 — v0.4 전수 재실행.** 첫 런 권장 = **tier3 arm A per-slice**
   (`perslice_biomarker.py --tier tier3 --arm A`, 학습불필요 zero-shot baseline, 파이프라인
   smoke). 그 뒤 eye-level `harness.py --tier tier3 --arm A/B/C/D [--meta] --mode train/eval`.
   순서: tier3·tier2 먼저(빠름) → tier1c(QLoRA 느림) 후순위.
3. 환경: GPU=`conda activate oct_llm` + LD_PRELOAD nvjitlink + PYTHONNOUSERSITE=1 +
   CUDA_VISIBLE_DEVICES=2 (래퍼+`python3 -u`). 분석=`PYTHONNOUSERSITE=1 conda run -n aptos2021`.

**제목 미정(검토 필요):** "Cannot" 단정은 공격 위험 → **balanced "can and cannot"** 또는
질문형 추천(PAPER_OUTLINE 가제는 잠정). 사용자 결정 대기.

**커밋 이력:** d0cc719(re-spine+이동) · 175cc44(v0.3 실험) · 2af0277(v0.4 split).

---

## ★★★ 서사 전환 (2026-06-10): 3개 설계문서 spine 재정렬 + `../paper/`로 이동 완료
**논문 설계·서술 문서는 `oct_llm_xai/paper/`로 분리 이동**(인덱스 `../paper/README.md`):
`../paper/{DESIGN,EXPERIMENTAL_PROTOCOL,PAPER_OUTLINE}.md` + `../paper/references.bib`(구
`code/references.bib`). 이 폴더엔 코드·데이터·결과만 남음(포인터 `PAPER_DOCS.md`).
spine을 **"4-tier 전문화 스펙트럼 + forgetting + KG 보상(H1~H8)"(구 Part A 프롬프트 서사)에서
→ "단일 이미지 CDSS의 환상 + SFT의 역설"(Part B instill 매트릭스)** 로 전면 재작성. 새 가제:
*The Single-Image Illusion: Why SFT Cannot Instill Anti-VEGF Treatment Decisions in Medical
VLMs*. 새 기여=C1(grounding⊥decision decoupling이 training-time 개입으로도 불변)·C2(단일
pre-image 정보론적 천장)·C3(SFT 역설: 정형 CoT SFT가 메타 판별력을 net-negative로 덮음).
포지셔닝=모델 결함이 아니라 **패러다임 한계의 인과 실증**. 발견 라벨은 **matrix (A)~(E)로 전
문서 통일**(DESIGN §6). references.bib에 Jain&Wallace2019·Geirhos2020·DeGrave2021·
Kirkpatrick2017 등 추가(DOI 검증; RetinaVLM npj DM DOI만 TODO).
**다음: 논문 작성 전 분석 = MATRIX_ANALYSIS §다음작업 #1(A_meta→B_meta case-level)·#2(Qwen
perceptibility)·#4(bootstrap CI/n.s.).**

### ★ v0.4 N-확대 이슈 (2026-06-11 등록)
- **v0.4 split 완료(커밋 2af0277):** `make_split_v04.py` → `sft_data/split_v04.json`
  (train128/val21/test69, Core-only Quota: dry≥2·stop≥1·poor≥1, seed0 통과).
  `gen_sft_kg_cot.py`가 이를 읽어 `sft_kg_cot.json` 재생성(252/41/136). val held-out.
- **Issue 2.2 [진행중] Per-slice biomarker eval:** CSV(`train_anno_pic.csv`) per-slice
  IRF/SRF/PED로 test-eye 470슬라이스 **독립 채점**(per-biomarker balAcc/AUC + eye-clustered
  CI). 누수방지=test-eye 슬라이스만. → Table A. 신규 `code/perslice_biomarker.py`.
- **Issue 2.3 [대기, 2.2 의존] 매트릭스 재실행:** v0.4 split로 전수 재학습+평가(tier2+tier3
  먼저, tier1c 후순위). `assemble_matrix.py`가 Table A(per-slice)+Table B(eye-level CDSS)
  두 표 emit. `verify_v3.py`에 `assert_min_inclusion` 게이트. 핵심 셀 bootstrap CI.

## ★★ v0.3 전수 재실행 = 완료 (2026-06-06). 매트릭스 `sft_data/matrix.{md,json}`. **분석 전문 `sft_data/MATRIX_ANALYSIS_v0.3.md`.**
**완료:** verify 18/18 → tier3·tier2·tier1c 17셀(tier1c C=N/A) 전부 v0.3 재실행. **tier1c QLoRA OOM은 gradient-checkpointing 무력화(.eval() 탓)였고 backbones.py에서 수정 → B/D/B_meta 정상 학습**(상세 changelog 2026-06-06).
**테스트셋(n=35):** 결정 continue21/stop14(prior 0.60) · responder good24/poor10/no_active1(binary n=34, prior 0.71) · 예후 4-class majority 0.314.
**헤드라인(3백본 일치):** (A)보이는 biomarker는 SFT로 instill(백본무관) (B)continue/stop 결정 collapse(백본무관, balAcc≈0.5; B/C→continue cont~1.0, D→stop) (C)attn-KL(arm C)도 결정/예후 불변=attention≠explanation 일반성 (D)**메타가 핵심 레버**: tier2 A_meta balAcc0.619/rGPbal0.758/resp3 0.824로 도약(백본의존; tier3·tier1c는 미동) (E)**★SFT가 그 메타 신호를 덮음**: tier2 A_meta 0.758→B_meta 0.500 역전 = v0.3 새 메시지(구조 instill≠결정능력 instill, net-negative 가능).
**⚠️ 주의:** n=35 소표본 → 개별 셀 차이 대부분 n.s.(유의성 미검정); cfFlip/faithGap은 결정 상수붕괴로 대부분 무의미; D occlusion 신호는 이슈#1에서 *지각불가 confound* 판정(tier1c D만 cfFlip1.0/faithGap+0.118로 예외=검증 필요).

## ★ 다음 작업 (우선순위; 상세=MATRIX_ANALYSIS_v0.3.md §다음작업)
1. **[최우선] A_meta>B_meta 역전(E) 정밀 분석** — tier2 case-level diff(zero-shot 적중 vs SFT 오답) + preCST-only vs 전체 메타 ablation → "SFT가 무슨 신호를 덮나" 규명. 논문 핵심 figure 후보.
2. **Qwen perceptibility_check 재실행** — tier1c D cfFlip1.0/faithGap+0.118이 진짜 유체-grounding인지 generic 픽셀반응인지(`code/perceptibility_check.py`를 tier1c로). 이슈#1을 강한 비전 백본으로.
3. **메타-천장 대비 정렬** — tier2 A_meta 0.758 vs Part A logreg responder 천장(`code/meta_ceiling.py` responder 버전) 직접 비교.
4. **유의성 표기** — 핵심 셀(tier3 B resp 0.613, tier2 A_meta 0.758)에 bootstrap CI/permutation → 표에 n.s. 명시.
5. **논문 표/서사 갱신** — matrix.md→paper.tex 결과표 교체, 서사 (A)(B)(C)+(E) 중심 재구성.

## ── (구) 다음 작업 = v0.3 데이터로 전 백본×arm 전수 재실행 [← 위에서 완료] ──
**왜:** CoT/KG/SFT를 v0.3로 **인과 재설계**(Visual→Pathophysiology→**Decision→Response**; 진단명 제거; Step4=종합반응 ΔCST+ΔVA+fluid)했고 RetinaVLM sanity-check 통과. **기존 11셀+메타 매트릭스는 전부 v0.2(구식) → 폐기. v0.3로 다시 돌려야 함.**

**돌릴 것 (전부 `code/harness.py`, oct_llm env):**
- 백본 3: tier3(RetinaVLM)·tier2(LLaVA-Med)·tier1c(Qwen3.6-27B)
- arm: A(eval만)·B(train+eval)·C(train+eval; tier1c는 **N/A**=linear-attn)·D(train+eval) + 메타 **A_meta·B_meta**
- 명령: `python3 -u harness.py --tier <t> --arm <B|C|D> --mode train [--meta]` 후 `--mode eval`. tier1c는 학습 시 자동 4-bit QLoRA.
- eval은 이제 **decision·biomarker·prognosis(ΔCST)·response(composite good/poor)** 전부 자동 채점.
- 매트릭스: `python3 assemble_matrix.py` → `sft_data/matrix.{md,json}`.
- 빠른 순서 권장: tier3·tier2(빠름) 먼저 A/B/C/D + 메타 → tier1c(느림, A/B/D + 메타; C 생략) 후순위.

**v0.3 핵심 결과(sanity, tier3):** 구조 정상 instill. 결정 balAcc≈0.45(단일 pre-image 한계), **종합 responder good/poor balAcc B=0.613>chance>A=0.409**(반응성은 부분 학습 가능). 전 백본으로 확정 필요.

**알려진 제약/주의:**
- arm C 학습: `attn_kl.py` + **bf16 autocast + grad-finite guard**(NaN 방지) 이미 반영.
- Qwen3.6-27B: **arm C는 linear-attention(gated-delta-net)이라 rollout 불가=N/A**. 학습은 4-bit QLoRA(자동). **B_meta는 긴 프롬프트로 OOM**(expandable_segments로도 일부 OOM). ⚠️ **flash-linear-attention 설치 금지**(fla 0.5.0 ↔ tf5.3.0 비호환 → Qwen import 깨짐; 설치됐으면 `site-packages/fla` 디렉토리 삭제로 복구). Qwen B_meta는 env-limited로 두거나 image max_pixels 캡으로 시도.
- LLaVA-Med: fork forward를 cache_position/logits_to_keep 흡수하도록 런타임 패치 + generate 슬라이스 수정 — `backbones.py`에 이미 반영.

## ── (구) v0.2 핸드오프 ── 직전 커밋 `561206a` (main). 이슈 #1~#4는 완료/대체됨.

## 한 줄 요약
RetinaVLM의 grounding⊥decision(decoupling)을 **training-time instill(LoRA)**로 고칠 수 있나
검증. 결론 윤곽: **보이는 concept(유체)는 instill 가능하나, 단일 pre-image로 anti-VEGF
continue/stop 결정을 instill하는 건 정보론적으로 불가** (라벨=치료반응 함수).

## 작업 디렉토리 / 환경
`/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/oct_llm_xai/paper_xai_antivegf/`
- **tier3 GPU(학습/추론):** `conda activate oct_llm` + `export LD_PRELOAD=.../envs/oct_llm/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12` + `PYTHONNOUSERSITE=1` + `CUDA_VISIBLE_DEVICES=2`. nohup은 래퍼스크립트 + `python3 -u`.
- **분석(GPU불필요):** `PYTHONNOUSERSITE=1 conda run -n aptos2021 ...`
- **예외:** `occlude.py`(cv2 inpaint)는 cv2가 `~/.local`에 있어 **`PYTHONNOUSERSITE` 없이** 실행.

## 지금까지 한 것
- **fluid_masks_v2/** — 218 eye 전체, **12×12** fluid 마스크(Claude-vision subagent 16 병렬, GT mismatch 0) + 메타데이터 + continue/stop 라벨 + `occluded/`·`occluded_negctrl/`(cv2 inpaint). source-of-truth = `masks_12x12.{json,npz}`, `metadata_v2.json` (렌더 이미지는 .gitignore, 스크립트로 재생성).
- **KG v2** `code/antivegf_guideline_kg_v2.json` — 다단계 Node-Edge-Node(visual→pathophysiology→prognosis(ΔCST)→decision). prognosis는 실측 ΔCST를 **예측 타깃**으로(입력 힌트 아님).
- **SFT** `sft_data/sft_kg_cot.json` — 429행(factual 218 + counterfactual 211). divergent 77 eye 정직 렌더. 생성기 `code/gen_sft_kg_cot.py`.
- **LoRA 4-arm**: `code/train_lora_b.py`(arm B/D), `code/eval_lora_b.py`, 스캐폴드 `lora_ablation.py`/`eval_ablation.py`. 어댑터는 .gitignore(재학습 ~30s).
- **B(L_LM), D(L_LM+counterfactual) 학습+eval 완료** (RetinaVLM tier3). 결과 `sft_data/eval_{A_baseline,B_sft_text,D_counterfactual}.json`.

## 핵심 결과 (test 35)
| 지표 | A base | B (SFT) | D (CF) |
|---|---|---|---|
| continue_rate | 0.80 | 1.00 | 0.00 |
| biomarker-node acc | 0.45 | **0.81** | 0.48 |
| prognosis-node acc (vs ΔCST) | 0.06 | 0.26 | 0.14 |
| (majority) | — | 0.31 | — |
| CF flip-rate | 0.11 | 0.0 | None* |
\* D는 clean도 전부 stop이라 flip 분모=0.
- prognosis-node ≤ majority → **단일 이미지 정보론적 한계 실증**.
- decision은 라벨 prior로 collapse(B→continue, D→stop) → **결정 grounding 불가**.
- biomarker concept은 학습됨(0.45→0.81) → 보이는 것은 instill 가능.

## 다음 작업 (우선순위 = 이슈 #1~#4)
1. **(#1) occlusion 지각가능성 체크 — ✅ 완료 2026-06-04 (`code/perceptibility_check.py`, `sft_data/perceptibility_check.json`).** 결론: **occlusion은 지각 불가 → arm D counterfactual 신호 void → D all-stop collapse는 confound(grounding 결과 아님).** 증거: SFT B는 clean vs occluded 출력이 28/34(82%) **byte-identical**, fluid bm 드롭 0/34, 결정 flip 0/34. base A는 출력은 변하나 fluid-비특이적(gap_bm 0.0, gap_dec 0.074). → 핸드오프가 예측한 최대 confound 확증. 자세히 changelog 2026-06-04.
2. **(#4, 다음 차례) counterfactual 재설계** — #1이 원인을 짚음: 미세 inpaint를 192×192 grayscale ResNet이 무시 + SFT가 text-prior화. 재설계: occlusion **강화**(더 큰 영역/zero-out/대비 파괴) → **반드시 perceptibility_check로 지각가능성 먼저 재검증**(gap_bm 또는 gap_dec가 유의해질 때까지) → 그 뒤에야 CF row class balance + arm D 재학습. 지각 안 되면 재학습 무의미.
3. **(#2) arm C(attention-KL) 완주 — ✅ 완료 2026-06-04.** `attn_kl.py`(미분가능 torch rollout + 12×12→6×6 average-pool + forward KL), `train_lora_b.py --arm C`, `lora_ablation.py` 배선. 학습 KL 2.09→0.9(attention이 유체로 끌려옴=FER↑ proxy), eval `eval_C_attn_guide.json`: biomarker 0.84↗ 이지만 **continue_rate 1.0 collapse·prognosis 0.257(≤maj)·cf_flip 0** → **attention 정렬해도 결정/예후 grounding 불변**(예측대로 attention≠explanation 확증). 상세 changelog 2026-06-04.
4. **(#3, 다음 차례) 다중 백본 확장** — `attn_kl.py`는 이미 backend-agnostic(attention텐서+image slice+grid만 받음). LLaVA-Med(tier2)·Qwen3.6(tier1c)에 A/B/C **3×3 매트릭스** 하네스 구축(HF LlavaNext/Qwen은 `output_attentions=True`+peft 표준, image-token grid·slice만 백본별로 지정). 헤드라인: FER↑ but Decision/Prognosis collapse의 백본 일반성. C의 eval-time rollout FER(B 대비↑) 직접측정도 이 매트릭스에 포함.

## 먼저 읽을 것
`sft_data/README.md`, `code/changelog.md`(맨 끝 2026-06-03 항목), eval JSON 3개,
`code/{train_lora_b.py, eval_lora_b.py, gen_sft_kg_cot.py, occlude.py}`.

## git
- branch `main`, 직전 커밋 `561206a`. origin=github(jmin8008/OCT_LLM_XAI). gitlab remote는 프로젝트 moved(405) → 새 경로로 갱신 전엔 push 불가.
- 무거운 산출물(가중치/렌더 이미지/heatmap)은 `.gitignore` — 스크립트로 재생성.
