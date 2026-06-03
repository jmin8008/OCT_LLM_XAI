# Fluid Masks v2 — Full-cohort 12×12 grounding + decision dataset

> 작성 2026-06-03. v1(`../fluid_masks/`, 35 test eye만, 6×6)을 **전체 218 eye, 12×12,
> 풍부한 메타데이터 + 결정 라벨**로 확장. Stage 3(training-time instill) 학습/평가용.

## 왜 v2

- v1은 **test 35 eye만** 마스크가 있어 학습에 못 씀(train 183 eye에 grounding 라벨 부재).
- 6×6은 너무 coarse. attention-KL을 폐기(→ "attention is not explanation")하면서
  마스크가 토큰격자(6×6)에 묶일 이유가 사라져 **12×12로 상향** 가능.
- counterfactual occlusion / concept-bottleneck 등 어느 Stage 3 메커니즘이든 쓸 수 있도록
  픽셀이 아닌 셀 단위지만 4× 조밀한 마스크 + 전 cohort + 결정 라벨을 한 번에 제공.

## 구성

| 파일/폴더 | 내용 |
|---|---|
| `metadata_v2.json` | 218 eye 전체 메타데이터 (아래 스키마) |
| `masks_12x12.json` | eye_id → 12×12 binary list |
| `masks_12x12.npz` | 동일 (np.load, key=eye_id, uint8 12×12) |
| `clean/<eye>.png` | 대비보정+업스케일 OCT 크롭 (360px 폭) |
| `grid/<eye>.png` | 12×12 격자+축라벨 (주석 캔버스) |
| `overlay/<eye>.png` | 마스크(빨강) 오버레이 — 시각 QC |
| `ALL_overlays_montage_v2.png` | 218 eye 한 장 요약 |
| `annotation_manifest.json` | 주석 work-list (GT biomarker 제약) |
| `batches/`, `annot_parts/` | 주석 분할/원본 (16 배치) |
| `ANNOTATION_GUIDE.md` | 주석 규칙(판독 정의·스키마) |

### metadata_v2.json 스키마 (eye당)
```
eye_id, split(train/test), continue_injection(LABEL 0/1),
diagnosis, drug, age, gender,
pre_va, pre_cst, post_va, post_cst, delta_cst, delta_va,
biomarkers{IRF,SRF,PED,HRF}, fluid_types[], has_fluid,
fluid_cell_count, mask_confidence(high/med/low), mask_types_seen[], mask_note,
img_size, grid_n(12)
```

## 통계

- **커버리지:** 218 eye = train 183 + test 35 (test split·seed=42는 v1과 동일).
- **결정 라벨:** continue 139 / stop 79 (전체). train 118/65, test 21/14.
- **진단:** DME 137 / CNVM 67 / PCV 14.
- **마스크:** 평균 10.2 cell/eye (144 중), median 10, max 26. 범람·좌표오류 0.
- **GT 교차검증:** has_fluid(GT biomarker) vs 마스크 **mismatch 0** —
  유체 음성 2 eye(114R, 209R)는 빈 마스크(음성대조), 나머지 216 eye 전부 ≥1 cell.
- **주석 신뢰도:** high 37 / med 142 / low 39 (low = 얇거나 노이즈 큰 스캔).

## 제작 방법 (재현)

1. `code/gen_dataset_v2.py` — 218 eye 렌더(clean/grid) + metadata + manifest.
2. `code/` 없이 16배치로 분할 → **Claude vision subagent 16개** 병렬 주석
   (`ANNOTATION_GUIDE.md` 규칙, GT fluid_types로 타입 제약). → `annot_parts/`.
3. `code/assemble_v2.py` — 마스크 통합(json/npz) + 오버레이/몽타주 + GT 교차검증 + 메타 병합.

판독 정의(요약): SRF=밝은 망막 band 바로 아래 암부 돔 / IRF=band 내부 둥근 낭종 /
PED=RPE 융기 돔 하부. HRF·drusen 단독은 유체 아님(미표기).

## 한계 (정직한 기록)

- **단일 평가자 = Claude vision (멀티 subagent).** 임상의 검증 아님. v1과 동일 방법론,
  다만 12×12로 조밀·전 cohort. 12×12는 6×6보다 셀당 면적이 작아 **개별 셀 노이즈는 증가**
  (false precision 주의) — occlusion 영역 정밀화엔 유리, 정밀 경계 주장엔 부적합.
- B-scan은 eye당 대표 1장(중심 pre-injection). 3D 볼륨 전체 아님.
- `low` 신뢰도 39 eye는 학습 시 가중치↓ 또는 임상의 재주석 후보.
- 결정 라벨(continue/stop)은 APTOS-2021 전용 → 218이 라벨 데이터 상한.
  추가 grounding 볼륨이 필요하면 UMN/OCT5k/Duke의 **전문가 픽셀 마스크**를 별도
  보조 corpus로(단, 결정 라벨 없음) 결합 가능.
