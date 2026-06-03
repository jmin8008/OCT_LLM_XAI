# Fluid Masks — Claude-Vision Coarse Localization (6×6)

> **목적:** Stage 1/2 의 FER(Fluid-Energy Ratio) 계산에 쓰이던 **상수 가짜 마스크**
> (`fluid_mask[4:8,2:6]=1`, 모든 eye 동일)를 **eye별 실제 유체 위치 마스크**로 교체.
>
> **제작:** Claude (Sonnet/Opus, vision) 단일 평가자, 2026-06-02. SAM·CNN 미사용 —
> 사용자 지시("마스킹 데이터를 너가 직접 판단해서 만들어")에 따라 멀티모달 모델의
> 시각 판단으로 직접 주석.

---

## 무엇이 문제였나

기존 FER=0.028(“모델이 유체를 거의 안 본다 → text-bias”)은 **마스크 위치가 틀려서** 나온 수치였다.
상수 마스크는 8×8 격자의 하단-중앙 4×4(`[4:8,2:6]`)에 고정 — 이 영역은 OCT에서 보통
**맥락막/배경**이며, 유체(망막 내·하)가 있는 위치가 아니다. 게다가 eye마다 유체 위치가
크게 다른데(5L 중단, 115R 상단-좌, DME는 망막 띠 내부) 모두 같은 박스를 썼다.

→ FER은 "유체에 쏠린 attention"이 아니라 "하단-중앙에 쏠린 attention"을 재던 셈.
→ FER↔KG 무상관(r=0.057)도 이 아티팩트의 영향을 받았을 가능성이 큼.

---

## 제작 방법

1. `data.representative_pre_bscan()` 로 VLM이 실제 입력받는 pre-injection 중심 B-scan
   (macular crop, 210×596) 을 36 test eye에 대해 렌더 (325L은 이미지 없음 → 제외, 35 eye).
2. 2% / 99% percentile 대비 스트레칭 + 2× 업스케일 + **6×6 격자**(saliency `map_shape`와 동일)
   오버레이, 각 셀에 `행렬` id(`00`..`55`) 라벨 → `annot/<eye>.png`.
3. Claude가 각 영상을 보고 유체(SRF 암부 돔 / IRF 낭종성 암부 / PED 융기) 가 포함된
   셀을 표기. 라벨 바이오마커(IRF/SRF/PED/HRF)로 유체 유무·종류를 교차 확인.
4. `masks_6x6.json` / `masks_6x6.npz` 저장. `overlay/<eye>.png`, `ALL_overlays_montage.png`
   로 시각 검증.

판독 규칙(일관성):
- **SRF** = 분리된 신경망막(밝은 띠) 아래의 암부 돔/삼각 공간.
- **IRF** = 밝은 망막 띠 *내부*의 둥근 암부(낭종). DME에서 지배적.
- **PED** = RPE 융기 돔. 융기 정점부 셀 포함.
- HRF 단독(유체 없음) → 마스크 비움. 유체 라벨이 모두 0인 eye(209R) → 비움(음성 대조).

---

## 산출물

| 파일 | 내용 |
|------|------|
| `masks_6x6.json` / `.npz` | eye_id → 6×6 binary mask (35 eye) |
| `manifest.json` | eye별 dx / has_fluid / biomarkers / img_size |
| `annot/` | 6×6 격자+라벨 주석용 영상 |
| `overlay/` | 마스크(빨강) 오버레이 — 검증용 |
| `ALL_overlays_montage.png` | 35 eye 한 장 요약 |

- fluid-present 마스크: **34 / 35** eye (209R 비움)
- 평균 유체 셀 수: **5.3 / 36** cells per eye

---

## 최종 결과 — 36 eye 전체 (GPU rollout 재실행)

`gen_rollout_realmask.py` 로 36 eye 전부의 raw 6×6 map을 재생성(Z1, query=마지막 토큰) →
`code/xai_e3_tier3_rollout_realmask.json`. 실제 마스크로 FER 계산 + KG-align 상관:

```
FER fluid eyes:  const=0.018  →  REAL=0.210   (11.4×↑, n=34)
Spearman r(FER_real, KG-align Z1) = -0.051  p=0.77   (여전히 무상관)
  FER|aligned=0.198 (n=20)  vs  FER|misaligned=0.212 (n=15)   차이 없음
```

**두 결론:** ① FER 11× ↑ → 모델은 유체를 *본다* ("안 본다"는 마스크 아티팩트였음).
② 그럼에도 FER↔KG-align 무상관 → *보는 정도*가 *맞게 말하는지*와 분리(decoupling).
가설 재정의: instill = "보게 만들기"가 아니라 grounding↔결정의 **인과적 결합**.
한계: FER이 prompt 마지막 토큰 기준(생성-시점 아님) — 생성-토큰 attention은 Stage 1에서 별도 캡처.

---

## 예비 결과 — FER이 ~13× 상승 (8 eye, 기존 저장 map)

`xai_e3_tier3_maps.json` 에 raw 6×6 attention map이 저장된 8 eye에 대해 재계산(검증용):

| eye | dx | FER (상수 가짜 마스크) | FER (실제 마스크) |
|-----|-----|------:|------:|
| 5L | PCV | 0.0153 | 0.0575 |
| 9L | PCV | 0.0135 | 0.1968 |
| 20R | PCV | 0.0107 | 0.1754 |
| 45L | CNVM | 0.0165 | 0.1911 |
| 69R | CNVM | 0.0236 | 0.2432 |
| 110R | CNVM | 0.0176 | 0.1970 |
| 115R | CNVM | 0.0179 | 0.4220 |
| 128R | CNVM | 0.0127 | 0.1480 |
| **평균** | | **0.0160** | **0.203** |

→ RetinaVLM의 attention은 **실제 유체 영역에 상당히 쏠려 있다** (평균 FER≈0.20, 최대 0.42).
"모델이 유체를 안 본다"는 기존 결론은 **마스크 아티팩트**였다.

---

## 한계 (정직한 기록)

- **단일 평가자 = Claude vision.** 임상의 검증 아님. 6×6 coarse(셀≈35×99px) — 픽셀 경계 부정확.
- 일부 eye(예: 115R)는 ~1셀 좌측 편이 관찰됨(overlay 검증). 정량 분석엔 허용 수준이나
  논문용으로는 임상의 subset 재주석 권장.
- FER 재계산은 raw map이 있는 **8 eye 한정**. 36 eye 전체 FER↔KG 상관 재검증은
  rollout 재실행(raw map 저장) 필요 → GPU 단계.
- B-scan은 eye당 대표 1장(중심). 3D 볼륨 전체 유체 아님.

## 다음 단계

1. tier3 rollout 재실행 시 `map_values`(6×6)를 **36 eye 전부** 저장하도록 수정.
2. 실제 마스크로 36 eye FER 재계산 → FER↔KG-align Spearman r 재검증.
3. (가설 검정) Stage 2 attention steering 전후 KG-align 비교.
