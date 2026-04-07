# KAD Knowledge Graph 학습을 위한 OCT / Fundus 데이터셋

> 모달리티(OCT / Fundus) × 카테고리(질병·병변 / 바이오마커 / 해부학 구조) 조합으로 총 6개 데이터셋을 선정·수집.

---

## Quick Start: 데이터셋 수집

개별 스크립트 또는 일괄 스크립트로 다운로드할 수 있습니다.

```bash
# 개별 다운로드
python collect/01_kermany_oct.py
python collect/02_umn_oct_fluid.py
python collect/03_duke_sd_oct.py
python collect/04_aptos_2019.py
python collect/05_stare.py
python collect/06_rim_one_dl.py

# 일괄 다운로드
python collect/download_datasets.py              # 전체
python collect/download_datasets.py --only oct   # OCT만
python collect/download_datasets.py --only fundus # Fundus만
```

### 사전 요구사항

```bash
pip install kaggle
# ~/.kaggle/kaggle.json 에 Kaggle API 토큰 설정
```

### 저장 경로

```
data/
├── OCT2017/           # 1. Kermany OCT      (24GB,  84,495장)
├── UMN_OCT_Fluid/     # 2. UMN OCT Fluid    (597MB, 600장)
├── Duke_SD_OCT/       # 3. Duke SD-OCT      (389MB, 110장)
├── APTOS_2019/        # 4. APTOS 2019       (14GB,  3,662장)
├── STARE/             # 5. STARE            (42MB,  397장)
└── RIM_ONE_DL/        # 6. RIM-ONE DL       (221MB, 485장)
```

---

## 데이터셋 요약

| # | 모달리티 | 카테고리 | 데이터셋 | 데이터 수 | 주요 타겟 | 다운로드 방식 |
|---|---------|---------|---------|----------|----------|-------------|
| 1 | OCT | 질병/병변 | Kermany OCT | 84,495장 | CNV, DME, DRUSEN 등 4개 질환 | Kaggle API |
| 2 | OCT | 바이오마커 | UMN OCT Fluid | 600장 | IRF, SRF, PED 유체 바이오마커 | Kaggle API |
| 3 | OCT | 해부학 구조 | Duke SD-OCT | 110장 | 8개 망막 층(Layer) 경계 및 유체 | Kaggle API |
| 4 | Fundus | 질병/병변 | APTOS 2019 | 3,662장 | 당뇨망막병증(DR) 중증도 0~4단계 | Kaggle API |
| 5 | Fundus | 바이오마커 | STARE | 397장 | 신생혈관(NV) 포함 14개 병변 레이블 | wget (원본 사이트) |
| 6 | Fundus | 해부학 구조 | RIM-ONE DL | 485장 | 시신경 유두(OD) 및 시신경 잔(OC) | Kaggle API |

---

## 1. OCT — 질병/병변: Kermany OCT

- **규모**: 84,495 OCT B-scan (train 83,484 + test 1,000)
- **레이블**: CNV, DME, Drusen, Normal (4 classes)
- **출처**: Cell (2018), Kermany et al.
- **Kaggle**: `paultimothymooney/kermany2018`
- **저장 경로**: `data/OCT2017/`
- **스크립트**: `collect/01_kermany_oct.py`

---

## 2. OCT — 바이오마커: UMN OCT Fluid

- **규모**: 600장
- **레이블**: IRF (intraretinal fluid), SRF (subretinal fluid), PED (pigment epithelial detachment) 유체 바이오마커
- **원본 출처**: University of Minnesota — `conservancy.umn.edu/handle/11299/215706`
- **Kaggle 미러**: `zeeshanahmed13/intraretinal-cystoid-fluid`
- **저장 경로**: `data/UMN_OCT_Fluid/`
- **스크립트**: `collect/02_umn_oct_fluid.py`

---

## 3. OCT — 해부학 구조: Duke SD-OCT (Chiu 2015)

- **규모**: 110 annotated OCT B-scans (severe DME patients)
- **레이블**: 8개 망막 레이어 경계 (ILM, NFL-GCL, IPL-INL, INL-OPL, OPL-ONL, ELM, IS-OS junction, OS-RPE)
- **출처**: BOE (2014), Chiu et al., Duke University
- **Kaggle**: `paultimothymooney/chiu-2015`
- **저장 경로**: `data/Duke_SD_OCT/`
- **스크립트**: `collect/03_duke_sd_oct.py`

---

## 4. Fundus — 질병/병변: APTOS 2019

- **규모**: 3,662 fundus images
- **레이블**: 당뇨망막병증(DR) 중증도 5단계 (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR)
- **원본 출처**: Kaggle Competition `aptos2019-blindness-detection`
- **Kaggle 미러**: `sovitrath/diabetic-retinopathy-224x224-2019-data` (224×224 리사이즈 버전)
- **저장 경로**: `data/APTOS_2019/`
- **스크립트**: `collect/04_aptos_2019.py`

---

## 5. Fundus — 바이오마커: STARE

- **규모**: 397 fundus images
- **레이블**: 신생혈관(NV) 포함 14개 병변 레이블, 혈관 세그멘테이션 (2명 annotator: AH, VK)
- **출처**: Clemson University, Hoover et al.
- **다운로드**: `cecas.clemson.edu/~ahoover/stare/` (wget 직접 다운로드)
- **저장 경로**: `data/STARE/`
- **스크립트**: `collect/05_stare.py`
- **참고**: 이미지는 `.ppm.gz` 형태로 제공, 스크립트에서 자동 압축 해제

---

## 6. Fundus — 해부학 구조: RIM-ONE DL

- **규모**: 485 fundus images
- **레이블**: 시신경 유두(Optic Disc) 및 시신경 잔(Optic Cup) 세그멘테이션, 녹내장 분류
- **원본 출처**: `kaggle.com/datasets/tavoosi/rim-one-dl`
- **Kaggle 미러**: `orvile/rim-one-retinal-dataset-for-assessing-glaucoma`
- **저장 경로**: `data/RIM_ONE_DL/`
- **스크립트**: `collect/06_rim_one_dl.py`

---

## Sources

- Kermany OCT: https://www.kaggle.com/datasets/paultimothymooney/kermany2018
- UMN OCT Fluid: https://conservancy.umn.edu/handle/11299/215706
- Duke SD-OCT (Chiu 2015): https://www.kaggle.com/datasets/paultimothymooney/chiu-2015
- APTOS 2019: https://www.kaggle.com/c/aptos2019-blindness-detection
- STARE: http://cecas.clemson.edu/~ahoover/stare/
- RIM-ONE DL: https://www.kaggle.com/datasets/tavoosi/rim-one-dl
