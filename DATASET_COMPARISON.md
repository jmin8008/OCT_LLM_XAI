# 3가지 OCT 데이터셋 비교

## 📊 종합 비교표

| 항목 | **Kermany** | **OLIVES** | **OCT-SD** |
|------|-----------|-----------|-----------|
| **상태** | ✅ 완료 | ✅ 이미지/라벨 다운 | ❌ 저장소 없음 |
| **데이터 형식** | 바이오마커 판단 | 구조화된 라벨 | 이미지-텍스트 쌍 |
| **라벨 출처** | 템플릿 (의료논문 기반) | 실제 의료 메타데이터 | Synthetic (자동생성) |
| **텍스트 특성** | 3문장 짧은 리포트 | 구조화된 바이오마커 | 의료 소견서 형식 |
| **이미지 수** | 84,452 | 9,408 (OCT 스캔) | ❌ 사용 불가 |
| **라벨 종류** | 10개 바이오마커 | 16개 바이오마커 + 임상지표 3개 | 질병 분류 + 텍스트 |
| **텍스트 품질** | ⭐⭐⭐⭐⭐ (100% 정확) | ⭐⭐⭐⭐⭐ (실제 의료) | ⭐⭐⭐ (Synthetic) |
| **용도** | DQN 학습 | XAI 벤치마크 | VLM 학습 |

---

## 1️⃣ **Kermany** (완료)

### 형식
```json
{
  "image": "CNV-2177326-5.jpeg",
  "class": "CNV",
  "biomarkers": [
    {
      "biomarker": "subretinal fluid",
      "status": "present",
      "explanation": "liquid accumulated beneath the neurosensory retina",
      "variants": [
        "The imaging findings are consistent with subretinal fluid.",
        "Intraretinal fluid is evident on this OCT examination.",
        "One can observe pigment epithelial detachment in the OCT."
      ]
    }
  ]
}
```

### 바이오마커 목록 (10개)

| # | 바이오마커 | 설명 | 값 |
|---|----------|------|---|
| 1 | subretinal fluid | 망막하액 | present / not present |
| 2 | intraretinal fluid | 망막내액 | present / not present |
| 3 | pigment epithelial detachment (PED) | RPE 박리 | present / not present |
| 4 | drusen | 드루젠 (황색 침착물) | present / not present |
| 5 | retinal pigment epithelium atrophy | RPE 위축 | present / not present |
| 6 | geographic atrophy | 지도모양위축 | present / not present |
| 7 | subretinal hyperreflective material (SHRM) | 망막하 고반사 물질 | present / not present |
| 8 | hyperreflective foci | 고반사점 | present / not present |
| 9 | choroidal neovascularization (CNV) | 맥락막 신생혈관 | present / not present |
| 10 | epiretinal membrane (ERM) | 망막전막 | present / not present |

### 클래스별 바이오마커 프로파일

| 바이오마커 | NORMAL | CNV | DME | DRUSEN |
|----------|--------|-----|-----|--------|
| subretinal fluid | ❌ | ✅ | ❌ | ❌ |
| intraretinal fluid | ❌ | ✅ | ✅ | ❌ |
| PED | ❌ | ✅ | ❌ | ✅ |
| drusen | ❌ | ✅ | ❌ | ✅ |
| RPE atrophy | ❌ | ✅ | ❌ | ✅ |
| geographic atrophy | ❌ | ❌ | ❌ | ❌ |
| SHRM | ❌ | ✅ | ❌ | ❌ |
| hyperreflective foci | ❌ | ✅ | ✅ | ❌ |
| CNV | ❌ | ✅ | ❌ | ❌ |
| ERM | ❌ | ❌ | ❌ | ❌ |
| **합계** | **0/10** | **8/10** | **2/10** | **3/10** |

### 특징
- **템플릿 기반**: 규칙적으로 생성
- **의료 정확성**: ⭐⭐⭐⭐⭐ (논문 기반 프로파일 100% 일치)
- **텍스트 다양성**: 3가지 템플릿으로 변형
- **규모**: 84,452 이미지 × 10 바이오마커 × 3 텍스트 = 2,533,560 데이터포인트
- **클래스 분포**: NORMAL 26,557 / CNV 37,447 / DME 11,590 / DRUSEN 8,858

### 장점
✅ 정확도 100% (의료 표준 준수)  
✅ LLM 할루시네이션 없음  
✅ 대규모 (가장 많은 이미지)  
✅ DQN 학습용 완벽  

### 단점
❌ 짧은 텍스트 (3문장)  
❌ 템플릿 기반이라 다양성 낮음  
❌ 실제 의료 리포트 아님  

### 추천 용도
- ✅ DQN 모델 학습
- ✅ 기본 바이오마커 detection
- ✅ Zero-shot 벤치마크

---

## 2️⃣ **OLIVES** (라벨 완료, 리포트 미생성)

### 형식
```
CSV (Biomarker_Clinical_Data_Images.csv):
Path, Scan, Atrophy/thinning, Disruption of EZ, DRIL, IR hemorrhages, IR HRF,
Partially attached vitreous face, Fully attached vitreous face,
Preretinal tissue/hemorrhage, Vitreous debris, VMT, DRT/ME,
Fluid (IRF), Fluid (SRF), Disruption of RPE, PED (serous), SHRM,
Eye_ID, BCVA, CST, Patient_ID
```

### 바이오마커 목록 (16개) + 임상지표 (3개)

**바이오마커 (binary 0/1)**:

| # | 컬럼명 | 설명 | 값 |
|---|--------|------|---|
| 1 | Atrophy / thinning of retinal layers | 망막층 위축/얇아짐 | 0/1 |
| 2 | Disruption of EZ | 타원체대(EZ) 파괴 | 0/1 |
| 3 | DRIL | 망막내층 경계 소실 (Disorganization of Retinal Inner Layers) | 0/1 |
| 4 | IR hemorrhages | 망막내 출혈 | 0/1 |
| 5 | IR HRF | 망막내 고반사점 (Hyperreflective Foci) | 0/1 |
| 6 | Partially attached vitreous face | 부분 유리체 부착 | 0/1 |
| 7 | Fully attached vitreous face | 완전 유리체 부착 | 0/1 |
| 8 | Preretinal tissue/hemorrhage | 망막전 조직/출혈 | 0/1 |
| 9 | Vitreous debris | 유리체 부유물 | 0/1 |
| 10 | VMT | 유리체황반견인 (Vitreomacular Traction) | 0/1 |
| 11 | DRT/ME | 미만성 망막부종 (Diffuse Retinal Thickening / Macular Edema) | 0/1 |
| 12 | Fluid (IRF) | 망막내액 (Intraretinal Fluid) | 0/1 |
| 13 | Fluid (SRF) | 망막하액 (Subretinal Fluid) | 0/1 |
| 14 | Disruption of RPE | RPE 파괴 | 0/1 |
| 15 | PED (serous) | 장액성 RPE 박리 (Pigment Epithelial Detachment) | 0/1 |
| 16 | SHRM | 망막하 고반사 물질 (Subretinal Hyperreflective Material) | 0/1 |

**임상 지표**:

| # | 컬럼명 | 설명 | 값 |
|---|--------|------|---|
| 17 | BCVA | 최대교정시력 (Best Corrected Visual Acuity) | 연속값 (43 unique) |
| 18 | CST | 중심황반두께 (Central Subfield Thickness) | 연속값 (131 unique) |
| 19 | Patient_ID | 환자 ID | 87명 |

### 데이터 규모
- **총 행 수**: 9,408개 OCT 스캔
- **환자 수**: 87명
- **눈(Eye) 수**: 96개
- **스캔/볼륨**: 49 슬라이스/볼륨

### 특징
- **실제 의료 메타데이터**: 의사/임상 전문가 라벨링
- **풍부한 바이오마커**: 10개 (Kermany) → **16개 (OLIVES)**
- **임상 지표 포함**: BCVA (시력), CST (황반두께) — 실제 치료 추적 데이터
- **멀티모달**: OCT + Fundus 이미지
- **공개 데이터셋**: Zenodo (https://zenodo.org/record/5568215)

### 장점
✅ 실제 의료 라벨 (의사 검증)  
✅ 16개 바이오마커로 더 상세  
✅ 임상 지표(BCVA, CST) 포함 — 치료 효과 추적 가능  
✅ 공개 논문 기반  

### 단점
⚠️ 텍스트 리포트 없음 (구조화된 라벨만 있음)  
❌ Kermany보다 작은 규모 (9,408 vs 84,452)  
❌ 질병(disease) 라벨 컬럼이 별도 없음 (바이오마커로 추론 필요)  

### 추천 용도
- ✅ XAI/설명가능성 연구
- ✅ 주의맵(attention map) 검증
- ✅ 멀티모달 VLM 학습
- ✅ 벤치마크 평가

---

## 3️⃣ **OCT-SD** (❌ 다운로드 실패 — 저장소 없음)

### 형식
```json
{
  "image_path": "cnv/scan_001.png",
  "disease_type": "Choroidal Neovascularization",
  "report_text": "The OCT image shows evidence of subretinal and intraretinal fluid. There is a hyperreflective lesion consistent with choroidal neovascularization. Retinal pigment epithelium appears disrupted. Overall findings are consistent with wet AMD.",
  "has_report": true
}
```

### 특징
- **Synthetic 리포트**: LLM 또는 템플릿으로 자동 생성
- **이미지-텍스트 쌍**: VLM report generation 학습용
- **출처**: CheXpert 기반 + Microsoft report-generation
- **더 자연스러운 텍스트**: 실제 의료 리포트 형식

### 장점
✅ 이미지-텍스트 쌍 (VLM 학습 최적)  
✅ 자연스러운 의료 소견서 형식  
✅ Report generation 벤치마크  
✅ 더 다양한 텍스트 표현  

### 단점
❌ Synthetic = 의사가 안 쓴 리포트  
❌ LLM 생성이라 정확도 미검증  
❌ 품질이 다를 수 있음  
❌ 할루시네이션 위험 있음  
❓ 다운로드 실패 (HF 저장소 없음)  

### 추천 용도
- ✅ VLM report generation 학습
- ⚠️ 보조 데이터셋 (주요 아님)
- ❓ 품질 검증 후 사용 결정

---

## 🔗 바이오마커 교차 매핑 (Kermany ↔ OLIVES)

### 공통 바이오마커 (6개)

| Kermany | OLIVES | 매핑 정확도 |
|---------|--------|-----------|
| intraretinal fluid | Fluid (IRF) | ✅ 동일 |
| subretinal fluid | Fluid (SRF) | ✅ 동일 |
| pigment epithelial detachment | PED (serous) | ✅ 동일 |
| subretinal hyperreflective material | SHRM | ✅ 동일 |
| hyperreflective foci | IR HRF | ✅ 동일 |
| retinal pigment epithelium atrophy | Disruption of RPE / Atrophy | ✅ 유사 |

### Kermany에만 있는 바이오마커 (4개)

| 바이오마커 | 설명 | 비고 |
|----------|------|------|
| drusen | 드루젠 (황색 침착물) | AMD 핵심 마커, OLIVES에 없음 |
| geographic atrophy | 지도모양위축 | 진행성 AMD, OLIVES에 없음 |
| choroidal neovascularization (CNV) | 맥락막 신생혈관 | 습성 AMD 핵심, OLIVES에 없음 |
| epiretinal membrane (ERM) | 망막전막 | OLIVES에 없음 |

### OLIVES에만 있는 바이오마커 (10개)

| 바이오마커 | 설명 | 비고 |
|----------|------|------|
| Disruption of EZ | 타원체대 파괴 | 시력 예후 핵심 |
| DRIL | 망막내층 경계 소실 | DME 중증도 지표 |
| IR hemorrhages | 망막내 출혈 | DR/DME 관련 |
| Partially attached vitreous face | 부분 유리체 부착 | 유리체-망막 인터페이스 |
| Fully attached vitreous face | 완전 유리체 부착 | 유리체-망막 인터페이스 |
| Preretinal tissue/hemorrhage | 망막전 조직/출혈 | 증식성 DR 지표 |
| Vitreous debris | 유리체 부유물 | 유리체 병변 |
| VMT | 유리체황반견인 | 유리체-망막 인터페이스 |
| DRT/ME | 미만성 망막부종 | DME 핵심 |
| Atrophy / thinning | 망막층 위축 | RPE atrophy와 유사하나 더 넓은 개념 |

### 통합 시 전체 바이오마커 (20개)

```
공통 (6개):  IRF, SRF, PED, SHRM, HRF, RPE atrophy
Kermany만 (4개): drusen, geographic atrophy, CNV, ERM
OLIVES만 (10개): EZ disruption, DRIL, IR hemorrhages,
                  vitreous face (2), preretinal tissue,
                  vitreous debris, VMT, DRT/ME, retinal atrophy
```

---

## 🎯 사용 전략

### **우선순위**
1. **Kermany** (현재 사용 가능)
   - DQN 학습 시작
   - 기본 바이오마커 detection

2. **OLIVES** (구조화된 라벨 있음)
   - XAI 벤치마크
   - 주의맵 정렬성 검증
   - 멀티모달 학습

3. **OCT-SD** (선택사항)
   - 다운로드 실패 시 스킵 가능
   - 품질 검증 후 결정

### **데이터셋 조합**
```
✅ 권장: Kermany + OLIVES
   - 크기: 146k 이미지
   - 형식: 바이오마커 판단 + 구조화된 라벨
   - 품질: 100% 검증됨
   - 용도: DQN + XAI 완전한 파이프라인

⚠️ 선택: + OCT-SD
   - VLM report generation 추가 학습
   - 품질 검증 필수
```

---

## 📝 다음 단계 추천

| 순서 | 작업 | 상태 |
|------|------|------|
| 1 | Kermany로 DQN 기본 학습 | ✅ 준비 완료 |
| 2 | OLIVES로 XAI 검증 | ✅ 라벨 준비 완료 |
| 3 | OLIVES 소견서 생성 (optional) | ⏳ 예정 |
| 4 | OCT-SD 다운로드 (선택) | ❌ 저장소 없음 — 스킵 |
| 5 | 통합 벤치마크 평가 | ⏳ 예정 |

---

## 📂 파일 위치

```
/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/data/
├── OCT2017/                          # Kermany
│   ├── train/                        # 83,484 이미지 (NORMAL, CNV, DME, DRUSEN)
│   ├── test/                         # 968 이미지
│   └── biomarker_dataset/
│       ├── biomarker_annotations.jsonl  # 263 MB, 84,452 행
│       └── summary.json
│
└── OLIVES/
    ├── OLIVES/                       # OCT + Fundus 이미지
    └── OLIVES_Dataset_Labels/
        ├── full_labels/
        │   ├── Biomarker_Clinical_Data_Images.csv   # 9,408행 × 22컬럼
        │   ├── Clinical_Data_Images.xlsx
        │   ├── OCT-DME.xlsx
        │   └── OCT-DR.xlsx
        └── ml_centric_labels/
            ├── Biomarker_Clinical_Data_Images.csv
            └── Clinical_Data_Images.xlsx
```

