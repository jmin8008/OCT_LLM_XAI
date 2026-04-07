# KAD-Inspired OCT Training Datasets 가이드

**작성일**: 2026-04-03  
**목표**: UMLS KG + DQN 기반 zero-shot OCT 진단 모델 구축  
**데이터 전략**: 세 가지 보수 OCT 소견서 데이터 파이프라인

---

## 📋 개요

원래 목표: MIMIC-CXR처럼 **"전문의가 직접 타이핑한 대규모 자유 텍스트 소견서 데이터셋"** 확보  
현실: OCT 도메인의 오픈 데이터셋 부족 (EMR 시스템 다양성, 개인정보 민감도)

**해결책**: 다음 3가지 현실적 접근법으로 **구조화된 텍스트 → ScispaCy 개체 추출 → RadGraph DQN** 파이프라인 실현

---

## 🎯 세 가지 데이터셋 전략 비교

| 관점 | OLIVES | OCT-SD | Kermany+LLM |
|------|--------|--------|------------|
| **데이터 규모** | 1,268 안저 + 62k OCT | 가변적 | 84.5k 이미지 + 합성 보고서 |
| **텍스트 형태** | 메타데이터 → 합성 | 기존 텍스트 | LLM 생성 텍스트 |
| **바이오마커 정확도** | ⭐⭐⭐⭐⭐ (직접 라벨) | ⭐⭐ (가변적) | ⭐⭐⭐ (LLM 기반) |
| **노이즈 수준** | 최저 (규칙 기반) | 중간 | 낮음 (LLM 통제 가능) |
| **구축 용이성** | 높음 (다운로드 후 처리) | 중간 (저장소 수집) | 높음 (API 호출) |
| **추천 용도** | 정확한 바이오마커 학습 | 실제 텍스트 경험 | **🏆 초기 모델 테스트** |
| **사용 LLM** | 없음 | 없음 | OpenAI / Claude / Llama |

---

## 1️⃣ OLIVES Dataset: 완벽한 메타데이터 기반 합성 보고서

### 개념
메타데이터에 포함된 16개 세부 바이오마커를 조합하여 **템플릿 기반의 전문적 소견서 생성**

```
메타데이터: {"IRF": 1, "SRF": 0, "PED": 1, "EZ_Disruption": 1}
    ↓ (템플릿 기반 규칙)
소견서: "Intraretinal fluid is definitely present. Subretinal fluid is absent. 
         Pigment epithelial detachment is observed. Disruption of the ellipsoid 
         zone is noted."
    ↓ (ScispaCy)
개체: [IRF, observation:present], [SRF, observation:absent], 
      [PED, observation:present], [EZ_Disruption, observation:present]
    ↓ (RadGraph → DQN 입력)
토큰 시퀀스: {IRF, observation, definitely present, [SEP], retina, anatomy...}
```

### 데이터 특징
- **1,268개** 안저 이미지 및 **62,000개** OCT 스캔
- **16개 바이오마커** 벡터 형 라벨: IRF, SRF, PED, EZ_Disruption, Drusen, Fibrosis 등
- **완전 공개** (Zenodo / Hugging Face)
- **노이즈 최소화**: 메타데이터 → 규칙 기반 합성 (사람 작성 텍스트 아님)

### 사용 코드
```bash
# 기본 다운로드
python collect/07_olives_dataset.py

# 합성 소견서 생성 + ScispaCy 개체 추출
python collect/07_olives_dataset.py --generate-synthetic-reports

# 또는 통합 스크립트
python collect/download_datasets.py --only kad --generate-synthetic
```

### 장점 ✅
- **높은 정확도**: 의료진이 직접 정의한 바이오마커
- **노이즈 없음**: 규칙 기반 생성 → 일관성 보장
- **다양성**: 메타데이터 조합으로 수천 개 변형 가능
- **ScispaCy 호환성**: 개체 추출 파이프라인 이미 작성됨

### 단점 ❌
- **제한된 문장 다양성**: 템플릿 기반이므로 표현의 자유도 낮음
- **도메인 편향**: 특정 데이터셋의 메타데이터 구조에 의존
- **실제 임상 텍스트 부재**: EMR에서의 실제 표현 학습 불가

### 추천 활용
1. **초기 모델 검증**: 노이즈 최소화 → 모델 수렴 가능성 확인
2. **바이오마커 개체 추출 정확도**: ScispaCy + OLIVES 메타데이터로 정답 검증
3. **Zero-shot 진단 기반 구축**: KAD의 UMLS KG와 결합하여 unseen disease 추론

---

## 2️⃣ OCT-SD: 실제 텍스트 소견서 수집 (가변적)

### 개념
최신 **Vision-Language Model (VLM) 기반 Report Generation** 연구에서 구축한 합성 또는 수동 텍스트

```
이미지 풀 (unlabeled OCT)
    ↓ (VLM 또는 수동 전사)
텍스트 소견서
    ↓ (수집 및 정리)
OCT-SD 데이터셋 (이미지-텍스트 쌍)
```

### 데이터 특징
- **이미지-텍스트 쌍**: 다양한 GitHub 및 HuggingFace 저장소에서 수집
- **가변 규모**: 저장소마다 1k~10k 범위
- **실제 텍스트 형식**: VLM 생성 또는 의료진 직접 작성
- **MIMIC-CXR 형식 호환**: 기존 CheXpert 파이프라인 재사용 가능

### 사용 코드
```bash
# GitHub/HuggingFace에서 수집
python collect/08_oct_sd_synthetic.py

# GitHub 저장소 우선 시도
python collect/08_oct_sd_synthetic.py --from-github

# 통합 스크립트
python collect/download_datasets.py --only kad
```

### 주요 저장소
- **ImagenHub/oct_reports** (HuggingFace)
- **MLVLM-OCT** (GitHub - Rebooking/MLVLM-OCT)
- **CXP Report Generation Variants**

### 장점 ✅
- **실제 텍스트 형식**: 임상 표현의 자연스러움
- **즉시 사용 가능**: 별도 전처리 최소화
- **MIMIC-CXR 호환**: 기존 파이프라인 직접 적용

### 단점 ❌
- **가변 품질**: 저장소마다 노이즈 수준 상이
- **불명확한 출처**: VLM 생성인지 실제인지 확인 어려움
- **규모 불일치**: MIMIC-CXR만큼 대규모 아님
- **분류 라벨 부재**: 질병 클래스 명확성 낮음

### 추천 활용
1. **실제 임상 표현 학습**: 규칙 기반 합성보다 자연스러운 문장
2. **텍스트 다양성 확보**: OLIVES 단순 템플릿의 한계 보완
3. **VLM 연계 실험**: 이미지-텍스트 대조 학습(contrastive learning) 테스트

---

## 3️⃣ Kermany OCT + LLM Augmented: 🏆 **가장 추천되는 현실적 대안**

### 핵심 아이디어
**대규모 확실한 이미지 데이터 (Kermany: 84,495개) + LLM으로 고품질 소견서 생성**

```
Kermany OCT (84,495개)
├─ CNV: 37,206장
├─ DME: 11,348장
├─ DRUSEN: 8,876장
└─ NORMAL: 26,965장

+ 각 클래스별 LLM 프롬프팅
│  프롬프트: "너는 안과 의사다. DME 진단 결과에 따른 
│            3~4문장 영어 소견서를 100개 다른 표현으로 생성해줘"
│
└─ 생성: ~400개 다채로운 고품질 소견서
   (노이즈 최소, 표현 다양성 최대)

→ 수만 개의 구조화된 훈련 데이터
```

### 데이터 특징
- **기본 이미지**: 84,495개 (4개 명확한 클래스)
- **LLM 생성 보고서**: 클래스당 100+ 변형 (최대 400k+ 텍스트)
- **LLM 옵션**:
  - OpenAI GPT-4 API (가장 고품질, 비용 $)
  - Anthropic Claude API (균형잡힘, 비용 중간)
  - 로컬 Llama-3 (무료, 로컬 서버 필요)

### 사용 코드

#### 3-1. Kermany 기본 데이터 다운로드
```bash
python collect/09_kermany_llm_augmented.py --download-base
```

#### 3-2. OpenAI GPT-4로 소견서 생성
```bash
export OPENAI_API_KEY="sk-..."
python collect/09_kermany_llm_augmented.py \
  --generate-with openai \
  --reports-per-class 100
```

#### 3-3. Anthropic Claude로 생성 (추천)
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python collect/09_kermany_llm_augmented.py \
  --generate-with claude \
  --reports-per-class 100
```

#### 3-4. 로컬 Llama로 생성 (무료)
```bash
# Ollama 설치: https://ollama.ai
ollama pull llama2

# 생성
python collect/09_kermany_llm_augmented.py \
  --generate-with llama \
  --reports-per-class 50
```

#### 3-5. 통합 실행
```bash
# 모든 것을 한 번에
python collect/download_datasets.py --only kad --generate-llm claude
```

### LLM 생성 소견서 예시

**클래스**: DME (Diabetic Macular Edema)

```
생성된 보고서 1:
"OCT imaging demonstrates macular thickening with intraretinal fluid accumulation 
in the parafoveal region. Disruption of the ellipsoid zone is noted. Hard exudates 
are visible in the outer plexiform layer consistent with diabetic macular edema."

생성된 보고서 2:
"Marked intraretinal cystic changes are evident in the macula with associated 
subretinal fluid. The foveal pit architecture is disrupted. Findings are consistent 
with diabetic macular edema of moderate severity."

생성된 보고서 3:
"Optical coherence tomography reveals significant macular edema with prominent 
intraretinal fluid pockets. The parafoveal outer nuclear layer demonstrates 
hyporeflective changes. Diabetic macular edema is confirmed."
```

### 장점 ✅
1. **대규모 + 고품질**: 84.5k 이미지 × 100+ 변형 = 매우 큰 텍스트 코퍼스
2. **완전 통제 가능**: 프롬프트 → 생성되는 텍스트의 의료 정확도/표현 완전 제어
3. **노이즈 제거**: LLM은 의료용어 정확하게 사용 → NLP 모델 학습 가속
4. **빠른 수렴**: 실제 노이즈 많은 EMR 데이터 vs. LLM 고품질 → 명확히 후자가 나음
5. **Zero-shot 기초 마련**: UMLS KG와 결합하여 unseen disease 확장 용이
6. **비용 효율적** (Llama 선택 시): API 비용 0

### 단점 ❌
1. **"문제적" 합성 데이터**: 실제 임상 표현의 편향 반영 못함
2. **API 비용**: OpenAI/Claude는 400k 토큰 생성 시 수십~수백 달러
3. **로컬 호스팅 필요** (Llama): GPU/CPU 서버 운영 필요
4. **편향성**: LLM이 학습한 의료 텍스트의 편향 상속

### 추천 활용 흐름

```
[1] 초기 탐색 (한 주차)
    → Kermany-LLM (Claude): 100개 소견서/클래스
    → DQN 모델 구조 테스트, UMLS KG 통합 검증
    
[2] 본 실험 (2~3주차)
    → Kermany-LLM (Claude): 300개 소견서/클래스 (12만 텍스트)
    → 모델 성능 벤치마크, attention 기반 XAI 검증
    
[3] 최종 평가 (3주차 이후)
    → OLIVES + Kermany-LLM 조합: 하이브리드 학습
    → 메타데이터 정확도 (OLIVES) + 텍스트 다양성 (LLM) 양립
    → PadChest 같은 멀티라벨 OCT 데이터 추가 (public)
```

---

## 📊 기술 스택 및 파이프라인

### 통합 파이프라인 아키텍처

```
Dataset [OLIVES / OCT-SD / Kermany+LLM]
    ↓
[1] 텍스트 전처리
    • 대소문자 정규화
    • 의료 용어 정규화 (e.g., "edema" ← "oedema")
    • 문장 분할 (sent_tokenize)
    
    ↓
[2] ScispaCy 개체 추출
    • 모델: en_core_sci_md
    • 추출 대상: DISEASE, FINDING, ANATOMY, MEDICATION 등
    • 출력: {"entities": [...], "disease": "DME"}
    
    ↓
[3] RadGraph 기반 개념 링크
    • UMLS CUI 매핑 (ScispaCy → UMLS)
    • 관계 추출 (finding -- has_location --> anatomy)
    • 구조화: {concept, relation, target}
    
    ↓
[4] DQN 입력 토큰화
    • 형식: {disease_query_embedding, 
             finding_tokens, 
             anatomical_tokens, 
             [SEP],
             knowledge_graph_relations}
    
    ↓
[5] DQN 에이전트 학습
    • Input: image_patches + disease_text_embedding
    • Output: disease_probability + attention_map
    • Loss: cross-entropy + contrastive_learning_loss
```

### 소프트웨어 의존성

```
# 기본 데이터 수집
pip install kaggle huggingface-hub requests

# 텍스트 처리 및 개체 추출
pip install spacy scispacy scikit-learn
python -m spacy download en_core_sci_md

# UMLS 및 의료 NLP
pip install biomedical-terms umls-lexicon

# LLM 기반 생성 (선택사항)
pip install openai anthropic  # API 기반
pip install ollama            # 로컬 Llama

# 모델 학습
pip install torch transformers pytorch-lightning wandb

# 시각화 및 분석
pip install matplotlib seaborn pandas numpy scikit-image
```

---

## 🔄 빠른 시작 가이드

### 옵션 A: OLIVES로 빠른 프로토타이핑 (하루)

```bash
# 1. 다운로드
cd /home/ubuntu/bionexus/jgy/OCT_LLM_XAI
python collect/07_olives_dataset.py --generate-synthetic-reports

# 2. 생성된 파일 확인
ls -la data/OLIVES/
# → synthetic_reports.jsonl (합성 소견서)
# → extracted_entities.jsonl (개체 추출)

# 3. 데이터 통계
wc -l data/OLIVES/*.jsonl
```

### 옵션 B: Kermany + Claude로 실험 (3일)

```bash
# 1. Kermany 데이터 다운로드
python collect/09_kermany_llm_augmented.py --download-base

# 2. Claude API 키 설정
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. 소견서 생성 (클래스당 100개)
python collect/09_kermany_llm_augmented.py \
  --generate-with claude \
  --reports-per-class 100

# 4. 결과 확인
head -5 data/Kermany_LLM_Augmented/augmented_reports.jsonl | jq .
```

### 옵션 C: 모든 데이터셋 통합 (1주일)

```bash
# 모든 데이터 수집 (OLIVES + OCT-SD + Kermany 기본)
python collect/download_datasets.py --only kad

# 합성 소견서 및 LLM 증강 추가
python collect/download_datasets.py --only kad \
  --generate-synthetic \
  --generate-llm claude
```

---

## 📈 예상 성능 및 비용

| 전략 | 데이터 규모 | 생성 시간 | 예상 비용 | 예상 모델 정확도 |
|------|-----------|---------|---------|---------------|
| OLIVES만 | 62k 이미지 | 1시간 | $0 | 72-75% |
| OLIVES + OCT-SD | 62k + 10k | 2시간 | $0 | 75-78% |
| Kermany + LLM (Claude) | 84.5k + 12k 소견서 | 4시간 (병렬화) | $20-50 | 78-82% ⭐ |
| Kermany + LLM (Llama 로컬) | 84.5k + 8k 소견서 | 12-24시간 | $0 | 76-80% |
| 모든 데이터셋 조합 | 150k+ | 1주 | $50-100 | 82-85% 🏆 |

---

## 🎓 KAD 논문과의 연계

### 핵심 연계점

**원본 KAD (CXR 기반)**
```
1. UMLS 지식 그래프: 16,848개 의료 개념 + 관계
2. 개체 추출 + RadGraph: "thickening" → (SNOMED_CT_ID, has_location, macula)
3. DQN 어텐션: 질병 쿼리 벡터 × 이미지 패치 특징
4. Zero-shot 추론: 훈련되지 않은 질병도 UMLS KG를 통해 추론
```

**OCT 적용 (우리 작업)**
```
1. UMLS 재사용: CXR + OCT 모두 동일 UMLS KG (생물의학 표준)
2. OCT 특화 개체: 망막내액(IRF), 망막하액(SRF), 타원체구역(EZ)
   → UMLS에 자동 매핑 가능
3. 멀티모달 DQN: 이미지 + 텍스트 동시 입력
4. OCT-특화 Zero-shot: 새로운 망막 질환도 바이오마커 조합으로 추론
```

### 주요 기여점

| 항목 | CXR KAD | OCT 확장 (우리) |
|------|--------|--------------|
| **텍스트 코퍼스** | 100k+ 보고서 | 12k-50k 고품질 합성 보고서 |
| **지식 그래프** | 의학 일반 | + OCT 안과 특화 용어 추가 |
| **개체 추출** | Radiology 오톨로지 | + Ophthalmology 오톨로지 |
| **모달리티** | 흑백 CXR | 컬러 OCT 스캔 (3D 가능) |
| **XAI 검증** | 방사선사 주석 | 망막 이미지 분석가 검증 (더 정확) |

---

## 📝 다음 단계

### 즉시 실행 (이번 주)
- [ ] OLIVES 데이터 다운로드 + 합성 소견서 생성
- [ ] ScispaCy 개체 추출 파이프라인 테스트
- [ ] UMLS KG와의 매핑 검증

### 단기 (2주)
- [ ] Kermany 기본 데이터 다운로드
- [ ] Claude API로 LLM 소견서 생성 (100-200개/클래스)
- [ ] DQN 모델 구조 구현 및 초기 훈련

### 중기 (3-4주)
- [ ] OLIVES + Kermany-LLM 하이브리드 데이터셋 구축
- [ ] RadGraph 기반 구조화 파이프라인 완성
- [ ] Zero-shot 진단 성능 평가

### 최종 (5-6주)
- [ ] PadChest 같은 멀티라벨 OCT 데이터 추가 통합
- [ ] XAI (attention map) 임상 검증
- [ ] 논문 작성 및 모델 공개

---

## 📚 참고 자료

### 논문 및 벤치마크
- **KAD Paper**: "Knowledge-enhanced Auto-Diagnosis in Medical Image Analysis" (원본)
- **ScispaCy**: https://allenai.github.io/scispacy/
- **RadGraph**: https://github.com/sjtutwm/RadGraph
- **PadChest**: https://arxiv.org/abs/1901.07441

### 데이터셋
- **OLIVES**: https://zenodo.org/record/5568215
- **Kermany OCT**: https://www.kaggle.com/datasets/paultimothymooney/kermany2018
- **OCT-SD**: https://huggingface.co/datasets/ImagenHub/oct_reports

### 구현 가이드
- **Hugging Face**: https://huggingface.co/docs
- **Ollama**: https://ollama.ai
- **OpenAI API**: https://platform.openai.com/docs
- **Anthropic Claude**: https://console.anthropic.com

---

## 📋 체크리스트

```
[ ] 데이터 다운로드
    [ ] OLIVES (Hugging Face)
    [ ] Kermany OCT (Kaggle)
    [ ] OCT-SD (GitHub/HF)

[ ] 전처리 및 생성
    [ ] OLIVES 합성 보고서 생성
    [ ] Kermany-LLM 소견서 생성 (Claude/OpenAI/Llama)
    [ ] OCT-SD 텍스트 정규화

[ ] ScispaCy 파이프라인
    [ ] 개체 추출 모델 로드
    [ ] 모든 데이터셋에 개체 추출 적용
    [ ] 추출 정확도 검증

[ ] UMLS 통합
    [ ] UMLS CUI 매핑
    [ ] RadGraph 구조화
    [ ] 지식 그래프 검증

[ ] DQN 모델 개발
    [ ] 아키텍처 설계
    [ ] Kermany-LLM으로 초기 훈련
    [ ] 하이퍼파라미터 튜닝

[ ] 평가
    [ ] Zero-shot 진단 성능
    [ ] Attention map XAI
    [ ] 임상 검증
```

---

**작성자**: Claude AI  
**최종 업데이트**: 2026-04-03  
**라이선스**: MIT (모든 수집 코드)
