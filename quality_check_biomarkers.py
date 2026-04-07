"""
바이오마커 데이터셋 품질 검증
각 클래스별 6개 이미지 샘플링 및 생성된 텍스트 확인
"""

import json
from pathlib import Path
from collections import defaultdict

DATASET_FILE = Path('/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/data/OCT2017/biomarker_dataset/biomarker_annotations.jsonl')

def load_samples_by_class(num_samples=6):
    """각 클래스별 샘플 로드"""
    samples_by_class = defaultdict(list)

    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            class_label = data['class']

            if len(samples_by_class[class_label]) < num_samples:
                samples_by_class[class_label].append(data)

    return samples_by_class

def display_biomarker_details(sample, class_label):
    """샘플의 바이오마커 정보 표시"""
    print(f"\n{'='*80}")
    print(f"📄 Image: {sample['image']} | Class: {class_label}")
    print(f"{'='*80}")

    for bm in sample['biomarkers']:
        biomarker = bm['biomarker']
        status = bm['status']
        explanation = bm['explanation']
        variants = bm['variants']

        # 상태에 따른 색상 표시
        status_mark = "✅" if status == "present" else "❌"

        print(f"\n{status_mark} {biomarker.upper()}")
        print(f"   Status: {status}")
        print(f"   Explanation: {explanation}")
        print(f"   Generated texts:")
        for i, variant in enumerate(variants, 1):
            print(f"      {i}. {variant}")

def analyze_class_accuracy(class_label, samples):
    """클래스별 정확도 분석"""
    print(f"\n\n{'#'*80}")
    print(f"📊 {class_label} 클래스 정확도 분석 ({len(samples)}개 샘플)")
    print(f"{'#'*80}")

    # 바이오마커별 통계
    biomarker_stats = defaultdict(lambda: {'present': 0, 'absent': 0})

    for sample in samples:
        for bm in sample['biomarkers']:
            biomarker = bm['biomarker']
            status = bm['status']
            if status == 'present':
                biomarker_stats[biomarker]['present'] += 1
            else:
                biomarker_stats[biomarker]['absent'] += 1

    print("\n바이오마커별 분포 (6개 이미지 중):")
    print("-" * 60)
    for biomarker in sorted(biomarker_stats.keys()):
        stats = biomarker_stats[biomarker]
        present_pct = stats['present'] * 100 / (len(samples) * 1) if stats['present'] > 0 else 0
        print(f"  {biomarker:40s} | Present: {stats['present']}/6 | Absent: {stats['absent']}/6")

    # 각 샘플 상세 표시
    for i, sample in enumerate(samples, 1):
        display_biomarker_details(sample, class_label)

def main():
    print("\n🔍 Kermany 바이오마커 데이터셋 품질 검증 시작...\n")

    samples = load_samples_by_class(num_samples=6)

    print(f"로드된 클래스: {list(samples.keys())}")
    print(f"각 클래스별 샘플: {len(samples[list(samples.keys())[0]])}개\n")

    # 클래스별 분석
    for class_label in ['NORMAL', 'CNV', 'DME', 'DRUSEN']:
        if class_label in samples:
            analyze_class_accuracy(class_label, samples[class_label])

    print(f"\n\n{'='*80}")
    print("📋 요약: 임상적 정확도 판단")
    print(f"{'='*80}\n")

    # 의료 프로파일 확인
    expected_profiles = {
        'NORMAL': {'present': 0, 'absent': 10},
        'CNV': {'present': 8, 'absent': 2},
        'DME': {'present': 2, 'absent': 8},
        'DRUSEN': {'present': 3, 'absent': 7},
    }

    print("예상되는 클래스별 바이오마커 프로파일:")
    print("-" * 60)
    for class_label, profile in expected_profiles.items():
        print(f"  {class_label:10s} | Present: {profile['present']}/10 | Absent: {profile['absent']}/10")

    print("\n생성된 데이터가 의료 논문 기반 프로파일과 일치하는지 확인하세요.")
    print("각 텍스트 변형이 자연스러운 임상 영어인지도 검토하세요.\n")

if __name__ == "__main__":
    main()
