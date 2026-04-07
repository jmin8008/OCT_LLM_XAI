"""
OCT-SD (Synthetic OCT-Report Pairs) 데이터셋 수집
==================================================
- 모달리티: OCT 이미지 + 합성 의료 소견서
- 특징: 최신 VLM 기반 Report Generation 연구에서 활용
- 접근성: GitHub 및 HuggingFace에 공개된 벤치마크
- 출처: https://github.com/CheXpert 및 관련 OCT Report Gen 프로젝트

사전 요구사항:
    pip install requests huggingface-hub pandas pillow

사용법:
    python 08_oct_sd_synthetic.py
    python 08_oct_sd_synthetic.py --data-dir /path/to/data
    python 08_oct_sd_synthetic.py --from-github
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import logging
import urllib.request
import tarfile

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_DIR = "OCT_SD"

# 주요 벤치마크 저장소 및 HuggingFace 데이터셋
GITHUB_SOURCES = [
    {
        "name": "OCT_Report_Generation",
        "repo": "https://github.com/Rebooking/MLVLM-OCT",
        "description": "Medical Language-Vision Model for OCT Report Generation"
    },
    {
        "name": "CheXpert_Variants",
        "hf_dataset": "microsoft/cxp-report-generation",
        "description": "CheXpert-style report generation adapted for OCT"
    }
]

HF_FALLBACK = "ImagenHub/oct_reports"  # 대안 HF 데이터셋


def try_github_clone(repo_url: str, dest: Path) -> bool:
    """GitHub 저장소 클론 시도"""
    try:
        logger.info(f"[DOWN] GitHub 저장소 → {repo_url}")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(dest)],
            check=True,
            capture_output=True
        )
        logger.info(f"[DONE] GitHub 클론 성공")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"[WARN] GitHub 클론 실패: {e}")
        return False
    except FileNotFoundError:
        logger.warning("[WARN] git 설치 필요")
        return False


def download_from_huggingface(dataset_id: str, dest: Path) -> bool:
    """HuggingFace에서 데이터셋 다운로드"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("[ERR] huggingface-hub 설치 필요: pip install huggingface-hub")
        return False

    try:
        logger.info(f"[DOWN] HuggingFace 데이터셋: {dataset_id}")
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=str(dest),
            resume_download=True
        )
        logger.info(f"[DONE] HF 다운로드 성공")
        return True
    except Exception as e:
        logger.warning(f"[WARN] HF 다운로드 실패: {e}")
        return False


def generate_report_index(data_dir: Path) -> Path:
    """
    다운로드한 OCT-SD 데이터의 인덱스 생성
    이미지와 텍스트를 매핑하는 메타데이터 JSON 생성
    """
    oct_sd_dir = data_dir / DATASET_DIR
    index_file = oct_sd_dir / "report_index.jsonl"

    if not oct_sd_dir.exists():
        logger.warning(f"[WARN] OCT-SD 디렉토리 없음: {oct_sd_dir}")
        return None

    # 이미지 파일 탐색
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    image_files = [
        f for f in oct_sd_dir.rglob('*')
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    # 텍스트 파일 탐색 (같은 이름의 .txt)
    index_entries = []
    for img_file in image_files:
        txt_file = img_file.with_suffix('.txt')
        report_file = img_file.with_stem(img_file.stem + '_report').with_suffix('.txt')

        # 여러 텍스트 파일 형식 지원
        text_content = None
        if txt_file.exists():
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()
        elif report_file.exists():
            with open(report_file, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()

        # 이미지-텍스트 쌍 저장
        entry = {
            "image_path": str(img_file.relative_to(oct_sd_dir)),
            "report_text": text_content if text_content else "",
            "has_report": text_content is not None,
            "image_size": img_file.stat().st_size,
            "disease_type": infer_disease_from_path(img_file)
        }
        index_entries.append(entry)

    # JSONL로 저장
    with open(index_file, 'w', encoding='utf-8') as f:
        for entry in index_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logger.info(f"[DONE] 인덱스 생성: {len(index_entries)}개 항목 → {index_file}")
    return index_file


def infer_disease_from_path(file_path: Path) -> str:
    """파일 경로에서 질병 유형 추론"""
    path_lower = str(file_path).lower()

    disease_keywords = {
        'cme': 'Central Macular Edema',
        'dme': 'Diabetic Macular Edema',
        'drusen': 'Drusen',
        'cnv': 'Choroidal Neovascularization',
        'normal': 'Normal',
        'amd': 'Age-related Macular Degeneration'
    }

    for keyword, disease_name in disease_keywords.items():
        if keyword in path_lower:
            return disease_name

    return "Unknown"


def standardize_report_format(data_dir: Path) -> Path:
    """
    OCT-SD 소견서를 표준화된 MIMIC-CXR 형식으로 변환
    → RadGraph/ScispaCy 파이프라인과의 호환성 향상
    """
    index_file = data_dir / DATASET_DIR / "report_index.jsonl"
    output_file = data_dir / DATASET_DIR / "standardized_reports.jsonl"

    if not index_file.exists():
        logger.warning(f"[WARN] 인덱스 파일 없음: {index_file}")
        return None

    standardized = []
    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)

            # 보고서 텍스트를 MIMIC-CXR 섹션 형식으로 변환
            report = entry.get('report_text', '')

            standardized_entry = {
                "image_path": entry['image_path'],
                "disease_type": entry['disease_type'],
                "findings": report,  # Findings 섹션
                "impression": "",     # Impression 섹션 (추론 필요)
                "original_report": report,
                "report_length": len(report.split()),
                "has_structured_report": bool(report)
            }
            standardized.append(standardized_entry)

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in standardized:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logger.info(f"[DONE] 표준화 완료: {len(standardized)}개 → {output_file}")
    return output_file


def main(data_dir: Path = DATA_DIR, from_github: bool = False):
    """OCT-SD 데이터셋 다운로드 및 처리"""

    dest = data_dir / DATASET_DIR
    dest.mkdir(parents=True, exist_ok=True)

    if dest.exists() and any(dest.iterdir()):
        logger.info(f"[SKIP] OCT-SD — 이미 존재 ({dest})")
    else:
        success = False

        if from_github:
            # GitHub 저장소 시도
            for source in GITHUB_SOURCES:
                if "repo" in source:
                    success = try_github_clone(source["repo"], dest)
                    if success:
                        logger.info(f"[INFO] {source['name']} 다운로드 성공")
                        break

        # GitHub 실패 시 HuggingFace 시도
        if not success:
            success = download_from_huggingface(HF_FALLBACK, dest)

        if not success:
            logger.warning("[WARN] 모든 다운로드 방법 실패")
            logger.info("[INFO] 수동 다운로드: https://huggingface.co/datasets/ImagenHub/oct_reports")
            return

    # 인덱스 및 표준화 처리
    generate_report_index(data_dir)
    standardize_report_format(data_dir)

    logger.info(f"[INFO] OCT-SD 데이터 위치: {dest}")
    logger.info(f"[INFO] 인덱스: {dest}/report_index.jsonl")
    logger.info(f"[INFO] 표준화 보고서: {dest}/standardized_reports.jsonl")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OCT-SD Synthetic Report Pairs 수집")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--from-github", action="store_true",
                       help="GitHub 저장소에서 먼저 시도 (기본값: HuggingFace)")
    args = parser.parse_args()

    main(
        Path(args.data_dir) if args.data_dir else DATA_DIR,
        from_github=args.from_github
    )
