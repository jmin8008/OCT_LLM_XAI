"""
OLIVES Dataset 다운로드 및 처리
================================
- 모달리티: OCT + Fundus 이미지
- 데이터 수: 1,268개 안저 이미지, 62,000개 OCT 스캔
- 메타데이터: 16개 세부 바이오마커 (IRF, SRF, PED, EZ 손상 등)
- 출처: Zenodo / Hugging Face (완전 공개)
- URL: https://zenodo.org/record/5568215 또는
       https://huggingface.co/datasets/chboukarov/OLIVES

사전 요구사항:
    pip install requests huggingface-hub pandas pillow

사용법:
    python 07_olives_dataset.py
    python 07_olives_dataset.py --data-dir /path/to/data
    python 07_olives_dataset.py --generate-synthetic-reports
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_DIR = "OLIVES"
HF_REPO = "chboukarov/OLIVES"


def download_from_zenodo(data_dir: Path) -> Path:
    """Zenodo에서 OLIVES 데이터셋 다운로드"""
    try:
        import urllib.request
        import zipfile
    except ImportError:
        logger.error("[ERR] urllib 설치 필요")
        raise

    dest = data_dir / DATASET_DIR
    if dest.exists() and any(dest.iterdir()):
        logger.info(f"[SKIP] OLIVES — 이미 존재 ({dest})")
        return dest

    dest.mkdir(parents=True, exist_ok=True)

    # Zenodo에서 OLIVES 데이터 다운로드
    zenodo_url = "https://zenodo.org/api/records/5568215"

    logger.info(f"[DOWN] OLIVES (Zenodo) 메타데이터 조회 중...")
    try:
        import requests
        response = requests.get(zenodo_url, timeout=10)
        response.raise_for_status()
        record = response.json()

        # 가장 큰 파일 (완전한 데이터셋) 찾기
        files = record.get('files', [])
        if not files:
            logger.warning("[WARN] Zenodo 파일 없음")
            return None

        # ZIP 파일 찾기
        zip_file = next((f for f in files if f['filename'].endswith('.zip')),
                        files[0] if files else None)

        if not zip_file:
            logger.warning("[WARN] ZIP 파일을 찾을 수 없음")
            return None

        download_url = zip_file['links']['self']
        logger.info(f"[DOWN] OLIVES → {download_url}")

        # 다운로드
        zip_path = dest / "olives.zip"
        urllib.request.urlretrieve(download_url, zip_path, reporthook=_download_progress)

        # 압축 해제
        logger.info(f"[PROC] 압축 해제 중...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest)

        zip_path.unlink()
        logger.info(f"[DONE] OLIVES 다운로드 및 압축 해제 완료")

    except Exception as e:
        logger.error(f"[ERR] Zenodo 다운로드 실패: {e}")
        raise

    return dest


def _download_progress(block_num, block_size, total_size):
    """다운로드 진행률 표시"""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(downloaded * 100 / total_size, 100)
        if block_num % 10 == 0:  # 10 블록마다 출력
            logger.info(f"[PROG] {percent:.1f}% ({downloaded / (1024**2):.1f} MB / {total_size / (1024**2):.1f} MB)")


def download_from_huggingface(data_dir: Path) -> Path:
    """Hugging Face에서 OLIVES 데이터셋 다운로드 (대안)"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("[WARN] huggingface-hub 없음 - Zenodo 사용")
        return download_from_zenodo(data_dir)

    dest = data_dir / DATASET_DIR
    if dest.exists() and any(dest.iterdir()):
        logger.info(f"[SKIP] OLIVES — 이미 존재 ({dest})")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"[DOWN] OLIVES (HuggingFace 시도) → {dest}")

    try:
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            local_dir=str(dest),
            resume_download=True
        )
        logger.info(f"[DONE] OLIVES 다운로드 완료")
    except Exception as e:
        logger.warning(f"[WARN] HF 다운로드 실패: {e}")
        logger.info("[INFO] Zenodo 대안으로 재시도...")
        return download_from_zenodo(data_dir)

    return dest


def load_metadata(data_dir: Path) -> Dict:
    """메타데이터 로드"""
    meta_file = data_dir / DATASET_DIR / "metadata.json"
    if not meta_file.exists():
        logger.warning(f"[WARN] 메타데이터 없음: {meta_file}")
        return {}

    with open(meta_file, 'r') as f:
        return json.load(f)


def generate_synthetic_reports(data_dir: Path, output_file: Optional[Path] = None) -> Path:
    """
    OLIVES 메타데이터로부터 합성 소견서 생성

    바이오마커 조합 → 템플릿 기반 텍스트 생성
    예: IRF=1, SRF=0, EZ_Disruption=1
        → "Intraretinal fluid is definitely present.
             Subretinal fluid is absent.
             Disruption of the ellipsoid zone is observed."
    """
    metadata = load_metadata(data_dir)
    if not metadata:
        logger.warning("[WARN] 메타데이터 로드 실패 - 합성 소견서 생성 불가")
        return None

    output_file = output_file or (data_dir / DATASET_DIR / "synthetic_reports.jsonl")

    # 바이오마커 → 텍스트 매핑 템플릿
    biomarker_templates = {
        'IRF': {
            1: "Intraretinal fluid is definitely present.",
            0: "Intraretinal fluid is absent."
        },
        'SRF': {
            1: "Subretinal fluid is detected.",
            0: "Subretinal fluid is not present."
        },
        'PED': {
            1: "Pigment epithelial detachment is observed.",
            0: "No pigment epithelial detachment."
        },
        'EZ_Disruption': {
            1: "Disruption of the ellipsoid zone is noted.",
            0: "Ellipsoid zone integrity is preserved."
        },
        'Drusen': {
            1: "Drusen are visible within the macula.",
            0: "No drusen detected."
        },
        'Fibrosis': {
            1: "Subretinal or intraretinal fibrosis is present.",
            0: "No fibrosis noted."
        }
    }

    reports = []
    for sample_id, sample_meta in metadata.items():
        # 바이오마커 추출
        biomarkers = {}
        report_text = []

        for marker in biomarker_templates.keys():
            if marker in sample_meta:
                value = sample_meta[marker]
                biomarkers[marker] = value
                if marker in biomarker_templates:
                    report_text.append(biomarker_templates[marker].get(value, ""))

        # 빈 항목 제거 후 문장 결합
        report_text = [t for t in report_text if t]
        full_report = " ".join(report_text)

        if full_report:
            reports.append({
                "sample_id": sample_id,
                "biomarkers": biomarkers,
                "synthetic_report": full_report,
                "disease_label": sample_meta.get("disease", "Unknown")
            })

    # JSONL 형식으로 저장
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for report in reports:
            f.write(json.dumps(report, ensure_ascii=False) + '\n')

    logger.info(f"[DONE] 합성 소견서 {len(reports)}개 생성 → {output_file}")
    return output_file


def extract_text_entities(reports_file: Path) -> Path:
    """
    ScispaCy를 이용한 개체 추출
    합성 소견서에서 의료 개체 및 바이오마커 추출
    """
    try:
        import scispacy
        import spacy
    except ImportError:
        logger.error("[ERR] scispacy 설치 필요: pip install scispacy")
        return None

    try:
        nlp = spacy.load("en_core_sci_md")
    except OSError:
        logger.warning("[WARN] en_core_sci_md 모델 없음. 다운로드 중...")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_sci_md"], check=True)
        nlp = spacy.load("en_core_sci_md")

    output_file = reports_file.parent / "extracted_entities.jsonl"

    entities_list = []
    with open(reports_file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            doc = nlp(sample['synthetic_report'])

            entities = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "biomarker": sample['biomarkers']
                }
                for ent in doc.ents
            ]

            entities_list.append({
                "sample_id": sample['sample_id'],
                "report": sample['synthetic_report'],
                "entities": entities,
                "disease": sample['disease_label']
            })

    with open(output_file, 'w') as f:
        for item in entities_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info(f"[DONE] 개체 추출 완료 → {output_file}")
    return output_file


def main(data_dir: Path = DATA_DIR, generate_reports: bool = False):
    """OLIVES 데이터셋 다운로드 및 처리"""
    try:
        data_path = download_from_huggingface(data_dir)

        if generate_reports:
            reports_file = generate_synthetic_reports(data_dir)
            if reports_file:
                extract_text_entities(reports_file)

        logger.info(f"[INFO] OLIVES 데이터 위치: {data_path}")
        logger.info(f"[INFO] 메타데이터: {data_path}/metadata.json")

    except Exception as e:
        logger.error(f"[ERR] OLIVES 다운로드 실패: {e}")
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OLIVES Dataset 다운로드")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--generate-synthetic-reports", action="store_true",
                       help="메타데이터로부터 합성 소견서 생성")
    args = parser.parse_args()

    main(
        Path(args.data_dir) if args.data_dir else DATA_DIR,
        generate_reports=args.generate_synthetic_reports
    )
