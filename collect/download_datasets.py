"""
안과 AI 연구용 데이터셋 전체 다운로드
======================================
개별 스크립트를 일괄 실행합니다.

사용법:
    python download_datasets.py              # 전체 다운로드
    python download_datasets.py --only oct   # OCT 데이터만
    python download_datasets.py --only fundus # Fundus 데이터만
"""

import argparse
from pathlib import Path

import importlib
import sys

# 같은 디렉토리의 개별 스크립트를 import
sys.path.insert(0, str(Path(__file__).resolve().parent))

download_kermany = importlib.import_module("01_kermany_oct").main
download_umn     = importlib.import_module("02_umn_oct_fluid").main
download_duke    = importlib.import_module("03_duke_sd_oct").main
download_aptos   = importlib.import_module("04_aptos_2019").main
download_stare   = importlib.import_module("05_stare").main
download_rim_one = importlib.import_module("06_rim_one_dl").main

# KAD 논문 파이프라인용 세 가지 새로운 데이터셋
download_olives  = importlib.import_module("07_olives_dataset").main
download_oct_sd  = importlib.import_module("08_oct_sd_synthetic").main
download_kermany_llm = importlib.import_module("09_kermany_llm_augmented").main

# fmt: off
OCT_DOWNLOADS    = [download_kermany, download_umn, download_duke]
FUNDUS_DOWNLOADS = [download_aptos, download_stare, download_rim_one]
KAD_DATASETS     = [download_olives, download_oct_sd, download_kermany_llm]
# fmt: on

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    parser = argparse.ArgumentParser(description="안과 AI 데이터셋 전체 다운로드")
    parser.add_argument("--only", choices=["oct", "fundus", "kad"], default=None,
                       help="특정 데이터셋 카테고리만 다운로드")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--generate-synthetic", action="store_true",
                       help="OLIVES에서 합성 보고서 생성")
    parser.add_argument("--generate-llm", choices=["openai", "claude", "llama"], default=None,
                       help="Kermany LLM 증강 보고서 생성")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR

    downloads = []
    if args.only == "oct":
        downloads = OCT_DOWNLOADS
    elif args.only == "fundus":
        downloads = FUNDUS_DOWNLOADS
    elif args.only == "kad":
        downloads = KAD_DATASETS
    else:
        downloads = OCT_DOWNLOADS + FUNDUS_DOWNLOADS + KAD_DATASETS

    for fn in downloads:
        fn(data_dir)

    # 추가 처리
    if args.generate_synthetic:
        print("[PROC] OLIVES 합성 보고서 생성 중...")
        download_olives(data_dir, generate_reports=True)

    if args.generate_llm:
        print(f"[PROC] Kermany LLM 증강 ({args.generate_llm}) 중...")
        download_kermany_llm(data_dir, generate_with=args.generate_llm)

    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
