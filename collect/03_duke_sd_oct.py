"""
Duke SD-OCT 데이터셋 다운로드
================================
- 모달리티: OCT
- 카테고리: 해부학 구조
- 데이터 수: 110장
- 타겟: 8개의 망막 층(Layer) 경계 및 유체
- 출처: Kaggle (Chiu 2015 / ReLayNet)

사용법:
    python 03_duke_sd_oct.py
"""

import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_DIR = "Duke_SD_OCT"
KAGGLE_REF = "paultimothymooney/chiu-2015"


def main(data_dir: Path = DATA_DIR):
    dest = data_dir / DATASET_DIR
    if dest.exists() and any(dest.iterdir()):
        print(f"[SKIP] Duke SD-OCT — 이미 존재 ({dest})")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[DOWN] Duke SD-OCT → {dest}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_REF, "--unzip", "-p", str(dest)],
        check=True,
    )
    print(f"[DONE] Duke SD-OCT ({dest})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Duke SD-OCT 다운로드")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()
    main(Path(args.data_dir) if args.data_dir else DATA_DIR)
