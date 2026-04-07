"""
Kermany OCT 데이터셋 다운로드
==============================
- 모달리티: OCT
- 카테고리: 질병/병변
- 데이터 수: 84,495장
- 타겟: CNV, DME, DRUSEN 등 주요 4개 질환 분류
- 출처: kaggle.com/datasets/paultimothymooney/kermany2018

사전 요구사항:
    pip install kaggle
    ~/.kaggle/kaggle.json 에 API 토큰 설정

사용법:
    python 01_kermany_oct.py
    python 01_kermany_oct.py --data-dir /path/to/data
"""

import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_DIR = "OCT2017"
KAGGLE_REF = "paultimothymooney/kermany2018"


def main(data_dir: Path = DATA_DIR):
    dest = data_dir / DATASET_DIR
    if dest.exists() and any(dest.iterdir()):
        print(f"[SKIP] Kermany OCT — 이미 존재 ({dest})")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[DOWN] Kermany OCT → {dest}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_REF, "--unzip", "-p", str(dest)],
        check=True,
    )
    print(f"[DONE] Kermany OCT ({dest})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kermany OCT 다운로드")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()
    main(Path(args.data_dir) if args.data_dir else DATA_DIR)
