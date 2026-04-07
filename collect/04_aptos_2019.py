"""
APTOS 2019 데이터셋 다운로드
===============================
- 모달리티: Fundus
- 카테고리: 질병/병변
- 데이터 수: 3,662장
- 타겟: 당뇨망막병증(DR) 중증도 0~4단계 분류
- 출처: kaggle.com/c/aptos2019-blindness-detection
- Kaggle 미러: sovitrath/diabetic-retinopathy-224x224-2019-data (224x224 리사이즈)

사용법:
    python 04_aptos_2019.py
"""

import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_DIR = "APTOS_2019"
KAGGLE_REF = "sovitrath/diabetic-retinopathy-224x224-2019-data"


def main(data_dir: Path = DATA_DIR):
    dest = data_dir / DATASET_DIR
    if dest.exists() and any(dest.iterdir()):
        print(f"[SKIP] APTOS 2019 — 이미 존재 ({dest})")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[DOWN] APTOS 2019 → {dest}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_REF, "--unzip", "-p", str(dest)],
        check=True,
    )
    print(f"[DONE] APTOS 2019 ({dest})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="APTOS 2019 다운로드")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()
    main(Path(args.data_dir) if args.data_dir else DATA_DIR)
