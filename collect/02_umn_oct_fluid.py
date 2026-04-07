"""
UMN OCT Fluid 데이터셋 다운로드
=================================
- 모달리티: OCT
- 카테고리: 바이오마커
- 데이터 수: 600장
- 타겟: IRF, SRF, PED 유체 바이오마커
- 출처: conservancy.umn.edu/handle/11299/215706
- Kaggle 미러: zeeshanahmed13/intraretinal-cystoid-fluid

사용법:
    python 02_umn_oct_fluid.py
"""

import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_DIR = "UMN_OCT_Fluid"
KAGGLE_REF = "zeeshanahmed13/intraretinal-cystoid-fluid"


def main(data_dir: Path = DATA_DIR):
    dest = data_dir / DATASET_DIR
    if dest.exists() and any(dest.iterdir()):
        print(f"[SKIP] UMN OCT Fluid — 이미 존재 ({dest})")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[DOWN] UMN OCT Fluid → {dest}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_REF, "--unzip", "-p", str(dest)],
        check=True,
    )
    print(f"[DONE] UMN OCT Fluid ({dest})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UMN OCT Fluid 다운로드")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()
    main(Path(args.data_dir) if args.data_dir else DATA_DIR)
