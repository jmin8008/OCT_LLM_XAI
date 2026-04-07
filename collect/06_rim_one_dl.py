"""
RIM-ONE DL 데이터셋 다운로드
===============================
- 모달리티: Fundus
- 카테고리: 해부학 구조
- 데이터 수: 485장 + OD/OC segmentation mask 970개
- 타겟: 시신경 유두(OD) 및 시신경 잔(OC) 구조, 녹내장 분류
- 출처: kaggle.com/datasets/tavoosi/rim-one-dl
- Kaggle 미러 (이미지): orvile/rim-one-retinal-dataset-for-assessing-glaucoma
- Segmentation mask: github.com/miag-ull/rim-one-dl (Google Drive)

사전 요구사항:
    pip install kaggle gdown

사용법:
    python 06_rim_one_dl.py
"""

import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_DIR = "RIM_ONE_DL"
KAGGLE_REF = "orvile/rim-one-retinal-dataset-for-assessing-glaucoma"
GDRIVE_FILE_ID = "1eb1V9V65TuwFNYmYsIzgdAgdWyD6o7bG"
MASK_ZIP = "RIM-ONE_DL_reference_segmentations.zip"


def download_images(dest: Path):
    """Kaggle에서 분류용 fundus 이미지를 다운로드합니다."""
    print("  [1/2] Fundus 이미지 다운로드 (Kaggle)")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_REF, "--unzip", "-p", str(dest)],
        check=True,
    )


def download_segmentation_masks(dest: Path):
    """Google Drive에서 OD/OC segmentation mask를 다운로드합니다."""
    mask_zip = dest / MASK_ZIP
    mask_dir = dest / "RIM-ONE_DL_reference_segmentations"

    if mask_dir.exists():
        print("  [2/2] Segmentation mask — 이미 존재")
        return

    print("  [2/2] OD/OC segmentation mask 다운로드 (Google Drive)")
    subprocess.run(
        ["gdown", GDRIVE_FILE_ID, "-O", str(mask_zip)],
        check=True,
    )
    subprocess.run(
        ["unzip", "-qo", str(mask_zip), "-d", str(dest)],
        check=True,
    )
    print("  Segmentation mask 압축 해제 완료")


def main(data_dir: Path = DATA_DIR):
    dest = data_dir / DATASET_DIR

    if dest.exists() and any(dest.iterdir()):
        # 이미지는 있지만 mask가 없을 수 있으므로 mask만 추가 다운로드
        mask_dir = dest / "RIM-ONE_DL_reference_segmentations"
        if mask_dir.exists():
            print(f"[SKIP] RIM-ONE DL — 이미 존재 ({dest})")
            return
        print(f"[DOWN] RIM-ONE DL segmentation mask 추가 다운로드")
        download_segmentation_masks(dest)
        print(f"[DONE] RIM-ONE DL segmentation mask ({dest})")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[DOWN] RIM-ONE DL → {dest}")
    download_images(dest)
    download_segmentation_masks(dest)
    print(f"[DONE] RIM-ONE DL ({dest})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RIM-ONE DL 다운로드")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()
    main(Path(args.data_dir) if args.data_dir else DATA_DIR)
