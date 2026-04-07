"""
STARE 데이터셋 다운로드
=========================
- 모달리티: Fundus
- 카테고리: 바이오마커
- 데이터 수: 402장 (전체 이미지) + 20장 (혈관 세그멘테이션 서브셋)
- 타겟: 15개 진단코드 (multi-label) + 혈관 pixel-level segmentation
- 출처: cecas.clemson.edu/~ahoover/stare/

사용법:
    python 05_stare.py
"""

import gzip
import shutil
import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_DIR = "STARE"
URLS = [
    # 전체 이미지 (402장) + 진단 레이블
    "http://cecas.clemson.edu/~ahoover/stare/images/all-images.zip",
    "http://cecas.clemson.edu/~ahoover/stare/diagnoses/all-mg-codes.txt",
    # 혈관 세그멘테이션 서브셋 (20장)
    "http://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar",
    "http://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar",
    "http://cecas.clemson.edu/~ahoover/stare/probing/labels-vk.tar",
]


def download_and_extract(dest: Path):
    for url in URLS:
        filename = url.split("/")[-1]
        filepath = dest / filename
        if filepath.exists():
            print(f"  이미 존재: {filename}")
        else:
            subprocess.run(
                ["wget", "-q", "--no-check-certificate", "-O", str(filepath), url],
                check=True,
            )
            print(f"  다운로드: {filename}")

        if filename.endswith(".tar"):
            subprocess.run(["tar", "xf", str(filepath), "-C", str(dest)], check=True)
            print(f"  압축 해제: {filename}")
        elif filename.endswith(".zip"):
            subprocess.run(["unzip", "-qo", str(filepath), "-d", str(dest)], check=True)
            print(f"  압축 해제: {filename}")


def extract_gz(dest: Path):
    count = 0
    for gz_file in dest.glob("*.gz"):
        out_file = dest / gz_file.stem
        if out_file.exists():
            continue
        with gzip.open(gz_file, "rb") as f_in, open(out_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        count += 1
    if count:
        print(f"  .gz 파일 {count}개 압축 해제 완료")


def main(data_dir: Path = DATA_DIR):
    dest = data_dir / DATASET_DIR
    if dest.exists() and any(dest.iterdir()):
        print(f"[SKIP] STARE — 이미 존재 ({dest})")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[DOWN] STARE → {dest}")
    download_and_extract(dest)
    extract_gz(dest)
    print(f"[DONE] STARE ({dest})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="STARE 다운로드")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()
    main(Path(args.data_dir) if args.data_dir else DATA_DIR)
