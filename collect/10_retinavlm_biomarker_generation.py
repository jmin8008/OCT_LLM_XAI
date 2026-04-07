"""
RetinaVLM을 사용한 바이오마커 생성 및 판단 (Task D)
====================================================
기존 학습된 AMD 모델을 사용하여 OCT 이미지의 바이오마커 present/absent 판단

사용법:
    python collect/10_retinavlm_biomarker_generation.py --num-samples 100
    python collect/10_retinavlm_biomarker_generation.py --num-samples 50 --classes CNV NORMAL
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
from PIL import Image

import torch

# RetinaVLM import
sys.path.insert(0, '/home/ubuntu/bionexus/jgy/OCT_LLM_XAI')
sys.path.insert(0, '/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/SpecialistVLMs')
try:
    from models.retinavlm_wrapper import RetinaVLM, RetinaVLMConfig
except ImportError as e:
    print(f"[ERR] RetinaVLM import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path('/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/data')
SAVE_DIR = Path('/home/ubuntu/bionexus/jgy/OCT_LLM_XAI/saved_models/RetinaVLM-Specialist-Dequantized')
OUTPUT_DIR = DATA_DIR / 'Kermany_RetinaVLM_Biomarkers'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 바이오마커 정의 (Task D spec from notebook)
BIOMARKER_SPECS = [
    ("subretinal fluid", "is"),
    ("intraretinal fluid", "is"),
    ("pigment epithelial detachment", "is"),
    ("drusen", "are"),
    ("retinal pigment epithelium atrophy", "is"),
    ("geographic atrophy", "is"),
    ("subretinal hyperreflective material", "is"),
    ("hyperreflective foci", "are"),
    ("choroidal neovascularization", "is"),
    ("epiretinal membrane", "is"),
]

TASK_D_QUERY_TEMPLATE = (
    "Describe the OCT image in detail and list all biomarkers or abnormalities. "
    "Detail if there are any signs indicating that {biomarker} might be present, even if there is "
    "only a small amount.\n"
    "Finally, conclude your findings by telling me if {biomarker} {article} \"not present\", or if "
    "potentially any amount of {biomarker} {article} \"present\" in the OCT image."
)
TASK_D_STEP2_TEMPLATE = "To conclude these findings, in the OCT image {biomarker} {article}"


def load_retinavlm_model(save_dir: Path) -> Tuple[RetinaVLM, str]:
    """RetinaVLM 모델 로드"""
    logger.info("[PROC] RetinaVLM 모델 로드 중...")

    try:
        # Config 로드
        rvlm_config = RetinaVLMConfig.from_pretrained(
            "RobbieHolland/RetinaVLM", subfolder="RetinaVLM-Specialist"
        )
        rvlm_config.model.checkpoint_path = None

        # 모델 생성
        logger.info("[PROC] 새 RetinaVLM 인스턴스 생성...")
        model = RetinaVLM(rvlm_config)

        # 가중치 로드
        if save_dir.exists():
            weight_file = save_dir / "model.pt"
            if weight_file.exists():
                logger.info(f"[DOWN] 가중치 로드: {weight_file}")
                state_dict = torch.load(weight_file, map_location="cpu")
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                logger.info(f"[OK] 로드 완료 (missing={len(missing)}, unexpected={len(unexpected)})")
                del state_dict
            else:
                logger.warning(f"[WARN] 가중치 파일 없음: {weight_file}")
                logger.info("[INFO] HuggingFace에서 기본 모델 사용")
        else:
            logger.warning(f"[WARN] save_dir 없음: {save_dir}")
            logger.info("[INFO] HuggingFace에서 기본 모델 사용")

        # GPU 이동
        model.to(DEVICE)
        model.eval()

        logger.info(f"[DONE] RetinaVLM 준비 완료 ({DEVICE})")
        return model, DEVICE

    except Exception as e:
        logger.error(f"[ERR] 모델 로드 실패: {e}")
        raise


def two_step_biomarker_query(
    model: RetinaVLM,
    image_np: np.ndarray,
    biomarker: str,
    article: str,
    step1_tokens: int = 500,
    step2_tokens: int = 300
) -> Tuple[str, str, str]:
    """
    Task D 형식: 2-step biomarker present/absent 판단

    Returns:
        (step1_report, step2_conclusion, biomarker_status)
        biomarker_status: 'present' | 'not present' | 'unknown'
    """
    try:
        # 이미지 텐서 변환
        img_tensor = model.convert_any_image_to_normalized_tensor(image_np)
        param = next(model.model.parameters())
        images = torch.stack([img_tensor]).to(device=param.device, dtype=param.dtype)

        # Step 1: 상세 설명
        query = TASK_D_QUERY_TEMPLATE.format(biomarker=biomarker, article=article)

        step1_out = model.model.query(
            images, [query], answer_preamble=[''],
            max_new_tokens=step1_tokens, output_only=True
        )
        step1_text = step1_out[0]

        # Step 2: 결론
        step2_preamble = step1_text.strip() + "\n" + TASK_D_STEP2_TEMPLATE.format(
            biomarker=biomarker, article=article
        )

        step2_out = model.model.query(
            images, [query], answer_preamble=[step2_preamble],
            max_new_tokens=step2_tokens, output_only=True
        )
        step2_text = step2_out[0]

        full_text = step2_preamble + step2_text

        # 바이오마커 상태 추출
        status = extract_biomarker_status(full_text)

        return step1_text, step2_text, status

    except Exception as e:
        logger.warning(f"[WARN] 쿼리 실행 오류: {e}")
        return "", "", "unknown"


def extract_biomarker_status(text: str) -> str:
    """텍스트에서 바이오마커 상태 추출"""
    text_lower = text.lower()

    # 'not present'를 먼저 확인 (더 구체적)
    if "not present" in text_lower:
        return "not present"
    if "present" in text_lower:
        return "present"

    return "unknown"


def generate_biomarker_reports(
    model: RetinaVLM,
    image_paths: List[Path],
    output_dir: Path
) -> List[Dict]:
    """
    이미지들에 대해 바이오마커 생성 및 판단 수행
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    logger.info(f"[PROC] {len(image_paths)}개 이미지에 대해 바이오마커 분석 시작")
    logger.info(f"[INFO] 분석할 바이오마커: {len(BIOMARKER_SPECS)}개")

    for img_idx, img_path in enumerate(image_paths, 1):
        logger.info(f"\n[{img_idx}/{len(image_paths)}] {img_path.name}")

        try:
            # 이미지 로드
            img = Image.open(img_path).convert('L')
            img_np = np.array(img)

            # 클래스 추론
            relative_path = img_path.relative_to(DATA_DIR / 'OCT2017' / 'oct2017' / 'OCT2017')
            class_name = relative_path.parts[1] if len(relative_path.parts) > 1 else 'UNKNOWN'

            # 각 바이오마커에 대해 분석
            biomarker_results = []
            for biomarker, article in BIOMARKER_SPECS:
                logger.info(f"  → {biomarker}...")

                step1, step2, status = two_step_biomarker_query(
                    model, img_np, biomarker, article
                )

                biomarker_results.append({
                    "biomarker": biomarker,
                    "article": article,
                    "status": status,
                    "step1_text": step1[:200],  # 첫 200자만 저장
                    "step2_text": step2[:100]   # 첫 100자만 저장
                })

                logger.info(f"      ✓ {status}")

            # 결과 저장
            result_entry = {
                "image_name": img_path.name,
                "image_path": str(img_path.relative_to(DATA_DIR)),
                "class": class_name,
                "biomarkers": biomarker_results,
                "present_count": sum(1 for b in biomarker_results if b['status'] == 'present'),
                "absent_count": sum(1 for b in biomarker_results if b['status'] == 'not present'),
                "unknown_count": sum(1 for b in biomarker_results if b['status'] == 'unknown'),
            }
            results.append(result_entry)

            # 진행 상황 저장 (중간에 중단되어도 복구 가능)
            if img_idx % 10 == 0:
                save_results(results, output_dir)
                logger.info(f"[SAVE] {img_idx}개 완료, 임시 저장됨")

            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"[ERR] {img_path.name}: {e}")
            results.append({
                "image_name": img_path.name,
                "image_path": str(img_path.relative_to(DATA_DIR)),
                "class": "UNKNOWN",
                "error": str(e),
                "biomarkers": []
            })

    return results


def save_results(results: List[Dict], output_dir: Path) -> None:
    """결과를 JSON으로 저장"""
    output_file = output_dir / "biomarker_results.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    logger.info(f"[SAVE] {len(results)}개 결과 저장: {output_file}")


def generate_summary_report(results: List[Dict], output_dir: Path) -> Dict:
    """분석 결과 요약 보고서 생성"""
    summary = {
        "total_images": len(results),
        "successful": len([r for r in results if 'error' not in r]),
        "failed": len([r for r in results if 'error' in r]),
        "class_distribution": {},
        "biomarker_statistics": {},
        "sample_results": []
    }

    # 클래스별 분포
    for result in results:
        if 'error' not in result:
            cls = result.get('class', 'UNKNOWN')
            summary['class_distribution'][cls] = summary['class_distribution'].get(cls, 0) + 1

    # 바이오마커 통계
    all_biomarkers = {}
    for result in results:
        if 'biomarkers' in result and result['biomarkers']:
            for bm in result['biomarkers']:
                bm_name = bm['biomarker']
                if bm_name not in all_biomarkers:
                    all_biomarkers[bm_name] = {'present': 0, 'absent': 0, 'unknown': 0}
                all_biomarkers[bm_name][bm['status']] += 1

    summary['biomarker_statistics'] = all_biomarkers

    # 샘플 결과 3개
    summary['sample_results'] = [
        {
            'image': r['image_name'],
            'class': r['class'],
            'present': r.get('present_count', 0),
            'absent': r.get('absent_count', 0),
            'biomarkers_summary': [
                {'biomarker': bm['biomarker'], 'status': bm['status']}
                for bm in r.get('biomarkers', [])[:3]  # 첫 3개만
            ]
        }
        for r in results[:3]
    ]

    # 저장
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"분석 완료 요약")
    logger.info(f"{'='*60}")
    logger.info(f"총 이미지: {summary['total_images']}")
    logger.info(f"성공: {summary['successful']} | 실패: {summary['failed']}")
    logger.info(f"클래스 분포: {summary['class_distribution']}")
    logger.info(f"\n바이오마커 통계:")
    for bm, stats in all_biomarkers.items():
        total = sum(stats.values())
        pct = stats['present'] * 100 / total if total > 0 else 0
        logger.info(f"  {bm:40s} | Present: {stats['present']:3d} ({pct:5.1f}%) | Absent: {stats['absent']:3d}")
    logger.info(f"{'='*60}")

    return summary


def main(
    num_samples: int = 100,
    classes: Optional[List[str]] = None,
    seed: int = 42
):
    """메인 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if classes is None:
        classes = ["NORMAL", "CNV"]

    # 1. 모델 로드
    model, device = load_retinavlm_model(SAVE_DIR)

    # 2. 이미지 수집
    logger.info(f"\n[PROC] 데이터 수집: {classes} 클래스, 최대 {num_samples}개 샘플")

    oct_base = DATA_DIR / 'OCT2017' / 'oct2017' / 'OCT2017' / 'train'
    image_paths = []

    for cls in classes:
        cls_dir = oct_base / cls
        if cls_dir.exists():
            cls_images = list(cls_dir.glob('*.jpeg')) + list(cls_dir.glob('*.jpg'))
            sampled = random.sample(cls_images, min(len(cls_images), num_samples // len(classes)))
            image_paths.extend(sampled)
            logger.info(f"  {cls}: {len(sampled)}개")

    logger.info(f"총 {len(image_paths)}개 이미지 선택됨")

    # 3. 바이오마커 분석
    results = generate_biomarker_reports(model, image_paths, OUTPUT_DIR)

    # 4. 결과 저장 및 요약
    save_results(results, OUTPUT_DIR)
    summary = generate_summary_report(results, OUTPUT_DIR)

    logger.info(f"\n[DONE] 모든 작업 완료")
    logger.info(f"[INFO] 결과: {OUTPUT_DIR}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RetinaVLM 바이오마커 생성")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="생성할 샘플 수")
    parser.add_argument("--classes", nargs="+", default=["NORMAL", "CNV"],
                       help="분석할 클래스")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        num_samples=args.num_samples,
        classes=args.classes,
        seed=args.seed
    )
