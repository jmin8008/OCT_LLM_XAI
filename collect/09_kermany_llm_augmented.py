"""
Kermany OCT + LLM Augmented Reports
===================================
가장 추천되는 현실적 대안
- 베이스 데이터: 84,495개 Kermany OCT 이미지 (4개 클래스)
- 증강 방식: LLM 프롬프팅으로 다양한 의료 소견서 생성
- LLM 옵션: GPT-4 API, Claude API, 또는 오픈소스 Llama-3
- 목표: 노이즈 없는 고품질 훈련 데이터 생성

사전 요구사항:
    pip install kaggle openai anthropic torch transformers

    GPT-4: OpenAI API 키 필요 (OPENAI_API_KEY 환경변수)
    Claude: Anthropic API 키 필요 (ANTHROPIC_API_KEY 환경변수)
    Llama: ollama 또는 로컬 모델 서버 필요

사용법:
    # Kermany 데이터 기반 다운로드
    python 09_kermany_llm_augmented.py --download-base

    # LLM으로 소견서 생성 (OpenAI)
    python 09_kermany_llm_augmented.py --generate-with openai

    # LLM으로 소견서 생성 (Anthropic Claude)
    python 09_kermany_llm_augmented.py --generate-with claude

    # 로컬 Llama 모델로 생성
    python 09_kermany_llm_augmented.py --generate-with llama
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging
import subprocess
import time

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_DIR = "Kermany_LLM_Augmented"

# 질병별 프롬프트 템플릿
DISEASE_PROMPTS = {
    "CNV": {
        "name": "Choroidal Neovascularization",
        "description": "abnormal blood vessel growth beneath the retina",
        "key_findings": ["subretinal fluid", "hyperreflective material", "choroidal thickening"]
    },
    "DME": {
        "name": "Diabetic Macular Edema",
        "description": "macular swelling due to diabetes-related fluid accumulation",
        "key_findings": ["intraretinal fluid", "hard exudates", "macular thickening"]
    },
    "DRUSEN": {
        "name": "Drusen",
        "description": "yellow deposits under the retina indicating age-related macular degeneration",
        "key_findings": ["drusen", "RPE irregularity", "macular thinning"]
    },
    "NORMAL": {
        "name": "Normal",
        "description": "healthy retinal structure without significant pathology",
        "key_findings": ["normal retinal thickness", "intact ellipsoid zone", "intact RPE"]
    }
}


def download_kermany_base(data_dir: Path) -> Path:
    """Kermany OCT 기본 데이터 다운로드"""
    from pathlib import Path

    try:
        import subprocess
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "paultimothymooney/kermany2018",
             "--unzip", "-p", str(data_dir)],
            check=True
        )
        logger.info("[DONE] Kermany OCT 데이터 다운로드 완료")
        return data_dir / "OCT2017"
    except Exception as e:
        logger.error(f"[ERR] Kermany 다운로드 실패: {e}")
        raise


def create_dataset_structure(base_dir: Path) -> Path:
    """증강 데이터셋 구조 생성"""
    aug_dir = base_dir / DATASET_DIR
    (aug_dir / "images").mkdir(parents=True, exist_ok=True)
    (aug_dir / "reports").mkdir(parents=True, exist_ok=True)
    (aug_dir / "metadata").mkdir(parents=True, exist_ok=True)

    return aug_dir


def generate_with_openai(
    class_label: str,
    num_variations: int = 10,
    api_key: Optional[str] = None
) -> List[str]:
    """OpenAI GPT-4 API로 소견서 생성"""

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경변수 필요")

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("[ERR] openai 설치 필요: pip install openai")
        raise

    client = OpenAI(api_key=api_key)
    disease_info = DISEASE_PROMPTS.get(class_label, {})
    disease_name = disease_info.get("name", class_label)
    key_findings = ", ".join(disease_info.get("key_findings", []))

    prompt = f"""You are an ophthalmologist expert in OCT interpretation.
Generate {num_variations} different professional OCT reports for a case with {disease_name}.

Requirements:
- Each report should be 3-4 sentences long
- Include anatomical locations (macula, retina, RPE, choroid)
- Include specific findings: {key_findings}
- Vary the expression and phrasing significantly
- Use medical terminology appropriate for peer-reviewed journals
- Do NOT repeat findings exactly

Output format: Return a JSON array of strings, one report per element.
[
  "Report 1 here...",
  "Report 2 here...",
  ...
]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=1500
        )

        response_text = response.choices[0].message.content
        reports = json.loads(response_text)
        logger.info(f"[DONE] OpenAI: {disease_name}에 대해 {len(reports)}개 소견서 생성")
        return reports

    except json.JSONDecodeError:
        logger.warning(f"[WARN] JSON 파싱 실패: {response_text[:100]}")
        return []
    except Exception as e:
        logger.error(f"[ERR] OpenAI API 호출 실패: {e}")
        raise


def generate_with_claude(
    class_label: str,
    num_variations: int = 10,
    api_key: Optional[str] = None
) -> List[str]:
    """Anthropic Claude API로 소견서 생성"""

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY 환경변수 필요")

    try:
        from anthropic import Anthropic
    except ImportError:
        logger.error("[ERR] anthropic 설치 필요: pip install anthropic")
        raise

    client = Anthropic(api_key=api_key)
    disease_info = DISEASE_PROMPTS.get(class_label, {})
    disease_name = disease_info.get("name", class_label)
    key_findings = ", ".join(disease_info.get("key_findings", []))

    prompt = f"""You are an ophthalmologist expert in OCT interpretation.
Generate {num_variations} different professional OCT reports for a case with {disease_name}.

Requirements:
- Each report should be 3-4 sentences long
- Include anatomical locations (macula, retina, RPE, choroid)
- Include specific findings: {key_findings}
- Vary the expression and phrasing significantly
- Use medical terminology appropriate for peer-reviewed journals
- Do NOT repeat findings exactly

Output format: Return ONLY a JSON array of strings (no markdown, no extra text):
[
  "Report 1 here...",
  "Report 2 here...",
  ...
]
"""

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text
        # JSON 추출 (마크다운 블록 제거)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        reports = json.loads(response_text.strip())
        logger.info(f"[DONE] Claude: {disease_name}에 대해 {len(reports)}개 소견서 생성")
        return reports

    except json.JSONDecodeError as e:
        logger.warning(f"[WARN] JSON 파싱 실패: {e}")
        logger.info(f"[INFO] 응답: {response_text[:200]}")
        return []
    except Exception as e:
        logger.error(f"[ERR] Claude API 호출 실패: {e}")
        raise


def generate_with_llama(
    class_label: str,
    num_variations: int = 10,
    server_url: str = "http://localhost:11434"
) -> List[str]:
    """로컬 Llama 모델로 소견서 생성 (Ollama)"""

    try:
        import requests
    except ImportError:
        logger.error("[ERR] requests 설치 필요: pip install requests")
        raise

    disease_info = DISEASE_PROMPTS.get(class_label, {})
    disease_name = disease_info.get("name", class_label)
    key_findings = ", ".join(disease_info.get("key_findings", []))

    prompt = f"""You are an ophthalmologist expert in OCT interpretation.
Generate {num_variations} different professional OCT reports for {disease_name}.

Requirements:
- Each report 3-4 sentences
- Include anatomical terms and specific findings
- Key findings to cover: {key_findings}
- Vary phrasing significantly

Return only JSON array of strings:
[
  "Report 1...",
  "Report 2...",
  ...
]
"""

    try:
        # Ollama API 호출
        response = requests.post(
            f"{server_url}/api/generate",
            json={
                "model": "llama2",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.8
            },
            timeout=120
        )

        if response.status_code != 200:
            logger.error(f"[ERR] Llama 모델 오류: {response.status_code}")
            return []

        response_text = response.json()["response"]
        reports = json.loads(response_text)
        logger.info(f"[DONE] Llama: {disease_name}에 대해 {len(reports)}개 소견서 생성")
        return reports

    except json.JSONDecodeError:
        logger.warning("[WARN] Llama 응답 파싱 실패")
        return []
    except requests.exceptions.ConnectionError:
        logger.error(f"[ERR] Ollama 서버 연결 실패 ({server_url})")
        logger.info("[INFO] Ollama 설치: https://ollama.ai")
        raise
    except Exception as e:
        logger.error(f"[ERR] Llama 생성 실패: {e}")
        raise


def augment_dataset(
    data_dir: Path,
    llm_provider: str = "claude",
    reports_per_class: int = 100
) -> Path:
    """
    Kermany 데이터를 기반으로 LLM 증강 소견서 생성 및 저장
    """

    aug_dir = create_dataset_structure(data_dir)
    reports_file = aug_dir / "augmented_reports.jsonl"

    all_reports = []

    for disease_class, disease_info in DISEASE_PROMPTS.items():
        logger.info(f"[PROC] {disease_class} ({disease_info['name']})에 대해 {reports_per_class}개 보고서 생성 중...")

        # LLM 공급자별로 생성
        try:
            if llm_provider == "openai":
                reports = generate_with_openai(disease_class, num_variations=reports_per_class)
            elif llm_provider == "claude":
                reports = generate_with_claude(disease_class, num_variations=reports_per_class)
            elif llm_provider == "llama":
                reports = generate_with_llama(disease_class, num_variations=reports_per_class)
            else:
                logger.error(f"[ERR] 지원되지 않는 LLM 공급자: {llm_provider}")
                continue

            for idx, report in enumerate(reports):
                entry = {
                    "disease_class": disease_class,
                    "disease_name": disease_info["name"],
                    "report_text": report,
                    "report_idx": idx,
                    "llm_provider": llm_provider,
                    "key_findings": disease_info["key_findings"]
                }
                all_reports.append(entry)

        except Exception as e:
            logger.error(f"[ERR] {disease_class} 보고서 생성 실패: {e}")
            continue

        # API 속도 제한 고려
        if llm_provider in ["openai", "claude"]:
            time.sleep(1)

    # 모든 보고서를 JSONL로 저장
    with open(reports_file, 'w', encoding='utf-8') as f:
        for entry in all_reports:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logger.info(f"[DONE] 총 {len(all_reports)}개 증강 소견서 저장 → {reports_file}")

    # 통계 저장
    stats = {
        "total_reports": len(all_reports),
        "reports_per_class": reports_per_class,
        "llm_provider": llm_provider,
        "diseases": list(DISEASE_PROMPTS.keys()),
        "output_file": str(reports_file)
    }

    stats_file = aug_dir / "augmentation_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    return reports_file


def main(
    data_dir: Path = DATA_DIR,
    download_base: bool = False,
    generate_with: Optional[str] = None,
    reports_per_class: int = 100
):
    """Kermany OCT + LLM 증강 파이프라인"""

    if download_base:
        logger.info("[PROC] Kermany OCT 기본 데이터 다운로드 중...")
        download_kermany_base(data_dir)

    if generate_with:
        logger.info(f"[PROC] LLM 증강 보고서 생성 ({generate_with}) 중...")
        augment_dataset(data_dir, llm_provider=generate_with, reports_per_class=reports_per_class)

    aug_dir = data_dir / DATASET_DIR
    logger.info(f"\n[INFO] Kermany LLM 증강 데이터: {aug_dir}")
    if (aug_dir / "augmented_reports.jsonl").exists():
        logger.info(f"[INFO] 증강 보고서: {aug_dir}/augmented_reports.jsonl")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kermany OCT + LLM Augmented Reports")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--download-base", action="store_true",
                       help="Kermany OCT 기본 데이터 다운로드")
    parser.add_argument("--generate-with", choices=["openai", "claude", "llama"], default=None,
                       help="LLM 제공자 선택")
    parser.add_argument("--reports-per-class", type=int, default=100,
                       help="클래스당 생성할 보고서 수")

    args = parser.parse_args()

    main(
        Path(args.data_dir) if args.data_dir else DATA_DIR,
        download_base=args.download_base,
        generate_with=args.generate_with,
        reports_per_class=args.reports_per_class
    )
