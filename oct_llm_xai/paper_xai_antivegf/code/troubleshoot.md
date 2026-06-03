# Troubleshoot — paper_xai_antivegf/code

## sklearn ImportError (`cannot import name '__check_build'`)
**증상:** `conda run -n aptos2021 python ...` 시 `~/.local/lib/python3.10/site-packages/sklearn` 의 깨진 user-site 설치가 conda env 의 sklearn 을 가려 circular import 발생.

**우회:** 모든 실행에 `PYTHONNOUSERSITE=1` 를 붙인다.
```bash
PYTHONNOUSERSITE=1 conda run -n aptos2021 python <script>.py
```
(영구 수정은 `pip uninstall -y scikit-learn --user` 로 깨진 user-site 제거 권장 — 미적용.)

## RetinaVLM(Tier3, oct_llm env) — `libcusparse.so.12: undefined symbol __nvJitLinkCreate_12_8`
**증상:** `conda run -n oct_llm python infer.py --tier tier3 ...` 시 `import torch` 단계에서 cusparse가 더 신버전 nvJitLink 심볼을 못 찾아 ImportError. (Jupyter는 LD_LIBRARY_PATH가 이미 잡혀 재현 안 됨.)
**우회:** nvjitlink site-packages lib을 `LD_LIBRARY_PATH` 앞에 export.
```bash
export LD_LIBRARY_PATH=/home/ubuntu/bionexus/jgy/miniconda3/envs/oct_llm/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n oct_llm python infer.py --tier tier3 --variant Z0
```

## `conda run -n <env>` 캡처 모드에서 `ModuleNotFoundError: No module named 'numpy'`
**증상:** `conda run -n vllm python ...`(기본 캡처 모드)에서 numpy가 설치돼 있는데도 import 실패.
**우회:** `--no-capture-output` 플래그 추가 → 정상 환경 활성화.
```bash
conda run --no-capture-output -n vllm python infer.py --tier tier1 ...
```

## 환경
- 채점/데이터(`metrics.py`,`data.py`): `aptos2021` env + `PYTHONNOUSERSITE=1`.
- VLM 추론(`models.py`):
  - **Tier3 RetinaVLM** → `oct_llm` env + 위 nvjitlink `LD_LIBRARY_PATH` 우회. dequantized fp16 ckpt(`SpecialistVLMs/saved_models/RetinaVLM-Specialist-Dequantized/model.pt`, 16GB) 사용. (PYTHONNOUSERSITE 금지 — user-site httpx 필요.) 모델 로드 ~40s, 추론 ~1.5s/(eye·prompt), 전체 36 eyes×4프롬프트 ≈ 247s.
  - **Tier1 Qwen3-VL-8B-Instruct** → `vllm` env (transformers 4.57.1, `Qwen3VLForConditionalGeneration` 지원) + `--no-capture-output`. 가중치 ~16GB 최초 1회 다운로드(완료).
  - **Tier2 LLaVA-Med** → 외부 llava lib 필요, 미실행.
