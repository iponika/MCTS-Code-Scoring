#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/qwen35_9b_env.sh
source "${ROOT}/scripts/qwen35_9b_env.sh"

if [[ ! -d "${MCTS_QWEN35_ENV}" ]]; then
  source /home/ruiqi/miniconda3/etc/profile.d/conda.sh
  conda create --yes --prefix "${MCTS_QWEN35_ENV}" --clone /home/ruiqi/miniconda3
fi

source /home/ruiqi/miniconda3/etc/profile.d/conda.sh
conda activate "${MCTS_QWEN35_ENV}"

VLLM_VERSION="${VLLM_VERSION:-0.19.1}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=4.55.2}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade --only-binary=:all: \
  "vllm==${VLLM_VERSION}" \
  "accelerate>=1.0.0" \
  "datasets>=2.21.0" \
  "peft>=0.13.0" \
  "trl>=0.15.2" \
  "${TRANSFORMERS_SPEC}" \
  "numpy==2.2.6"

python - <<'PY'
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
import torch
import transformers
import vllm

print("python_env_ok")
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__, "qwen3_5", "qwen3_5" in CONFIG_MAPPING_NAMES)
print("vllm", vllm.__version__)
if "qwen3_5" not in CONFIG_MAPPING_NAMES:
    raise SystemExit("Transformers still does not recognize model_type=qwen3_5")
PY
