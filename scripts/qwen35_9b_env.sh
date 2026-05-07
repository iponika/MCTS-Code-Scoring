#!/usr/bin/env bash
# Source this file before running Qwen3.5-9B/vLLM jobs on the shared server.
# It keeps large caches and the isolated conda environment outside /home.
set -euo pipefail

export MCTS_QWEN35_ROOT="${MCTS_QWEN35_ROOT:-/data1/ruiqi}"
export MCTS_QWEN35_ENV="${MCTS_QWEN35_ENV:-${MCTS_QWEN35_ROOT}/conda_envs/mcts-vllm-qwen35}"

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${MCTS_QWEN35_ROOT}/conda_pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${MCTS_QWEN35_ROOT}/pip_cache}"
export HF_HOME="${HF_HOME:-${MCTS_QWEN35_ROOT}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${MCTS_QWEN35_ROOT}/hf_datasets_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${MCTS_QWEN35_ROOT}/xdg_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${MCTS_QWEN35_ROOT}/uv_cache}"
export TMPDIR="${TMPDIR:-/tmp}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p \
  "${CONDA_PKGS_DIRS}" \
  "${PIP_CACHE_DIR}" \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${UV_CACHE_DIR}"

if [[ -d "${MCTS_QWEN35_ENV}" ]]; then
  # shellcheck disable=SC1091
  source /home/ruiqi/miniconda3/etc/profile.d/conda.sh
  conda activate "${MCTS_QWEN35_ENV}"
fi
