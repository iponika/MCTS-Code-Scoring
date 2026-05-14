#!/usr/bin/env bash
# Source this file before running local Qwen3-4B/vLLM jobs on this server.
set -euo pipefail

export MCTS_QWEN3_4B_ROOT="${MCTS_QWEN3_4B_ROOT:-/data1/xianzhiwei}"
export HF_HOME="${HF_HOME:-${MCTS_QWEN3_4B_ROOT}/model/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="/tmp/hf-datasets-cache"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export XDG_CACHE_HOME="/tmp/xdg-cache"
export UV_CACHE_DIR="/tmp/uv-cache"
export TMPDIR="${TMPDIR:-/tmp}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

mkdir -p \
  "${HF_DATASETS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${UV_CACHE_DIR}"
