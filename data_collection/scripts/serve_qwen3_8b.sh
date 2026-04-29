#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"

export CUDA_VISIBLE_DEVICES
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

exec vllm serve "${MODEL_PATH}" \
  --host "${VLLM_HOST}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code
