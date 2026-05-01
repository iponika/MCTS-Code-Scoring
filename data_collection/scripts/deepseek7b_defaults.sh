#!/usr/bin/env bash
# Shared defaults for the current review-scoring experiments.
# Prefer the local snapshot on this server, but keep the Hugging Face id as a
# portable fallback for fresh clones.

DEEPSEEK7B_MODEL_ID="${DEEPSEEK7B_MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
DEEPSEEK7B_LOCAL_SNAPSHOT="${DEEPSEEK7B_LOCAL_SNAPSHOT:-/data1/xianzhiwei/model/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60}"

if [[ -z "${MODEL_KEY:-}" ]]; then
  export MODEL_KEY="${DEEPSEEK7B_MODEL_ID}"
fi

if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -d "${DEEPSEEK7B_LOCAL_SNAPSHOT}" ]]; then
    export MODEL_PATH="${DEEPSEEK7B_LOCAL_SNAPSHOT}"
  else
    export MODEL_PATH="${DEEPSEEK7B_MODEL_ID}"
  fi
fi
