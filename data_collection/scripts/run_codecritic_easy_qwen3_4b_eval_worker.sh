#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

RUN_NAME="${RUN_NAME:-qwen3_4b_codecritic_easy_correctness_20260513}"
TAG="${TAG:?Set TAG, e.g. static or mcts}"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to trained adapter directory}"
GPU_ID="${GPU_ID:?Set GPU_ID, e.g. 0}"

EVAL_DATA="${EVAL_DATA:-${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}/seed_eval.jsonl}"
INDICES_FILE="${INDICES_FILE:-${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}/eval_indices.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/model_training/src/output/review-eval-${RUN_NAME}-${TAG}-codecritic_easy}"
LOG_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}/logs"
LOG_FILE="${LOG_DIR}/eval_${TAG}.log"

case "${MODEL_DIR}" in
  /*) ;;
  *) MODEL_DIR="${ROOT}/${MODEL_DIR}" ;;
esac
case "${EVAL_DATA}" in
  /*) ;;
  *) EVAL_DATA="${ROOT}/${EVAL_DATA}" ;;
esac
case "${INDICES_FILE}" in
  /*) ;;
  *) INDICES_FILE="${ROOT}/${INDICES_FILE}" ;;
esac
case "${OUTPUT_DIR}" in
  /*) ;;
  *) OUTPUT_DIR="${ROOT}/${OUTPUT_DIR}" ;;
esac

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

cd "${ROOT}"
source "${ROOT}/scripts/qwen3_4b_env.sh"

echo "[eval-worker] host=$(hostname) time=$(date -Is)"
echo "[eval-worker] tag=${TAG} gpu=${GPU_ID}"
echo "[eval-worker] model_dir=${MODEL_DIR}"
echo "[eval-worker] eval_data=${EVAL_DATA}"
echo "[eval-worker] output_dir=${OUTPUT_DIR}"

cd "${ROOT}/model_training/src"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTHONDONTWRITEBYTECODE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_DATASETS_CACHE=/tmp/hf-datasets-cache \
HF_HUB_OFFLINE=1 \
UV_CACHE_DIR=/tmp/uv-cache \
python -m magicoder.batch_review_evaluator \
  --policy_model_path "${MODEL_DIR}" \
  --value_model_path "${MODEL_DIR}" \
  --share_policy_value_model \
  --input_record "${EVAL_DATA}" \
  --record_indices_file "${INDICES_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --dimensions "Correctness Verification" \
  --device cuda \
  --dtype bf16 \
  --max_steps 1 \
  --reasoning_steps 0 \
  --num_candidates 1 \
  --max_new_tokens 128 \
  --final_max_new_tokens 512 \
  --temperature 0 \
  --top_p 1.0 \
  --final_only_json \
  --prompt_variant codecritic_correctness \
  --show_tests_in_prompt \
  --chat_template_enable_thinking false \
  --max_final_retries 1 \
  --use_chat_template \
  2>&1 | tee -a "${LOG_FILE}"
