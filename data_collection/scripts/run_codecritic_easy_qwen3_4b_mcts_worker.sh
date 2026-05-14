#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

RUN_NAME="${RUN_NAME:-qwen3_4b_codecritic_easy_correctness_20260513}"
CFG="${CFG:-${ROOT}/data_collection/configs/mcts_codecritic_qwen3_4b_no_think.yaml}"
SEED_DATA="${SEED_DATA:-${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}/seed_train.jsonl}"
RUN_DIR="${RUN_DIR:-${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}}"

GPU_ID="${GPU_ID:?Set GPU_ID, e.g. 0}"
START="${START:?Set START, e.g. 0}"
LIMIT="${LIMIT:?Set LIMIT, e.g. 254}"
SHARD_NAME="${SHARD_NAME:-gpu${GPU_ID}_${START}_${LIMIT}}"

LOG_DIR="${RUN_DIR}/logs"
OUTPUT="${RUN_DIR}/mcts_${SHARD_NAME}.jsonl"
OUTPUT_DIR="${RUN_DIR}/mcts_samples_${SHARD_NAME}"
LOG_FILE="${LOG_DIR}/mcts_${SHARD_NAME}.log"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

cd "${ROOT}"
source "${ROOT}/scripts/qwen3_4b_env.sh"

echo "[mcts-worker] host=$(hostname) time=$(date -Is)"
echo "[mcts-worker] gpu=${GPU_ID} start=${START} limit=${LIMIT}"
echo "[mcts-worker] cfg=${CFG}"
echo "[mcts-worker] seed_data=${SEED_DATA}"
echo "[mcts-worker] output=${OUTPUT}"
echo "[mcts-worker] output_dir=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTHONPATH="${ROOT}/data_collection" \
uv run python data_collection/solver_review.py \
  --custom_cfg "${CFG}" \
  --dataset "${SEED_DATA}" \
  --start "${START}" \
  --limit "${LIMIT}" \
  --output "${OUTPUT}" \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee -a "${LOG_FILE}"
