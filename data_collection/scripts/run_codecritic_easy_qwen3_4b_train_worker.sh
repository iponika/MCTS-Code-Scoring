#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

RUN_NAME="${RUN_NAME:-qwen3_4b_codecritic_easy_correctness_20260513}"
TAG="${TAG:?Set TAG, e.g. static or mcts}"
DATA_FILE="${DATA_FILE:?Set DATA_FILE to a JSONL under model_training/review_mcts_train_data}"
GPU_ID="${GPU_ID:?Set GPU_ID, e.g. 0}"

MODEL_KEY="${MODEL_KEY:-Qwen/Qwen3-4B}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
MAX_STEPS="${MAX_STEPS:-240}"
MAX_TRAINING_SEQ_LENGTH="${MAX_TRAINING_SEQ_LENGTH:-2048}"

LOG_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}/logs"
OUTPUT_ROOT="${ROOT}/model_training/src/output"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/review-lora-${RUN_NAME}-${TAG}-${MAX_STEPS}step}"
LOG_FILE="${LOG_DIR}/train_${TAG}.log"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

cd "${ROOT}"
source "${ROOT}/scripts/qwen3_4b_env.sh"

echo "[train-worker] host=$(hostname) time=$(date -Is)"
echo "[train-worker] tag=${TAG} gpu=${GPU_ID}"
echo "[train-worker] data_file=${DATA_FILE}"
echo "[train-worker] output_dir=${OUTPUT_DIR}"
echo "[train-worker] max_steps=${MAX_STEPS} max_training_seq_length=${MAX_TRAINING_SEQ_LENGTH}"

cd "${ROOT}/model_training/src"
resume_args=()
latest_checkpoint="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -n "${latest_checkpoint}" ]]; then
  resume_args=(--resume_from_checkpoint "${latest_checkpoint}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTHONDONTWRITEBYTECODE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_DATASETS_CACHE=/tmp/hf-datasets-cache \
HF_HUB_OFFLINE=1 \
TRL_EXPERIMENTAL_SILENCE=1 \
UV_CACHE_DIR=/tmp/uv-cache \
uv run python -m magicoder.train_multi \
  --task review \
  --model_key "${MODEL_KEY}" \
  --model_name_or_path "${MODEL_PATH}" \
  --datafile_paths "../review_mcts_train_data/$(basename "${DATA_FILE}")" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_training_seq_length "${MAX_TRAINING_SEQ_LENGTH}" \
  --bf16 True \
  --logging_steps 20 \
  --save_strategy steps \
  --save_steps 80 \
  --save_total_limit 2 \
  --report_to none \
  --optim adafactor \
  --learning_rate 3e-5 \
  --lr_scheduler_type linear \
  --warmup_steps 24 \
  --peft lora \
  --value_weight 0.05 \
  --boundary_value_weight 0.02 \
  --pairwise_value_weight 0.0 \
  --disable_train_shuffle True \
  --train_sampling_strategy sequential \
  --num_proc 1 \
  --seed 20260513 \
  "${resume_args[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
