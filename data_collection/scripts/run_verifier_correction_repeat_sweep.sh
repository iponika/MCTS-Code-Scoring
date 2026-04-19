#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/xianzhiwei/mcts-code-review"
RUN_NAME="verifier_correction_repeat_sweep_20260419"
MODEL_PATH="/data1/xianzhiwei/model/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
WAIT_SESSION="${WAIT_SESSION:-verifier_correction_overnight_20260419}"

RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
EVAL_ROOT="${ROOT}/model_training/src/output/review-eval-${RUN_NAME}"
INDICES_FILE="${ROOT}/data_collection/review_mcts_runs/verifier_correction_overnight_20260419/heldout_indices.json"

INPUTS=(
  "${ROOT}/data_collection/review_mcts_runs/evidence_style_smoke4_v3_20260419/samples"
  "${ROOT}/data_collection/review_mcts_runs/supervised_medium_20260418/samples"
)

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${EVAL_ROOT}"
LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

NOTIFY_URL="${NTFY_URL:-https://ntfy.sh/iponika_mcts}"
NOTIFY_RESPONSE="${RUN_DIR}/notify_response.json"
NOTIFY_ERROR="${RUN_DIR}/notify_error.log"
CURRENT_STAGE="init"

notify() {
  local status="$1"
  local message="${RUN_NAME} ${status}: stage=${CURRENT_STAGE}, host=$(hostname), time=$(date -Is), log=${LOG_FILE}"
  if curl -fsS -d "${message}" "${NOTIFY_URL}" >"${NOTIFY_RESPONSE}" 2>"${NOTIFY_ERROR}"; then
    echo "[notify] sent ${status}"
  else
    echo "[notify] failed ${status}; see ${NOTIFY_ERROR}" >&2
  fi
}

on_exit() {
  local code=$?
  if [[ "${CURRENT_STAGE}" != "finished" ]]; then
    notify "FAILED(code=${code})"
  fi
  exit "${code}"
}
trap on_exit EXIT

wait_for_previous() {
  CURRENT_STAGE="wait_${WAIT_SESSION}"
  while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
    echo "[stage:${CURRENT_STAGE}] waiting for ${WAIT_SESSION} at $(date -Is)"
    sleep 300
  done
  echo "[stage:${CURRENT_STAGE}] previous session absent at $(date -Is)"
}

convert_one() {
  local repeat="$1"
  local data_file="$2"
  CURRENT_STAGE="convert_repeat${repeat}"
  if [[ -f "${data_file}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${data_file}"
    return
  fi
  cd "${ROOT}"
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${ROOT}/model_training/src" \
  HF_HUB_OFFLINE=1 \
  UV_CACHE_DIR=/tmp/uv-cache \
  uv run python -m magicoder.preprocess_review_mcts_data \
    --input "${INPUTS[@]}" \
    --output_file "${data_file}" \
    --policy_min_q 0.5 \
    --max_value_paths_per_dimension 0 \
    --emit_verifier_corrections \
    --verifier_correction_repeat "${repeat}" \
    --verifier_correction_q 0.8 \
    --verifier_correction_lm_weight 0.5 \
    --verifier_correction_value_weight 0.5 \
    --shuffle_seed 20260419
  wc -l "${data_file}"
}

train_one() {
  local repeat="$1"
  local cuda_device="$2"
  local data_file="$3"
  local out_dir="$4"
  local log_file="${LOG_DIR}/train_repeat${repeat}.log"

  if [[ -f "${out_dir}/adapter_model.safetensors" && -f "${out_dir}/value_head.pth" ]]; then
    echo "[stage:train_repeat${repeat}] final checkpoint exists, skipping: ${out_dir}"
    return
  fi

  (
    set -euo pipefail
    cd "${ROOT}/model_training/src"
    RESUME_ARGS=()
    LATEST_CHECKPOINT="$(find "${out_dir}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
    if [[ -n "${LATEST_CHECKPOINT}" ]]; then
      RESUME_ARGS=(--resume_from_checkpoint "${LATEST_CHECKPOINT}")
      echo "[stage:train_repeat${repeat}] resuming from ${LATEST_CHECKPOINT}"
    fi
    echo "[stage:train_repeat${repeat}] start=$(date -Is) cuda=${cuda_device}"
    CUDA_VISIBLE_DEVICES="${cuda_device}" \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_DATASETS_CACHE=/tmp/hf-datasets-cache \
    HF_HUB_OFFLINE=1 \
    TRL_EXPERIMENTAL_SILENCE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.train_multi \
      --task review \
      --model_key Qwen/Qwen3.5-9B \
      --model_name_or_path "${MODEL_PATH}" \
      --datafile_paths "../review_mcts_train_data/$(basename "${data_file}")" \
      --output_dir "${out_dir}" \
      --max_steps 120 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --max_training_seq_length 1024 \
      --bf16 True \
      --logging_steps 10 \
      --save_strategy steps \
      --save_steps 60 \
      --save_total_limit 2 \
      --report_to none \
      --optim adafactor \
      --learning_rate 5e-5 \
      --lr_scheduler_type linear \
      --warmup_steps 12 \
      --peft lora \
      --value_weight 0.025 \
      --num_proc 1 \
      "${RESUME_ARGS[@]}"
    echo "[stage:train_repeat${repeat}] done=$(date -Is)"
  ) >"${log_file}" 2>&1 &
}

eval_one() {
  local repeat="$1"
  local cuda_device="$2"
  local out_dir="$3"
  local eval_dir="${EVAL_ROOT}/correction_repeat${repeat}"
  local log_file="${LOG_DIR}/eval_repeat${repeat}.log"
  mkdir -p "${eval_dir}"

  (
    set -euo pipefail
    cd "${ROOT}/model_training/src"
    echo "[stage:eval_repeat${repeat}] start=$(date -Is) cuda=${cuda_device}"
    CUDA_VISIBLE_DEVICES="${cuda_device}" \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_DATASETS_CACHE=/tmp/hf-datasets-cache \
    HF_HUB_OFFLINE=1 \
    TRL_EXPERIMENTAL_SILENCE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.batch_review_evaluator \
      --policy_model_path "${out_dir}" \
      --value_model_path "${out_dir}" \
      --share_policy_value_model \
      --input_record ../../datasets/CodeCriticBench/data/CodeCriticBench.jsonl \
      --record_indices_file "${INDICES_FILE}" \
      --output_dir "${eval_dir}" \
      --dimensions "Correctness Verification" \
      --device cuda \
      --dtype bf16 \
      --max_steps 3 \
      --num_candidates 1 \
      --max_new_tokens 256 \
      --final_max_new_tokens 384 \
      --temperature 0.7 \
      --top_p 0.95 \
      --rethink_threshold -0.2 \
      --max_rethinks 1 \
      --max_final_retries 2
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${ROOT}/model_training/src:${ROOT}/data_collection" \
    HF_HUB_OFFLINE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python "${ROOT}/data_collection/scripts/summarize_review_eval_outputs.py" \
      --eval_dir "${eval_dir}" \
      --output "${eval_dir}/summary.json"
    echo "[stage:eval_repeat${repeat}] done=$(date -Is)"
  ) >"${log_file}" 2>&1 &
}

run_parallel_train_eval() {
  local data1="${ROOT}/model_training/review_mcts_train_data/verifier_sweep_correction_repeat1_20260419.jsonl"
  local data4="${ROOT}/model_training/review_mcts_train_data/verifier_sweep_correction_repeat4_20260419.jsonl"
  local out1="${ROOT}/model_training/src/output/review-lora-verifier-sweep-correction-repeat1-120step"
  local out4="${ROOT}/model_training/src/output/review-lora-verifier-sweep-correction-repeat4-120step"

  convert_one 1 "${data1}"
  convert_one 4 "${data4}"

  CURRENT_STAGE="train_parallel"
  train_one 1 0 "${data1}" "${out1}"
  pid1=$!
  train_one 4 1 "${data4}" "${out4}"
  pid4=$!
  echo "[stage:${CURRENT_STAGE}] waiting repeat1 pid=${pid1}, repeat4 pid=${pid4}"
  wait "${pid1}"
  wait "${pid4}"

  CURRENT_STAGE="eval_parallel"
  eval_one 1 0 "${out1}"
  pid1=$!
  eval_one 4 1 "${out4}"
  pid4=$!
  echo "[stage:${CURRENT_STAGE}] waiting repeat1 pid=${pid1}, repeat4 pid=${pid4}"
  wait "${pid1}"
  wait "${pid4}"
}

echo "[pipeline] start=$(date -Is)"
echo "[pipeline] log=${LOG_FILE}"
wait_for_previous
if [[ ! -f "${INDICES_FILE}" ]]; then
  echo "Missing indices file: ${INDICES_FILE}" >&2
  exit 1
fi
run_parallel_train_eval

CURRENT_STAGE="finished"
echo "[pipeline] finished=$(date -Is)"
notify "COMPLETED"
