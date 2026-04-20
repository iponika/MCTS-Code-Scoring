#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/xianzhiwei/mcts-code-review"
RUN_NAME="verifier_correction_filtered1536_20260420"
MODEL_PATH="/data1/xianzhiwei/model/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"

RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
EVAL_ROOT="${ROOT}/model_training/src/output/review-eval-${RUN_NAME}"
INDICES_FILE="${ROOT}/data_collection/review_mcts_runs/verifier_correction_overnight_20260419/heldout_indices.json"

BASE_RAW="${ROOT}/model_training/review_mcts_train_data/verifier_filtered1536_baseline_raw_20260420.jsonl"
CORR_RAW="${ROOT}/model_training/review_mcts_train_data/verifier_filtered1536_correction_weak_raw_20260420.jsonl"
BASE_DATA="${ROOT}/model_training/review_mcts_train_data/verifier_filtered1536_baseline_20260420.jsonl"
CORR_DATA="${ROOT}/model_training/review_mcts_train_data/verifier_filtered1536_correction_weak_20260420.jsonl"
BASE_OUT="${ROOT}/model_training/src/output/review-lora-verifier-filtered1536-baseline-160step"
CORR_OUT="${ROOT}/model_training/src/output/review-lora-verifier-filtered1536-correction-weak-160step"

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
  if env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    curl --noproxy '*' -fsS -d "${message}" "${NOTIFY_URL}" >"${NOTIFY_RESPONSE}" 2>"${NOTIFY_ERROR}"; then
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

convert_and_filter() {
  CURRENT_STAGE="convert_filter"
  cd "${ROOT}"
  if [[ ! -f "${BASE_RAW}" ]]; then
    echo "[stage:${CURRENT_STAGE}] baseline raw start=$(date -Is)"
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${ROOT}/model_training/src" \
    HF_HUB_OFFLINE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.preprocess_review_mcts_data \
      --input "${INPUTS[@]}" \
      --output_file "${BASE_RAW}" \
      --policy_min_q 0.5 \
      --max_value_paths_per_dimension 0 \
      --shuffle_seed 20260420
  else
    echo "[stage:${CURRENT_STAGE}] baseline raw exists: ${BASE_RAW}"
  fi

  if [[ ! -f "${CORR_RAW}" ]]; then
    echo "[stage:${CURRENT_STAGE}] correction weak raw start=$(date -Is)"
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${ROOT}/model_training/src" \
    HF_HUB_OFFLINE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.preprocess_review_mcts_data \
      --input "${INPUTS[@]}" \
      --output_file "${CORR_RAW}" \
      --policy_min_q 0.5 \
      --max_value_paths_per_dimension 0 \
      --emit_verifier_corrections \
      --verifier_correction_repeat 1 \
      --verifier_correction_q 0.65 \
      --verifier_correction_lm_weight 0.15 \
      --verifier_correction_value_weight 0.5 \
      --shuffle_seed 20260420
  else
    echo "[stage:${CURRENT_STAGE}] correction raw exists: ${CORR_RAW}"
  fi

  if [[ ! -f "${BASE_DATA}" ]]; then
    echo "[stage:${CURRENT_STAGE}] baseline filter start=$(date -Is)"
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${ROOT}/model_training/src" \
    TRL_EXPERIMENTAL_SILENCE=1 \
    HF_HUB_OFFLINE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.filter_review_train_data \
      --input_file "${BASE_RAW}" \
      --output_file "${BASE_DATA}" \
      --model_key Qwen/Qwen3.5-9B \
      --model_name_or_path "${MODEL_PATH}" \
      --max_tokens 1536 \
      --min_policy_items 40 \
      --shuffle_seed 20260420
  else
    echo "[stage:${CURRENT_STAGE}] baseline filtered exists: ${BASE_DATA}"
  fi

  if [[ ! -f "${CORR_DATA}" ]]; then
    echo "[stage:${CURRENT_STAGE}] correction weak filter start=$(date -Is)"
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${ROOT}/model_training/src" \
    TRL_EXPERIMENTAL_SILENCE=1 \
    HF_HUB_OFFLINE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.filter_review_train_data \
      --input_file "${CORR_RAW}" \
      --output_file "${CORR_DATA}" \
      --model_key Qwen/Qwen3.5-9B \
      --model_name_or_path "${MODEL_PATH}" \
      --max_tokens 1536 \
      --min_policy_items 70 \
      --shuffle_seed 20260420
  else
    echo "[stage:${CURRENT_STAGE}] correction weak filtered exists: ${CORR_DATA}"
  fi
  wc -l "${BASE_RAW}" "${CORR_RAW}" "${BASE_DATA}" "${CORR_DATA}"
}

train_one() {
  local tag="$1"
  local cuda_device="$2"
  local data_file="$3"
  local out_dir="$4"
  local log_file="${LOG_DIR}/train_${tag}.log"

  if [[ -f "${out_dir}/adapter_model.safetensors" && -f "${out_dir}/value_head.pth" ]]; then
    echo "[stage:train_${tag}] final checkpoint exists, skipping: ${out_dir}"
    return
  fi

  (
    set -euo pipefail
    cd "${ROOT}/model_training/src"
    RESUME_ARGS=()
    LATEST_CHECKPOINT="$(find "${out_dir}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
    if [[ -n "${LATEST_CHECKPOINT}" ]]; then
      RESUME_ARGS=(--resume_from_checkpoint "${LATEST_CHECKPOINT}")
      echo "[stage:train_${tag}] resuming from ${LATEST_CHECKPOINT}"
    fi
    echo "[stage:train_${tag}] start=$(date -Is) cuda=${cuda_device}"
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
      --max_steps 160 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --max_training_seq_length 1536 \
      --bf16 True \
      --logging_steps 10 \
      --save_strategy steps \
      --save_steps 80 \
      --save_total_limit 2 \
      --report_to none \
      --optim adafactor \
      --learning_rate 5e-5 \
      --lr_scheduler_type linear \
      --warmup_steps 16 \
      --peft lora \
      --value_weight 0.025 \
      --num_proc 1 \
      "${RESUME_ARGS[@]}"
    echo "[stage:train_${tag}] done=$(date -Is)"
  ) >"${log_file}" 2>&1 &
}

wait_for_train() {
  CURRENT_STAGE="train_parallel"
  train_one "baseline" 0 "${BASE_DATA}" "${BASE_OUT}"
  pid_base=$!
  train_one "correction_weak" 1 "${CORR_DATA}" "${CORR_OUT}"
  pid_corr=$!
  echo "[stage:${CURRENT_STAGE}] waiting baseline pid=${pid_base}, correction pid=${pid_corr}"
  wait "${pid_base}"
  wait "${pid_corr}"
  echo "[stage:${CURRENT_STAGE}] done=$(date -Is)"
}

eval_one() {
  local tag="$1"
  local cuda_device="$2"
  local out_dir="$3"
  local eval_dir="${EVAL_ROOT}/${tag}"
  local log_file="${LOG_DIR}/eval_${tag}.log"
  mkdir -p "${eval_dir}"

  (
    set -euo pipefail
    cd "${ROOT}/model_training/src"
    echo "[stage:eval_${tag}] start=$(date -Is) cuda=${cuda_device}"
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
    echo "[stage:eval_${tag}] done=$(date -Is)"
  ) >"${log_file}" 2>&1 &
}

wait_for_eval() {
  CURRENT_STAGE="eval_parallel"
  eval_one "baseline" 0 "${BASE_OUT}"
  pid_base=$!
  eval_one "correction_weak" 1 "${CORR_OUT}"
  pid_corr=$!
  echo "[stage:${CURRENT_STAGE}] waiting baseline pid=${pid_base}, correction pid=${pid_corr}"
  wait "${pid_base}"
  wait "${pid_corr}"
  echo "[stage:${CURRENT_STAGE}] done=$(date -Is)"
}

echo "[pipeline] start=$(date -Is)"
echo "[pipeline] log=${LOG_FILE}"
if [[ ! -f "${INDICES_FILE}" ]]; then
  echo "Missing held-out indices file: ${INDICES_FILE}" >&2
  exit 1
fi
convert_and_filter
wait_for_train
wait_for_eval

CURRENT_STAGE="finished"
echo "[pipeline] finished=$(date -Is)"
notify "COMPLETED"
