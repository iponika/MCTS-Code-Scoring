#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/xianzhiwei/mcts-code-review"
RUN_NAME="verifier_correction_ablation_20260419"
MODEL_PATH="/data1/xianzhiwei/model/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"

RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
EVAL_ROOT="${ROOT}/model_training/src/output/review-eval-${RUN_NAME}"
BASE_DATA="${ROOT}/model_training/review_mcts_train_data/verifier_ablation_baseline_20260419.jsonl"
CORR_DATA="${ROOT}/model_training/review_mcts_train_data/verifier_ablation_correction_repeat8_20260419.jsonl"
BASE_OUT="${ROOT}/model_training/src/output/review-lora-verifier-ablation-baseline-80step"
CORR_OUT="${ROOT}/model_training/src/output/review-lora-verifier-ablation-correction-repeat8-80step"

INPUTS=(
  "${ROOT}/data_collection/review_mcts_runs/evidence_style_smoke4_v3_20260419/samples"
  "${ROOT}/data_collection/review_mcts_runs/supervised_medium_20260418/samples"
)
EVAL_INDICES=(53 54 55 58 59 60)

mkdir -p "${LOG_DIR}" "${EVAL_ROOT}"
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

convert_data() {
  CURRENT_STAGE="convert"
  cd "${ROOT}"
  if [[ ! -f "${BASE_DATA}" ]]; then
    echo "[stage:convert] baseline start=$(date -Is)"
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${ROOT}/model_training/src" \
    HF_HUB_OFFLINE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.preprocess_review_mcts_data \
      --input "${INPUTS[@]}" \
      --output_file "${BASE_DATA}" \
      --policy_min_q 0.5 \
      --max_value_paths_per_dimension 0 \
      --shuffle_seed 20260419
  else
    echo "[stage:convert] baseline exists: ${BASE_DATA}"
  fi

  if [[ ! -f "${CORR_DATA}" ]]; then
    echo "[stage:convert] correction start=$(date -Is)"
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${ROOT}/model_training/src" \
    HF_HUB_OFFLINE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.preprocess_review_mcts_data \
      --input "${INPUTS[@]}" \
      --output_file "${CORR_DATA}" \
      --policy_min_q 0.5 \
      --max_value_paths_per_dimension 0 \
      --emit_verifier_corrections \
      --verifier_correction_repeat 8 \
      --verifier_correction_q 0.8 \
      --verifier_correction_lm_weight 0.5 \
      --verifier_correction_value_weight 0.5 \
      --shuffle_seed 20260419
  else
    echo "[stage:convert] correction exists: ${CORR_DATA}"
  fi
  wc -l "${BASE_DATA}" "${CORR_DATA}"
}

train_one() {
  local tag="$1"
  local data_file="$2"
  local out_dir="$3"

  CURRENT_STAGE="train_${tag}"
  cd "${ROOT}/model_training/src"
  if [[ -f "${out_dir}/adapter_model.safetensors" && -f "${out_dir}/value_head.pth" ]]; then
    echo "[stage:${CURRENT_STAGE}] final checkpoint exists, skipping: ${out_dir}"
    return
  fi

  RESUME_ARGS=()
  LATEST_CHECKPOINT="$(find "${out_dir}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "${LATEST_CHECKPOINT}" ]]; then
    RESUME_ARGS=(--resume_from_checkpoint "${LATEST_CHECKPOINT}")
    echo "[stage:${CURRENT_STAGE}] resuming from ${LATEST_CHECKPOINT}"
  fi

  echo "[stage:${CURRENT_STAGE}] start=$(date -Is)"
  CUDA_VISIBLE_DEVICES=0 \
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
    --max_steps 80 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_training_seq_length 1024 \
    --bf16 True \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 40 \
    --save_total_limit 2 \
    --report_to none \
    --optim adafactor \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --warmup_steps 8 \
    --peft lora \
    --value_weight 0.025 \
    --num_proc 1 \
    "${RESUME_ARGS[@]}"
  echo "[stage:${CURRENT_STAGE}] done=$(date -Is)"
}

eval_one() {
  local tag="$1"
  local out_dir="$2"
  local eval_dir="${EVAL_ROOT}/${tag}"
  mkdir -p "${eval_dir}"

  CURRENT_STAGE="eval_${tag}"
  cd "${ROOT}/model_training/src"
  echo "[stage:${CURRENT_STAGE}] start=$(date -Is)"
  for record_index in "${EVAL_INDICES[@]}"; do
    local output_file="${eval_dir}/raw_${record_index}_correctness.json"
    if [[ -f "${output_file}" ]]; then
      echo "[stage:${CURRENT_STAGE}] exists, skip ${output_file}"
      continue
    fi
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_DATASETS_CACHE=/tmp/hf-datasets-cache \
    HF_HUB_OFFLINE=1 \
    TRL_EXPERIMENTAL_SILENCE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.review_evaluator \
      --policy_model_path "${out_dir}" \
      --value_model_path "${out_dir}" \
      --share_policy_value_model \
      --input_record ../../datasets/CodeCriticBench/data/CodeCriticBench.jsonl \
      --record_index "${record_index}" \
      --output_file "${output_file}" \
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
  done

  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${ROOT}/model_training/src:${ROOT}/data_collection" \
  HF_HUB_OFFLINE=1 \
  UV_CACHE_DIR=/tmp/uv-cache \
  uv run python "${ROOT}/data_collection/scripts/summarize_review_eval_outputs.py" \
    --eval_dir "${eval_dir}" \
    --output "${eval_dir}/summary.json"
  echo "[stage:${CURRENT_STAGE}] done=$(date -Is)"
}

echo "[pipeline] start=$(date -Is)"
echo "[pipeline] log=${LOG_FILE}"
convert_data
train_one "baseline" "${BASE_DATA}" "${BASE_OUT}"
train_one "correction" "${CORR_DATA}" "${CORR_OUT}"
eval_one "baseline" "${BASE_OUT}"
eval_one "correction" "${CORR_OUT}"

CURRENT_STAGE="finished"
echo "[pipeline] finished=$(date -Is)"
notify "COMPLETED"
