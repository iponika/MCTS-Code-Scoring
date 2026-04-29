#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BASE_RUN_NAME="${BASE_RUN_NAME:-bootstrap_cmp_4b_stepwise_large_20260428}"
RUN_NAME="${RUN_NAME:-direct_stepwise_vs_review_20260428}"
CFG="${CFG:-${ROOT}/data_collection/configs/mcts_code_review_qwen3_4b.yaml}"
MODEL_KEY="${MODEL_KEY:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
DIRECT_REPEATS="${DIRECT_REPEATS:-4}"
DIRECT_BOOTSTRAP_REASONING_STEPS="${DIRECT_BOOTSTRAP_REASONING_STEPS:-3}"
MAX_STEPS="${MAX_STEPS:-320}"
MAX_TRAINING_SEQ_LENGTH="${MAX_TRAINING_SEQ_LENGTH:-3072}"
POLICY_MIN_Q="${POLICY_MIN_Q:-0.8}"
EVAL_PER_GRADE="${EVAL_PER_GRADE:-16}"
STEPWISE_MAX_STEPS="${STEPWISE_MAX_STEPS:-3}"
STEPWISE_NUM_CANDIDATES="${STEPWISE_NUM_CANDIDATES:-3}"
STEPWISE_MAX_RETHINKS="${STEPWISE_MAX_RETHINKS:-1}"
STEPWISE_RETHINK_THRESHOLD="${STEPWISE_RETHINK_THRESHOLD:--0.2}"
STEPWISE_RETHINK_SPREAD_THRESHOLD="${STEPWISE_RETHINK_SPREAD_THRESHOLD:-0.0}"
NTFY_URL="${NTFY_URL:-https://ntfy.sh/iponika_mcts}"

BASE_RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${BASE_RUN_NAME}"
RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
SEED_DATA="${SEED_DATA:-${BASE_RUN_DIR}/seed_codecritic_axiom.jsonl}"
BASE_REVIEW_TRAIN="${BASE_REVIEW_TRAIN:-${ROOT}/model_training/review_mcts_train_data/${BASE_RUN_NAME}_direct_bootstrap.jsonl}"
BASE_REVIEW_EVAL="${BASE_REVIEW_EVAL:-${ROOT}/model_training/src/output/review-eval-${BASE_RUN_NAME}_direct_axiom_eval/comparison.json}"

TRAIN_DIR="${ROOT}/model_training/review_mcts_train_data"
STEPWISE_RAW="${RUN_DIR}/direct_stepwise_bootstrap_raw.jsonl"
STEPWISE_TRAIN_RAW="${TRAIN_DIR}/${RUN_NAME}_direct_stepwise_raw.jsonl"
STEPWISE_TRAIN="${TRAIN_DIR}/${RUN_NAME}_direct_stepwise.jsonl"
BALANCE_META="${RUN_DIR}/train_balance_summary.json"
SUMMARY_JSON="${RUN_DIR}/summary.json"

OUTPUT_ROOT="${ROOT}/model_training/src/output"
STEPWISE_MODEL="${OUTPUT_ROOT}/review-lora-${RUN_NAME}-direct-stepwise-${MAX_STEPS}step"

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${TRAIN_DIR}"
LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE="init"

notify() {
  local status="$1"
  local message="${RUN_NAME} ${status}: stage=${CURRENT_STAGE}, host=$(hostname), time=$(date -Is), log=${LOG_FILE}, run_dir=${RUN_DIR}"
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    curl --noproxy '*' --connect-timeout 5 --max-time 20 -fsS -d "${message}" "${NTFY_URL}" >/dev/null 2>&1 || true
}

on_exit() {
  local code=$?
  if [[ "${CURRENT_STAGE}" != "finished" ]]; then
    notify "FAILED(code=${code})"
  fi
  exit "${code}"
}
trap on_exit EXIT

count_train() {
  local path="$1"
  python - <<PY
import json
from pathlib import Path
path = Path("${path}")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
print(json.dumps({
    "total": len(rows),
    "policy": sum(1 for row in rows if row.get("train_lm")),
    "value": sum(1 for row in rows if not row.get("train_lm")),
}))
PY
}

generate_stepwise_direct() {
  CURRENT_STAGE="generate_stepwise_direct"
  if [[ -f "${STEPWISE_RAW}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${STEPWISE_RAW}"
    return
  fi
  cd "${ROOT}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  PYTHONPATH="${ROOT}/data_collection" \
  UV_CACHE_DIR=/tmp/uv-cache \
  HF_HUB_OFFLINE=1 \
  uv run python data_collection/direct_bootstrap_review.py \
    --custom_cfg "${CFG}" \
    --dataset "${SEED_DATA}" \
    --start 0 \
    --limit "$(wc -l < "${SEED_DATA}" | tr -d ' ')" \
    --output "${STEPWISE_RAW}" \
    --dimension "Correctness Verification" \
    --batch_size 4 \
    --repeats "${DIRECT_REPEATS}" \
    --response_mode stepwise \
    --reasoning_steps "${DIRECT_BOOTSTRAP_REASONING_STEPS}"
}

prepare_stepwise_train() {
  CURRENT_STAGE="prepare_stepwise_train"
  if [[ -f "${STEPWISE_TRAIN_RAW}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${STEPWISE_TRAIN_RAW}"
    return
  fi
  cd "${ROOT}"
  PYTHONPATH="${ROOT}:${ROOT}/model_training/src:${ROOT}/data_collection" UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.preprocess_review_mcts_data \
      --input "${STEPWISE_RAW}" \
      --output_file "${STEPWISE_TRAIN_RAW}" \
      --policy_min_q "${POLICY_MIN_Q}" \
      --policy_response_mode path \
      --max_value_paths_per_dimension 0
}

balance_against_review_only() {
  CURRENT_STAGE="balance_against_review_only"
  if [[ -f "${STEPWISE_TRAIN}" && -f "${BALANCE_META}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${STEPWISE_TRAIN}"
    return
  fi
  local base_counts
  local step_counts
  base_counts="$(count_train "${BASE_REVIEW_TRAIN}")"
  step_counts="$(count_train "${STEPWISE_TRAIN_RAW}")"
  local target_policy
  local target_value
  local target_total
  target_policy="$(python - <<PY
import json
print(json.loads('''${base_counts}''')["policy"])
PY
)"
  target_value="$(python - <<PY
import json
print(json.loads('''${base_counts}''')["value"])
PY
)"
  target_total="$(python - <<PY
import json
print(json.loads('''${base_counts}''')["total"])
PY
)"
  python - <<PY > "${BALANCE_META}"
import json
print(json.dumps({
    "base_review_only": json.loads('''${base_counts}'''),
    "direct_stepwise_raw": json.loads('''${step_counts}'''),
    "target_policy": int("${target_policy}"),
    "target_value": int("${target_value}"),
    "target_total": int("${target_total}"),
}, indent=2))
PY
  cd "${ROOT}"
  PYTHONPATH="${ROOT}" UV_CACHE_DIR=/tmp/uv-cache \
    uv run python data_collection/rebalance_review_train_data.py \
      --input "${STEPWISE_TRAIN_RAW}" \
      --output "${STEPWISE_TRAIN}" \
      --target_policy_count "${target_policy}" \
      --target_value_count "${target_value}" \
      --target_total_count "${target_total}" \
      --stratify_by_dataset \
      --stratify_by_delta_bucket
}

train_stepwise() {
  CURRENT_STAGE="train_stepwise"
  local log_file="${LOG_DIR}/train_stepwise.log"
  if [[ -f "${STEPWISE_MODEL}/adapter_model.safetensors" && -f "${STEPWISE_MODEL}/value_head.pth" ]]; then
    echo "[stage:${CURRENT_STAGE}] final checkpoint exists: ${STEPWISE_MODEL}"
    return
  fi
  cd "${ROOT}/model_training/src"
  resume_args=()
  latest_checkpoint="$(find "${STEPWISE_MODEL}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "${latest_checkpoint}" ]]; then
    resume_args=(--resume_from_checkpoint "${latest_checkpoint}")
  fi
  CUDA_VISIBLE_DEVICES=0 \
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
    --datafile_paths "../review_mcts_train_data/$(basename "${STEPWISE_TRAIN}")" \
    --output_dir "${STEPWISE_MODEL}" \
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
    --seed 20260424 \
    "${resume_args[@]}" >"${log_file}" 2>&1
}

eval_stepwise() {
  CURRENT_STAGE="eval_stepwise"
  RUN_NAME="${RUN_NAME}_direct_stepwise_axiom_eval" \
  BASE_MODEL_PATH="${MODEL_PATH}" \
  TRAINED_MODEL_PATH="${STEPWISE_MODEL}" \
  TRAIN_DATA="${SEED_DATA}" \
  PER_GRADE="${EVAL_PER_GRADE}" \
  DROP_AXIOM_GRADE_ZERO=1 \
  FINAL_ONLY_JSON=0 \
  EVAL_BASE_DIRECT=0 \
  TRAINED_VALUE_MAX_STEPS="${STEPWISE_MAX_STEPS}" \
  TRAINED_VALUE_NUM_CANDIDATES="${STEPWISE_NUM_CANDIDATES}" \
  MAX_RETHINKS="${STEPWISE_MAX_RETHINKS}" \
  RETHINK_THRESHOLD="${STEPWISE_RETHINK_THRESHOLD}" \
  RETHINK_SPREAD_THRESHOLD="${STEPWISE_RETHINK_SPREAD_THRESHOLD}" \
  NTFY_URL="${NTFY_URL}" \
  "${ROOT}/data_collection/scripts/run_axiom_clean_eval.sh"
}

write_summary() {
  CURRENT_STAGE="write_summary"
  python - <<PY
import json
from pathlib import Path
root = Path("${ROOT}")
summary_path = Path("${SUMMARY_JSON}")
new_eval = root / "model_training" / "src" / "output" / "review-eval-${RUN_NAME}_direct_stepwise_axiom_eval" / "comparison.json"
baseline_eval = Path("${BASE_REVIEW_EVAL}")
payload = {
    "run_name": "${RUN_NAME}",
    "base_run_name": "${BASE_RUN_NAME}",
    "seed_data": "${SEED_DATA}",
    "base_review_train": "${BASE_REVIEW_TRAIN}",
    "stepwise_raw": "${STEPWISE_RAW}",
    "stepwise_train": "${STEPWISE_TRAIN}",
    "stepwise_model": "${STEPWISE_MODEL}",
    "max_training_seq_length": int("${MAX_TRAINING_SEQ_LENGTH}"),
    "max_steps": int("${MAX_STEPS}"),
    "direct_repeats": int("${DIRECT_REPEATS}"),
    "reasoning_steps": int("${DIRECT_BOOTSTRAP_REASONING_STEPS}"),
    "balance": json.loads(Path("${BALANCE_META}").read_text(encoding="utf-8")) if Path("${BALANCE_META}").exists() else None,
    "baseline_review_only_eval": json.loads(baseline_eval.read_text(encoding="utf-8")) if baseline_eval.exists() else None,
    "direct_stepwise_eval": json.loads(new_eval.read_text(encoding="utf-8")) if new_eval.exists() else None,
}
summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}

main() {
  echo "[start] run=${RUN_NAME} time=$(date -Is) log=${LOG_FILE}"
  if [[ ! -f "${SEED_DATA}" ]]; then
    echo "Missing seed data: ${SEED_DATA}" >&2
    exit 2
  fi
  if [[ ! -f "${BASE_REVIEW_TRAIN}" ]]; then
    echo "Missing baseline review-only train data: ${BASE_REVIEW_TRAIN}" >&2
    exit 2
  fi
  generate_stepwise_direct
  prepare_stepwise_train
  balance_against_review_only
  if [[ "${STOP_AFTER_DATA:-0}" == "1" ]]; then
    CURRENT_STAGE="finished"
    echo "[finished] STOP_AFTER_DATA=1 run_dir=${RUN_DIR}"
    exit 0
  fi
  train_stepwise
  if [[ "${STOP_AFTER_TRAIN:-0}" == "1" ]]; then
    CURRENT_STAGE="finished"
    echo "[finished] STOP_AFTER_TRAIN=1 model=${STEPWISE_MODEL}"
    exit 0
  fi
  eval_stepwise
  write_summary
  CURRENT_STAGE="finished"
  notify "FINISHED"
  echo "[finished] summary=${SUMMARY_JSON}"
}

main "$@"
