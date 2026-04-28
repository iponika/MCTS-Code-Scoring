#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data1/xianzhiwei/mcts-code-review}"
RUN_NAME="${RUN_NAME:-bootstrap_comparison_qwen3_4b_20260424}"
CFG="${CFG:-${ROOT}/data_collection/configs/mcts_code_review_qwen3_4b.yaml}"
MODEL_KEY="${MODEL_KEY:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
SEED_PER_GRADE="${SEED_PER_GRADE:-24}"
SEED_MIN_GRADE="${SEED_MIN_GRADE:-1}"
SEED_MAX_GRADE="${SEED_MAX_GRADE:-5}"
SEED_MAX_OBJECTIVE_ASSERTIONS_PER_SPLIT="${SEED_MAX_OBJECTIVE_ASSERTIONS_PER_SPLIT:-8}"
SEED_ASSERTION_TIMEOUT_SECONDS="${SEED_ASSERTION_TIMEOUT_SECONDS:-1.5}"
DIRECT_REPEATS="${DIRECT_REPEATS:-4}"
DIRECT_BOOTSTRAP_RESPONSE_MODE="${DIRECT_BOOTSTRAP_RESPONSE_MODE:-review}"
DIRECT_BOOTSTRAP_REASONING_STEPS="${DIRECT_BOOTSTRAP_REASONING_STEPS:-3}"
MAX_TRAINING_SEQ_LENGTH="${MAX_TRAINING_SEQ_LENGTH:-3072}"
MAX_STEPS="${MAX_STEPS:-240}"
EVAL_PER_GRADE="${EVAL_PER_GRADE:-8}"
POLICY_MIN_Q="${POLICY_MIN_Q:-0.5}"
MAX_VALUE_PATHS_PER_DIMENSION="${MAX_VALUE_PATHS_PER_DIMENSION:-0}"
DIRECT_POLICY_RESPONSE_MODE="${DIRECT_POLICY_RESPONSE_MODE:-path}"
MCTS_POLICY_RESPONSE_MODE="${MCTS_POLICY_RESPONSE_MODE:-path}"
DIRECT_KEEP_ALL_VALUE_PATHS="${DIRECT_KEEP_ALL_VALUE_PATHS:-1}"
MCTS_KEEP_ALL_VALUE_PATHS="${MCTS_KEEP_ALL_VALUE_PATHS:-1}"
BOOTSTRAP_STRATIFY_BY_DATASET="${BOOTSTRAP_STRATIFY_BY_DATASET:-1}"
BOOTSTRAP_STRATIFY_BY_DELTA_BUCKET="${BOOTSTRAP_STRATIFY_BY_DELTA_BUCKET:-1}"
EVAL_FINAL_ONLY_JSON="${EVAL_FINAL_ONLY_JSON:-1}"
STEPWISE_MAX_STEPS="${STEPWISE_MAX_STEPS:-3}"
STEPWISE_NUM_CANDIDATES="${STEPWISE_NUM_CANDIDATES:-3}"
STEPWISE_MAX_RETHINKS="${STEPWISE_MAX_RETHINKS:-1}"
STEPWISE_RETHINK_THRESHOLD="${STEPWISE_RETHINK_THRESHOLD:--0.2}"
STEPWISE_RETHINK_SPREAD_THRESHOLD="${STEPWISE_RETHINK_SPREAD_THRESHOLD:-0.0}"
NTFY_URL="${NTFY_URL:-https://ntfy.sh/iponika_mcts}"

RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
SEED_DATA="${RUN_DIR}/seed_codecritic_axiom.jsonl"
SEED_META="${RUN_DIR}/seed_codecritic_axiom.metadata.json"
DIRECT_RAW="${RUN_DIR}/direct_bootstrap_raw.jsonl"
MCTS_RAW="${RUN_DIR}/mcts_bootstrap_raw.jsonl"
MCTS_SAMPLES_DIR="${RUN_DIR}/mcts_samples"

TRAIN_DIR="${ROOT}/model_training/review_mcts_train_data"
STATIC_TRAIN_RAW="${TRAIN_DIR}/${RUN_NAME}_static_raw.jsonl"
DIRECT_TRAIN_RAW="${TRAIN_DIR}/${RUN_NAME}_direct_bootstrap_raw.jsonl"
MCTS_TRAIN_RAW="${TRAIN_DIR}/${RUN_NAME}_mcts_bootstrap_raw.jsonl"
STATIC_TRAIN="${TRAIN_DIR}/${RUN_NAME}_static.jsonl"
DIRECT_TRAIN="${TRAIN_DIR}/${RUN_NAME}_direct_bootstrap.jsonl"
MCTS_TRAIN="${TRAIN_DIR}/${RUN_NAME}_mcts_bootstrap.jsonl"
BALANCE_META="${RUN_DIR}/train_balance_summary.json"

OUTPUT_ROOT="${ROOT}/model_training/src/output"
STATIC_MODEL="${OUTPUT_ROOT}/review-lora-${RUN_NAME}-static-${MAX_STEPS}step"
DIRECT_MODEL="${OUTPUT_ROOT}/review-lora-${RUN_NAME}-direct-${MAX_STEPS}step"
MCTS_MODEL="${OUTPUT_ROOT}/review-lora-${RUN_NAME}-mcts-${MAX_STEPS}step"

SUMMARY_JSON="${RUN_DIR}/summary.json"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
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

prepare_seed_data() {
  CURRENT_STAGE="prepare_seed_data"
  if [[ ! -f "${SEED_DATA}" ]]; then
    cd "${ROOT}"
    PYTHONPATH="${ROOT}/data_collection" UV_CACHE_DIR=/tmp/uv-cache \
      uv run python data_collection/prepare_codecritic_axiom_seedset.py \
        --output "${SEED_DATA}" \
        --metadata "${SEED_META}" \
        --per_grade "${SEED_PER_GRADE}" \
        --min_grade "${SEED_MIN_GRADE}" \
        --max_grade "${SEED_MAX_GRADE}" \
        --max_objective_assertions_per_split "${SEED_MAX_OBJECTIVE_ASSERTIONS_PER_SPLIT}" \
        --assertion_timeout_seconds "${SEED_ASSERTION_TIMEOUT_SECONDS}" \
        --progress_every 50
  fi
}

generate_direct_bootstrap() {
  CURRENT_STAGE="generate_direct_bootstrap"
  if [[ -f "${DIRECT_RAW}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${DIRECT_RAW}"
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
    --output "${DIRECT_RAW}" \
    --dimension "Correctness Verification" \
    --batch_size 4 \
    --repeats "${DIRECT_REPEATS}" \
    --response_mode "${DIRECT_BOOTSTRAP_RESPONSE_MODE}" \
    --reasoning_steps "${DIRECT_BOOTSTRAP_REASONING_STEPS}"
}

generate_mcts_bootstrap() {
  CURRENT_STAGE="generate_mcts_bootstrap"
  if [[ -f "${MCTS_RAW}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${MCTS_RAW}"
    return
  fi
  cd "${ROOT}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  PYTHONPATH="${ROOT}/data_collection" \
  UV_CACHE_DIR=/tmp/uv-cache \
  HF_HUB_OFFLINE=1 \
  uv run python data_collection/solver_review.py \
    --custom_cfg "${CFG}" \
    --dataset "${SEED_DATA}" \
    --start 0 \
    --limit "$(wc -l < "${SEED_DATA}" | tr -d ' ')" \
    --output "${MCTS_RAW}" \
    --output_dir "${MCTS_SAMPLES_DIR}"
}

prepare_static_train() {
  CURRENT_STAGE="prepare_static_train"
  if [[ -f "${STATIC_TRAIN_RAW}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${STATIC_TRAIN_RAW}"
    return
  fi
  cd "${ROOT}"
  PYTHONPATH="${ROOT}:${ROOT}/model_training/src:${ROOT}/data_collection" UV_CACHE_DIR=/tmp/uv-cache \
    uv run python data_collection/prepare_static_review_train_data.py \
      --input "${SEED_DATA}" \
      --output "${STATIC_TRAIN_RAW}"
}

prepare_bootstrap_train() {
  local mode="$1"
  local raw_file="$2"
  local output_file="$3"
  local policy_response_mode="path"
  if [[ "${mode}" == "direct" ]]; then
    policy_response_mode="${DIRECT_POLICY_RESPONSE_MODE}"
  elif [[ "${mode}" == "mcts" ]]; then
    policy_response_mode="${MCTS_POLICY_RESPONSE_MODE}"
  fi
  CURRENT_STAGE="prepare_${mode}_train"
  if [[ -f "${output_file}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${output_file}"
    return
  fi
  cd "${ROOT}"
  PYTHONPATH="${ROOT}:${ROOT}/model_training/src:${ROOT}/data_collection" UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.preprocess_review_mcts_data \
      --input "${raw_file}" \
      --output_file "${output_file}" \
      --policy_min_q "${POLICY_MIN_Q}" \
      --policy_response_mode "${policy_response_mode}" \
      --max_value_paths_per_dimension "${MAX_VALUE_PATHS_PER_DIMENSION}"
}

balance_train_sets() {
  CURRENT_STAGE="balance_train_sets"
  if [[ -f "${STATIC_TRAIN}" && -f "${DIRECT_TRAIN}" && -f "${MCTS_TRAIN}" && -f "${BALANCE_META}" ]]; then
    echo "[stage:${CURRENT_STAGE}] balanced train files already exist"
    return
  fi
  local counts
  counts="$(
    python - <<PY
import json
from pathlib import Path

def count(path_str: str):
    path = Path(path_str)
    items = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    policy = sum(1 for item in items if item.get('train_lm'))
    value = sum(1 for item in items if not item.get('train_lm'))
    return {'total': len(items), 'policy': policy, 'value': value}

direct = count("${DIRECT_TRAIN_RAW}")
mcts = count("${MCTS_TRAIN_RAW}")
target_policy = min(direct['policy'], mcts['policy'])
direct_target_value = direct['value'] if "${DIRECT_KEEP_ALL_VALUE_PATHS}" == "1" else min(direct['value'], mcts['value'])
mcts_target_value = mcts['value'] if "${MCTS_KEEP_ALL_VALUE_PATHS}" == "1" else min(direct['value'], mcts['value'])
direct_target_total = target_policy + direct_target_value
mcts_target_total = target_policy + mcts_target_value
payload = {
    'direct': direct,
    'mcts': mcts,
    'target_policy': target_policy,
    'direct_target_value': direct_target_value,
    'mcts_target_value': mcts_target_value,
    'direct_target_total': direct_target_total,
    'mcts_target_total': mcts_target_total,
    'direct_keep_all_value_paths': "${DIRECT_KEEP_ALL_VALUE_PATHS}",
    'mcts_keep_all_value_paths': "${MCTS_KEEP_ALL_VALUE_PATHS}",
}
print(json.dumps(payload))
PY
  )"
  local target_policy
  local direct_target_value
  local mcts_target_value
  local direct_target_total
  local mcts_target_total
  target_policy="$(python - <<PY
import json
payload=json.loads('''${counts}''')
print(payload['target_policy'])
PY
)"
  direct_target_value="$(python - <<PY
import json
payload=json.loads('''${counts}''')
print(payload['direct_target_value'])
PY
)"
  mcts_target_value="$(python - <<PY
import json
payload=json.loads('''${counts}''')
print(payload['mcts_target_value'])
PY
)"
  direct_target_total="$(python - <<PY
import json
payload=json.loads('''${counts}''')
print(payload['direct_target_total'])
PY
)"
  mcts_target_total="$(python - <<PY
import json
payload=json.loads('''${counts}''')
print(payload['mcts_target_total'])
PY
)"
  echo "${counts}" | python -m json.tool > "${BALANCE_META}"

  cd "${ROOT}"
  direct_balance_args=()
  mcts_balance_args=()
  static_balance_args=()
  if [[ "${BOOTSTRAP_STRATIFY_BY_DATASET}" == "1" ]]; then
    direct_balance_args+=(--stratify_by_dataset)
    mcts_balance_args+=(--stratify_by_dataset)
    static_balance_args+=(--stratify_by_dataset)
  fi
  if [[ "${BOOTSTRAP_STRATIFY_BY_DELTA_BUCKET}" == "1" ]]; then
    direct_balance_args+=(--stratify_by_delta_bucket)
    mcts_balance_args+=(--stratify_by_delta_bucket)
  fi
  PYTHONPATH="${ROOT}" UV_CACHE_DIR=/tmp/uv-cache \
    uv run python data_collection/rebalance_review_train_data.py \
      --input "${DIRECT_TRAIN_RAW}" \
      --output "${DIRECT_TRAIN}" \
      --target_policy_count "${target_policy}" \
      --target_value_count "${direct_target_value}" \
      --target_total_count "${direct_target_total}" \
      "${direct_balance_args[@]}"
  PYTHONPATH="${ROOT}" UV_CACHE_DIR=/tmp/uv-cache \
    uv run python data_collection/rebalance_review_train_data.py \
      --input "${MCTS_TRAIN_RAW}" \
      --output "${MCTS_TRAIN}" \
      --target_policy_count "${target_policy}" \
      --target_value_count "${mcts_target_value}" \
      --target_total_count "${mcts_target_total}" \
      "${mcts_balance_args[@]}"
  PYTHONPATH="${ROOT}" UV_CACHE_DIR=/tmp/uv-cache \
    uv run python data_collection/rebalance_review_train_data.py \
      --input "${STATIC_TRAIN_RAW}" \
      --output "${STATIC_TRAIN}" \
      --target_total_count "${direct_target_total}" \
      "${static_balance_args[@]}"
}

train_one() {
  local tag="$1"
  local data_file="$2"
  local output_dir="$3"
  local log_file="${LOG_DIR}/train_${tag}.log"
  CURRENT_STAGE="train_${tag}"
  if [[ -f "${output_dir}/adapter_model.safetensors" && -f "${output_dir}/value_head.pth" ]]; then
    echo "[stage:${CURRENT_STAGE}] final checkpoint exists: ${output_dir}"
    return
  fi
  cd "${ROOT}/model_training/src"
  resume_args=()
  latest_checkpoint="$(find "${output_dir}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
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
    --datafile_paths "../review_mcts_train_data/$(basename "${data_file}")" \
    --output_dir "${output_dir}" \
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

run_eval_for_model() {
  local tag="$1"
  local trained_model="$2"
  CURRENT_STAGE="eval_${tag}"
  RUN_NAME="${RUN_NAME}_${tag}_axiom_eval" \
  BASE_MODEL_PATH="${MODEL_PATH}" \
  TRAINED_MODEL_PATH="${trained_model}" \
  TRAIN_DATA="${SEED_DATA}" \
  PER_GRADE="${EVAL_PER_GRADE}" \
  DROP_AXIOM_GRADE_ZERO=1 \
  FINAL_ONLY_JSON="${EVAL_FINAL_ONLY_JSON}" \
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
run_name = "${RUN_NAME}"
summary_path = Path("${SUMMARY_JSON}")

def load_eval(tag: str):
    path = root / "model_training" / "src" / "output" / f"review-eval-{run_name}_{tag}_axiom_eval" / "comparison.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "base_direct_clean": payload.get("base_direct_clean"),
        "trained_direct_clean": payload.get("trained_direct_clean"),
        "trained_value_rerank_clean": payload.get("trained_value_rerank_clean"),
    }

summary = {
    "run_name": run_name,
    "seed_data": "${SEED_DATA}",
    "static_train": "${STATIC_TRAIN}",
    "direct_train": "${DIRECT_TRAIN}",
    "mcts_train": "${MCTS_TRAIN}",
    "max_training_seq_length": int("${MAX_TRAINING_SEQ_LENGTH}"),
    "eval_final_only_json": "${EVAL_FINAL_ONLY_JSON}",
    "direct_bootstrap_response_mode": "${DIRECT_BOOTSTRAP_RESPONSE_MODE}",
    "direct_bootstrap_reasoning_steps": int("${DIRECT_BOOTSTRAP_REASONING_STEPS}"),
    "direct_policy_response_mode": "${DIRECT_POLICY_RESPONSE_MODE}",
    "mcts_policy_response_mode": "${MCTS_POLICY_RESPONSE_MODE}",
    "direct_keep_all_value_paths": "${DIRECT_KEEP_ALL_VALUE_PATHS}",
    "mcts_keep_all_value_paths": "${MCTS_KEEP_ALL_VALUE_PATHS}",
    "evaluations": {
        "static": load_eval("static"),
        "direct": load_eval("direct"),
        "mcts": load_eval("mcts"),
    },
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

main() {
  echo "[start] run=${RUN_NAME} time=$(date -Is) log=${LOG_FILE}"
  prepare_seed_data
  generate_direct_bootstrap
  generate_mcts_bootstrap
  prepare_static_train
  prepare_bootstrap_train "direct" "${DIRECT_RAW}" "${DIRECT_TRAIN_RAW}"
  prepare_bootstrap_train "mcts" "${MCTS_RAW}" "${MCTS_TRAIN_RAW}"
  balance_train_sets

  if [[ "${STOP_AFTER_DATA:-0}" == "1" ]]; then
    CURRENT_STAGE="finished"
    echo "[finished] STOP_AFTER_DATA=1 run_dir=${RUN_DIR}"
    exit 0
  fi

  train_one "static" "${STATIC_TRAIN}" "${STATIC_MODEL}"
  train_one "direct" "${DIRECT_TRAIN}" "${DIRECT_MODEL}"
  train_one "mcts" "${MCTS_TRAIN}" "${MCTS_MODEL}"

  if [[ "${STOP_AFTER_TRAIN:-0}" == "1" ]]; then
    CURRENT_STAGE="finished"
    echo "[finished] STOP_AFTER_TRAIN=1"
    exit 0
  fi

  run_eval_for_model "static" "${STATIC_MODEL}"
  run_eval_for_model "direct" "${DIRECT_MODEL}"
  run_eval_for_model "mcts" "${MCTS_MODEL}"
  write_summary
  CURRENT_STAGE="finished"
  notify "FINISHED"
  echo "[finished] summary=${SUMMARY_JSON}"
}

main "$@"
