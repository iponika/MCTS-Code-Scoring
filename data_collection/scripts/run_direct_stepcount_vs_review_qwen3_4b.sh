#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUN_NAME="${RUN_NAME:-direct_stepcount_vs_review_qwen3_4b_20260430}"
CFG="${CFG:-${ROOT}/data_collection/configs/mcts_code_review_qwen3_4b_thinking.yaml}"
MODEL_KEY="${MODEL_KEY:-Qwen/Qwen3-4B}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
SEED_PER_GRADE="${SEED_PER_GRADE:-20}"
SEED_MIN_GRADE="${SEED_MIN_GRADE:-1}"
SEED_MAX_GRADE="${SEED_MAX_GRADE:-5}"
SEED_MAX_OBJECTIVE_ASSERTIONS_PER_SPLIT="${SEED_MAX_OBJECTIVE_ASSERTIONS_PER_SPLIT:-8}"
SEED_ASSERTION_TIMEOUT_SECONDS="${SEED_ASSERTION_TIMEOUT_SECONDS:-1.5}"
DIRECT_REPEATS="${DIRECT_REPEATS:-3}"
STEP_COUNTS="${STEP_COUNTS:-1,2}"
MAX_TRAINING_SEQ_LENGTH="${MAX_TRAINING_SEQ_LENGTH:-3072}"
MAX_STEPS="${MAX_STEPS:-160}"
EVAL_PER_GRADE="${EVAL_PER_GRADE:-6}"
POLICY_MIN_Q="${POLICY_MIN_Q:-0.8}"
NTFY_URL="${NTFY_URL:-https://ntfy.sh/iponika_mcts}"

RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
SEED_DATA="${RUN_DIR}/seed_codecritic_axiom.jsonl"
SEED_META="${RUN_DIR}/seed_codecritic_axiom.metadata.json"
DIRECT_REVIEW_RAW="${RUN_DIR}/direct_review_bootstrap_raw.jsonl"

TRAIN_DIR="${ROOT}/model_training/review_mcts_train_data"
STATIC_TRAIN_RAW="${TRAIN_DIR}/${RUN_NAME}_static_raw.jsonl"
STATIC_TRAIN="${TRAIN_DIR}/${RUN_NAME}_static.jsonl"
DIRECT_REVIEW_TRAIN_RAW="${TRAIN_DIR}/${RUN_NAME}_direct_review_raw.jsonl"
DIRECT_REVIEW_TRAIN="${TRAIN_DIR}/${RUN_NAME}_direct_review.jsonl"
ALIGN_META="${RUN_DIR}/direct_review_alignment_summary.json"
SUMMARY_JSON="${RUN_DIR}/summary.json"

OUTPUT_ROOT="${ROOT}/model_training/src/output"
STATIC_MODEL="${OUTPUT_ROOT}/review-lora-${RUN_NAME}-static-${MAX_STEPS}step"
DIRECT_REVIEW_MODEL="${OUTPUT_ROOT}/review-lora-${RUN_NAME}-direct-review-${MAX_STEPS}step"

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

step_count_list() {
  PYTHONPATH="${ROOT}" python - <<PY
from data_collection.review_experiment_utils import normalize_step_counts
raw = "${STEP_COUNTS}".replace(",", " ").split()
counts = normalize_step_counts(int(item) for item in raw)
print(" ".join(str(item) for item in counts))
PY
}

step_tag() {
  local step_count="$1"
  echo "direct_stepwise_${step_count}step"
}

step_raw_path() {
  local step_count="$1"
  echo "${RUN_DIR}/$(step_tag "${step_count}")_bootstrap_raw.jsonl"
}

step_train_raw_path() {
  local step_count="$1"
  echo "${TRAIN_DIR}/${RUN_NAME}_$(step_tag "${step_count}")_raw.jsonl"
}

step_train_path() {
  local step_count="$1"
  echo "${TRAIN_DIR}/${RUN_NAME}_$(step_tag "${step_count}").jsonl"
}

step_model_path() {
  local step_count="$1"
  echo "${OUTPUT_ROOT}/review-lora-${RUN_NAME}-$(step_tag "${step_count}")-${MAX_STEPS}step"
}

prepare_seed_data() {
  CURRENT_STAGE="prepare_seed_data"
  if [[ -f "${SEED_DATA}" && -f "${SEED_META}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${SEED_DATA}"
    return
  fi
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
}

generate_direct_review() {
  CURRENT_STAGE="generate_direct_review"
  if [[ -f "${DIRECT_REVIEW_RAW}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${DIRECT_REVIEW_RAW}"
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
    --output "${DIRECT_REVIEW_RAW}" \
    --dimension "Correctness Verification" \
    --batch_size 4 \
    --repeats "${DIRECT_REPEATS}" \
    --response_mode review \
    --reasoning_steps 0
}

generate_stepwise_variant() {
  local step_count="$1"
  local output_path
  output_path="$(step_raw_path "${step_count}")"
  CURRENT_STAGE="generate_$(step_tag "${step_count}")"
  if [[ -f "${output_path}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${output_path}"
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
    --output "${output_path}" \
    --dimension "Correctness Verification" \
    --batch_size 4 \
    --repeats "${DIRECT_REPEATS}" \
    --response_mode stepwise \
    --reasoning_steps "${step_count}"
}

prepare_static_train() {
  CURRENT_STAGE="prepare_static_train"
  if [[ -f "${STATIC_TRAIN_RAW}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${STATIC_TRAIN_RAW}"
  else
    cd "${ROOT}"
    PYTHONPATH="${ROOT}:${ROOT}/model_training/src:${ROOT}/data_collection" UV_CACHE_DIR=/tmp/uv-cache \
      uv run python data_collection/prepare_static_review_train_data.py \
        --input "${SEED_DATA}" \
        --output "${STATIC_TRAIN_RAW}"
  fi
  cp "${STATIC_TRAIN_RAW}" "${STATIC_TRAIN}"
}

prepare_bootstrap_train() {
  local input_path="$1"
  local output_path="$2"
  local policy_mode="$3"
  CURRENT_STAGE="prepare_$(basename "${output_path}" .jsonl)"
  if [[ -f "${output_path}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${output_path}"
    return
  fi
  cd "${ROOT}"
  PYTHONPATH="${ROOT}:${ROOT}/model_training/src:${ROOT}/data_collection" UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.preprocess_review_mcts_data \
      --input "${input_path}" \
      --output_file "${output_path}" \
      --policy_min_q "${POLICY_MIN_Q}" \
      --policy_response_mode "${policy_mode}" \
      --max_value_paths_per_dimension 0
}

align_direct_review_to_static() {
  CURRENT_STAGE="align_direct_review_to_static"
  if [[ -f "${DIRECT_REVIEW_TRAIN}" && -f "${ALIGN_META}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${DIRECT_REVIEW_TRAIN}"
    return
  fi
  local static_counts
  local direct_counts
  local targets
  static_counts="$(count_train "${STATIC_TRAIN}")"
  direct_counts="$(count_train "${DIRECT_REVIEW_TRAIN_RAW}")"
  targets="$(
    PYTHONPATH="${ROOT}" python - <<PY
import json
from data_collection.review_experiment_utils import direct_review_alignment_targets
static_counts = json.loads('''${static_counts}''')
direct_counts = json.loads('''${direct_counts}''')
print(json.dumps(direct_review_alignment_targets(static_counts, direct_counts)))
PY
  )"
  echo "${targets}" | python -m json.tool > "${ALIGN_META}"
  cd "${ROOT}"
  PYTHONPATH="${ROOT}" UV_CACHE_DIR=/tmp/uv-cache \
    uv run python data_collection/rebalance_review_train_data.py \
      --input "${DIRECT_REVIEW_TRAIN_RAW}" \
      --output "${DIRECT_REVIEW_TRAIN}" \
      --target_policy_count "$(python - <<PY
import json
print(json.loads('''${targets}''')["target_policy_count"])
PY
)" \
      --target_value_count -1 \
      --target_total_count -1 \
      --stratify_by_dataset \
      --stratify_by_delta_bucket
}

train_one() {
  local tag="$1"
  local data_file="$2"
  local output_dir="$3"
  local gpu_id="$4"
  local log_file="${LOG_DIR}/train_${tag}.log"
  if [[ -f "${output_dir}/adapter_model.safetensors" && -f "${output_dir}/value_head.pth" ]]; then
    echo "[stage:train_${tag}] final checkpoint exists: ${output_dir}"
    return
  fi
  cd "${ROOT}/model_training/src"
  resume_args=()
  latest_checkpoint="$(find "${output_dir}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "${latest_checkpoint}" ]]; then
    resume_args=(--resume_from_checkpoint "${latest_checkpoint}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
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

run_parallel_training_round() {
  local left_tag="$1"
  local left_data="$2"
  local left_model="$3"
  local right_tag="$4"
  local right_data="$5"
  local right_model="$6"
  CURRENT_STAGE="train_${left_tag}_and_${right_tag}"
  (
    train_one "${left_tag}" "${left_data}" "${left_model}" 0
  ) &
  left_pid=$!
  (
    train_one "${right_tag}" "${right_data}" "${right_model}" 1
  ) &
  right_pid=$!
  wait "${left_pid}"
  wait "${right_pid}"
}

eval_final_only_model() {
  local tag="$1"
  local trained_model="$2"
  CURRENT_STAGE="eval_${tag}"
  RUN_NAME="${RUN_NAME}_${tag}_axiom_eval" \
  BASE_MODEL_PATH="${MODEL_PATH}" \
  TRAINED_MODEL_PATH="${trained_model}" \
  TRAIN_DATA="${SEED_DATA}" \
  PER_GRADE="${EVAL_PER_GRADE}" \
  DROP_AXIOM_GRADE_ZERO=1 \
  FINAL_ONLY_JSON=1 \
  EVAL_BASE_DIRECT=0 \
  EVAL_TRAINED_DIRECT=1 \
  EVAL_TRAINED_VALUE=0 \
  TRAINED_DIRECT_MAX_STEPS=1 \
  TRAINED_DIRECT_NUM_CANDIDATES=1 \
  USE_CHAT_TEMPLATE=1 \
  NTFY_URL="${NTFY_URL}" \
  "${ROOT}/data_collection/scripts/run_axiom_clean_eval.sh"
}

eval_stepwise_model() {
  local step_count="$1"
  local trained_model="$2"
  local tag
  tag="$(step_tag "${step_count}")"
  CURRENT_STAGE="eval_${tag}"
  RUN_NAME="${RUN_NAME}_${tag}_axiom_eval" \
  BASE_MODEL_PATH="${MODEL_PATH}" \
  TRAINED_MODEL_PATH="${trained_model}" \
  TRAIN_DATA="${SEED_DATA}" \
  PER_GRADE="${EVAL_PER_GRADE}" \
  DROP_AXIOM_GRADE_ZERO=1 \
  FINAL_ONLY_JSON=0 \
  EVAL_BASE_DIRECT=0 \
  EVAL_TRAINED_DIRECT=1 \
  EVAL_TRAINED_VALUE=0 \
  TRAINED_DIRECT_MAX_STEPS="${step_count}" \
  REASONING_STEPS="${step_count}" \
  TRAINED_DIRECT_NUM_CANDIDATES=1 \
  MAX_RETHINKS=0 \
  USE_CHAT_TEMPLATE=1 \
  NTFY_URL="${NTFY_URL}" \
  "${ROOT}/data_collection/scripts/run_axiom_clean_eval.sh"
}

write_summary() {
  CURRENT_STAGE="write_summary"
  local counts
  counts="$(step_count_list)"
  python - <<PY
import json
from pathlib import Path
from data_collection.review_experiment_utils import stepwise_variants

root = Path("${ROOT}")
run_name = "${RUN_NAME}"
summary_path = Path("${SUMMARY_JSON}")
step_counts = [int(item) for item in "${counts}".split() if item.strip()]

def load_eval(tag: str):
    path = root / "model_training" / "src" / "output" / f"review-eval-{run_name}_{tag}_axiom_eval" / "comparison.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("trained_direct_clean")

stepwise = {}
for variant in stepwise_variants(step_counts):
    tag = str(variant["tag"])
    stepwise[tag] = {
        "reasoning_steps": int(variant["reasoning_steps"]),
        "train_data": "${TRAIN_DIR}/" + f"{run_name}_{tag}.jsonl",
        "model_path": "${OUTPUT_ROOT}/" + f"review-lora-{run_name}-{tag}-${MAX_STEPS}step",
        "eval": load_eval(tag),
    }

payload = {
    "run_name": run_name,
    "training_seed_source": {
        "builder": "data_collection/prepare_codecritic_axiom_seedset.py",
        "input_dataset": "datasets/CodeCriticBench/data/CodeCriticBench.jsonl",
        "seed_output": "${SEED_DATA}",
        "seed_metadata": "${SEED_META}",
    },
    "supervised_eval_seed_source": {
        "builder": "data_collection/scripts/run_axiom_clean_eval.sh -> prepare_eval_set()",
        "input_dataset_dir": "datasets/axiom-llm-judge/axiombench",
        "heldout_manifest_root": "${ROOT}/model_training/src/output/review-eval-<run>_*_axiom_eval",
        "exclude_training_source": "${SEED_DATA}",
        "per_grade": int("${EVAL_PER_GRADE}"),
    },
    "static_train": "${STATIC_TRAIN}",
    "direct_review_train": "${DIRECT_REVIEW_TRAIN}",
    "direct_review_alignment": json.loads(Path("${ALIGN_META}").read_text(encoding="utf-8")) if Path("${ALIGN_META}").exists() else None,
    "static_eval": load_eval("static"),
    "direct_review_eval": load_eval("direct_review"),
    "stepwise_variants": stepwise,
}
summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}

main() {
  local step_counts
  step_counts="$(step_count_list)"
  echo "[start] run=${RUN_NAME} steps=${step_counts} time=$(date -Is) log=${LOG_FILE}"

  prepare_seed_data
  generate_direct_review
  for step_count in ${step_counts}; do
    generate_stepwise_variant "${step_count}"
  done

  prepare_static_train
  prepare_bootstrap_train "${DIRECT_REVIEW_RAW}" "${DIRECT_REVIEW_TRAIN_RAW}" "final_review"
  align_direct_review_to_static
  for step_count in ${step_counts}; do
    prepare_bootstrap_train "$(step_raw_path "${step_count}")" "$(step_train_raw_path "${step_count}")" "path"
    cp "$(step_train_raw_path "${step_count}")" "$(step_train_path "${step_count}")"
  done

  if [[ "${STOP_AFTER_DATA:-0}" == "1" ]]; then
    CURRENT_STAGE="finished"
    echo "[finished] STOP_AFTER_DATA=1 run_dir=${RUN_DIR}"
    exit 0
  fi

  run_parallel_training_round "static" "${STATIC_TRAIN}" "${STATIC_MODEL}" "direct_review" "${DIRECT_REVIEW_TRAIN}" "${DIRECT_REVIEW_MODEL}"

  step_array=(${step_counts})
  if [[ "${#step_array[@]}" -gt 0 ]]; then
    idx=0
    while [[ "${idx}" -lt "${#step_array[@]}" ]]; do
      left_step="${step_array[${idx}]}"
      right_idx=$((idx + 1))
      if [[ "${right_idx}" -lt "${#step_array[@]}" ]]; then
        right_step="${step_array[${right_idx}]}"
        run_parallel_training_round \
          "$(step_tag "${left_step}")" "$(step_train_path "${left_step}")" "$(step_model_path "${left_step}")" \
          "$(step_tag "${right_step}")" "$(step_train_path "${right_step}")" "$(step_model_path "${right_step}")"
        idx=$((idx + 2))
      else
        CURRENT_STAGE="train_$(step_tag "${left_step}")"
        train_one "$(step_tag "${left_step}")" "$(step_train_path "${left_step}")" "$(step_model_path "${left_step}")" 0
        idx=$((idx + 1))
      fi
    done
  fi

  if [[ "${STOP_AFTER_TRAIN:-0}" == "1" ]]; then
    CURRENT_STAGE="finished"
    echo "[finished] STOP_AFTER_TRAIN=1 run_dir=${RUN_DIR}"
    exit 0
  fi

  eval_final_only_model "static" "${STATIC_MODEL}"
  eval_final_only_model "direct_review" "${DIRECT_REVIEW_MODEL}"
  for step_count in ${step_counts}; do
    eval_stepwise_model "${step_count}" "$(step_model_path "${step_count}")"
  done
  write_summary

  CURRENT_STAGE="finished"
  notify "FINISHED"
  echo "[finished] summary=${SUMMARY_JSON}"
}

main "$@"
