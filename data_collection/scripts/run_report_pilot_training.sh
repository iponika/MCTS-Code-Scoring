#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/xianzhiwei/mcts-code-review"
RUN_NAME="report_pilot_training_20260420"
MODEL_PATH="/data1/xianzhiwei/model/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"

RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
EVAL_ROOT="${ROOT}/model_training/src/output/review-eval-${RUN_NAME}"
INDICES_FILE="${ROOT}/data_collection/review_mcts_runs/verifier_correction_overnight_20260419/heldout_indices.json"

STATIC_FULL="${ROOT}/model_training/review_mcts_train_data/axiom_codecritic_static_full.jsonl"
STATIC_FILTERED="${ROOT}/model_training/review_mcts_train_data/report_static_filtered1152_1600_20260420.jsonl"
MCTS_VALUEONLY="${ROOT}/model_training/review_mcts_train_data/verifier_valueonly_augmented_20260420.jsonl"
COMBINED_DATA="${ROOT}/model_training/review_mcts_train_data/report_static_mcts_valueonly_20260420.jsonl"
REPORT_OUT="${ROOT}/model_training/src/output/review-lora-report-static-mcts-valueonly-480step"

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

require_inputs() {
  CURRENT_STAGE="require_inputs"
  for path in "${STATIC_FULL}" "${MCTS_VALUEONLY}" "${INDICES_FILE}"; do
    if [[ ! -f "${path}" ]]; then
      echo "Missing required input: ${path}" >&2
      exit 1
    fi
  done
}

prepare_data() {
  CURRENT_STAGE="prepare_data"
  cd "${ROOT}"
  if [[ ! -f "${STATIC_FILTERED}" ]]; then
    echo "[stage:${CURRENT_STAGE}] filtering static AXIOM+CodeCritic data start=$(date -Is)"
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${ROOT}/model_training/src" \
    TRL_EXPERIMENTAL_SILENCE=1 \
    HF_HUB_OFFLINE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.filter_review_train_data \
      --input_file "${STATIC_FULL}" \
      --output_file "${STATIC_FILTERED}" \
      --model_key Qwen/Qwen3.5-9B \
      --model_name_or_path "${MODEL_PATH}" \
      --max_tokens 1152 \
      --max_items 1600 \
      --shuffle_seed 20260420
  else
    echo "[stage:${CURRENT_STAGE}] static filtered exists: ${STATIC_FILTERED}"
  fi

  if [[ ! -f "${COMBINED_DATA}" ]]; then
    echo "[stage:${CURRENT_STAGE}] combining static value labels with repeated MCTS value-only paths start=$(date -Is)"
    PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import json
import random
from pathlib import Path

static_path = Path("/data1/xianzhiwei/mcts-code-review/model_training/review_mcts_train_data/report_static_filtered1152_1600_20260420.jsonl")
mcts_path = Path("/data1/xianzhiwei/mcts-code-review/model_training/review_mcts_train_data/verifier_valueonly_augmented_20260420.jsonl")
out_path = Path("/data1/xianzhiwei/mcts-code-review/model_training/review_mcts_train_data/report_static_mcts_valueonly_20260420.jsonl")

items = []
for line in static_path.read_text(encoding="utf-8").splitlines():
    if line.strip():
        item = json.loads(line)
        item["data_mix"] = "static_axiom_codecritic"
        items.append(item)

mcts_rows = [json.loads(line) for line in mcts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
for repeat_index in range(3):
    for row in mcts_rows:
        item = dict(row)
        item["data_mix"] = "mcts_valueonly"
        if repeat_index:
            item["terminal_tag"] = f"{item.get('terminal_tag')}#report_repeat{repeat_index}"
            item["report_repeat_index"] = repeat_index
        items.append(item)

random.Random(20260420).shuffle(items)
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as handle:
    for item in items:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")

print(json.dumps({
    "output_file": str(out_path),
    "items": len(items),
    "policy_items": sum(1 for item in items if item.get("train_lm")),
    "value_only_items": sum(1 for item in items if not item.get("train_lm")),
    "static_items": sum(1 for item in items if item.get("data_mix") == "static_axiom_codecritic"),
    "mcts_items": sum(1 for item in items if item.get("data_mix") == "mcts_valueonly"),
}, ensure_ascii=False, indent=2, sort_keys=True))
PY
  else
    echo "[stage:${CURRENT_STAGE}] combined data exists: ${COMBINED_DATA}"
  fi
  wc -l "${STATIC_FILTERED}" "${MCTS_VALUEONLY}" "${COMBINED_DATA}"
}

train_report_model() {
  local log_file="${LOG_DIR}/train_report.log"
  if [[ -f "${REPORT_OUT}/adapter_model.safetensors" && -f "${REPORT_OUT}/value_head.pth" ]]; then
    echo "[stage:train_report] final checkpoint exists, skipping: ${REPORT_OUT}"
    return
  fi
  (
    set -euo pipefail
    cd "${ROOT}/model_training/src"
    RESUME_ARGS=()
    LATEST_CHECKPOINT="$(find "${REPORT_OUT}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
    if [[ -n "${LATEST_CHECKPOINT}" ]]; then
      RESUME_ARGS=(--resume_from_checkpoint "${LATEST_CHECKPOINT}")
      echo "[stage:train_report] resuming from ${LATEST_CHECKPOINT}"
    fi
    echo "[stage:train_report] start=$(date -Is) cuda=0"
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_DATASETS_CACHE=/tmp/hf-datasets-cache \
    HF_HUB_OFFLINE=1 \
    TRL_EXPERIMENTAL_SILENCE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.train_multi \
      --task review \
      --model_key Qwen/Qwen3.5-9B \
      --model_name_or_path "${MODEL_PATH}" \
      --datafile_paths "../review_mcts_train_data/$(basename "${COMBINED_DATA}")" \
      --output_dir "${REPORT_OUT}" \
      --max_steps 480 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --max_training_seq_length 1152 \
      --bf16 True \
      --logging_steps 20 \
      --save_strategy steps \
      --save_steps 160 \
      --save_total_limit 2 \
      --report_to none \
      --optim adafactor \
      --learning_rate 4e-5 \
      --lr_scheduler_type linear \
      --warmup_steps 32 \
      --peft lora \
      --value_weight 0.05 \
      --num_proc 1 \
      --seed 20260420 \
      "${RESUME_ARGS[@]}"
    echo "[stage:train_report] done=$(date -Is)"
  ) >"${log_file}" 2>&1 &
}

eval_one() {
  local tag="$1"
  local cuda_device="$2"
  local policy_path="$3"
  local value_path="$4"
  local share_flag="$5"
  local max_steps="$6"
  local num_candidates="$7"
  local temperature="$8"
  local top_p="$9"
  local max_rethinks="${10}"
  local eval_dir="${EVAL_ROOT}/${tag}"
  local log_file="${LOG_DIR}/eval_${tag}.log"
  mkdir -p "${eval_dir}"

  if [[ -f "${eval_dir}/summary.json" ]]; then
    echo "[stage:eval_${tag}] summary exists, skipping: ${eval_dir}/summary.json"
    return
  fi

  (
    set -euo pipefail
    cd "${ROOT}/model_training/src"
    SHARE_ARGS=()
    if [[ "${share_flag}" == "share" ]]; then
      SHARE_ARGS=(--share_policy_value_model)
    fi
    echo "[stage:eval_${tag}] start=$(date -Is) cuda=${cuda_device}"
    CUDA_VISIBLE_DEVICES="${cuda_device}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_DATASETS_CACHE=/tmp/hf-datasets-cache \
    HF_HUB_OFFLINE=1 \
    TRL_EXPERIMENTAL_SILENCE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.batch_review_evaluator \
      --policy_model_path "${policy_path}" \
      --value_model_path "${value_path}" \
      "${SHARE_ARGS[@]}" \
      --input_record ../../datasets/CodeCriticBench/data/CodeCriticBench.jsonl \
      --record_indices_file "${INDICES_FILE}" \
      --output_dir "${eval_dir}" \
      --dimensions "Correctness Verification" \
      --device cuda \
      --dtype bf16 \
      --max_steps "${max_steps}" \
      --num_candidates "${num_candidates}" \
      --max_new_tokens 256 \
      --final_max_new_tokens 384 \
      --temperature "${temperature}" \
      --top_p "${top_p}" \
      --final_temperature 0 \
      --rethink_threshold -0.2 \
      --max_rethinks "${max_rethinks}" \
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

run_base_eval_in_background() {
  CURRENT_STAGE="base_direct_eval"
  eval_one "base_direct_det" 1 "${MODEL_PATH}" "${MODEL_PATH}" "share" 1 1 0 1.0 0
  echo $! > "${RUN_DIR}/base_eval.pid"
  echo "[stage:${CURRENT_STAGE}] launched pid=$(cat "${RUN_DIR}/base_eval.pid")"
}

wait_for_training_and_base_eval() {
  CURRENT_STAGE="train_report_parallel_base_eval"
  train_report_model
  train_pid=$!
  echo "[stage:${CURRENT_STAGE}] waiting train pid=${train_pid}"
  wait "${train_pid}"
  if [[ -f "${RUN_DIR}/base_eval.pid" ]]; then
    base_pid="$(cat "${RUN_DIR}/base_eval.pid")"
    echo "[stage:${CURRENT_STAGE}] waiting base eval pid=${base_pid}"
    wait "${base_pid}" || true
  fi
  echo "[stage:${CURRENT_STAGE}] done=$(date -Is)"
}

run_report_evals() {
  CURRENT_STAGE="report_evals"
  eval_one "report_direct_det" 0 "${REPORT_OUT}" "${REPORT_OUT}" "share" 1 1 0 1.0 0
  pid_direct=$!
  eval_one "report_value_guided_mcts" 1 "${REPORT_OUT}" "${REPORT_OUT}" "share" 3 2 0.7 0.95 1
  pid_mcts=$!
  echo "[stage:${CURRENT_STAGE}] waiting report direct pid=${pid_direct}, report mcts pid=${pid_mcts}"
  wait "${pid_direct}"
  wait "${pid_mcts}"
  echo "[stage:${CURRENT_STAGE}] done=$(date -Is)"
}

write_comparison() {
  CURRENT_STAGE="summarize"
  PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import json
from pathlib import Path

root = Path("/data1/xianzhiwei/mcts-code-review/model_training/src/output/review-eval-report_pilot_training_20260420")
metrics = [
    "valid_rate",
    "grade_mae",
    "grade_median_abs_error",
    "boundary_acc",
    "low_grade_false_positive_rate",
    "high_grade_false_negative_rate",
    "unsupported_evidence_rate",
]
rows = {}
for name in ["base_direct_det", "report_direct_det", "report_value_guided_mcts"]:
    path = root / name / "summary.json"
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows[name] = {metric: summary.get(metric) for metric in metrics}
out = root / "comparison.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
PY
}

echo "[pipeline] start=$(date -Is)"
echo "[pipeline] log=${LOG_FILE}"
require_inputs
prepare_data
run_base_eval_in_background
wait_for_training_and_base_eval
run_report_evals
write_comparison

CURRENT_STAGE="finished"
echo "[pipeline] finished=$(date -Is)"
notify "COMPLETED"
