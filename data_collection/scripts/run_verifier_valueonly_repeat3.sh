#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/xianzhiwei/mcts-code-review"
RUN_NAME="verifier_valueonly_repeat3_20260420"
MODEL_PATH="/data1/xianzhiwei/model/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"

RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
EVAL_ROOT="${ROOT}/model_training/src/output/review-eval-${RUN_NAME}"
INDICES_FILE="${ROOT}/data_collection/review_mcts_runs/verifier_correction_overnight_20260419/heldout_indices.json"

BASE_DATA="${ROOT}/model_training/review_mcts_train_data/verifier_valueonly_baseline_20260420.jsonl"
VALUE_DATA="${ROOT}/model_training/review_mcts_train_data/verifier_valueonly_augmented_20260420.jsonl"
EXISTING_BASE_OUT="${ROOT}/model_training/src/output/review-lora-verifier-valueonly-baseline-160step"
EXISTING_VALUE_OUT="${ROOT}/model_training/src/output/review-lora-verifier-valueonly-augmented-160step"

SEEDS=(20260421 20260422 20260423)

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
  for path in "${BASE_DATA}" "${VALUE_DATA}" "${INDICES_FILE}"; do
    if [[ ! -f "${path}" ]]; then
      echo "Missing required input: ${path}" >&2
      exit 1
    fi
  done
  for path in "${EXISTING_BASE_OUT}/adapter_model.safetensors" "${EXISTING_VALUE_OUT}/adapter_model.safetensors"; do
    if [[ ! -f "${path}" ]]; then
      echo "Missing existing checkpoint for deterministic eval: ${path}" >&2
      exit 1
    fi
  done
}

train_one() {
  local tag="$1"
  local cuda_device="$2"
  local data_file="$3"
  local out_dir="$4"
  local seed="$5"
  local log_file="${LOG_DIR}/train_${tag}_seed${seed}.log"

  if [[ -f "${out_dir}/adapter_model.safetensors" && -f "${out_dir}/value_head.pth" ]]; then
    echo "[stage:train_${tag}_seed${seed}] final checkpoint exists, skipping: ${out_dir}"
    return
  fi

  (
    set -euo pipefail
    cd "${ROOT}/model_training/src"
    RESUME_ARGS=()
    LATEST_CHECKPOINT="$(find "${out_dir}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
    if [[ -n "${LATEST_CHECKPOINT}" ]]; then
      RESUME_ARGS=(--resume_from_checkpoint "${LATEST_CHECKPOINT}")
      echo "[stage:train_${tag}_seed${seed}] resuming from ${LATEST_CHECKPOINT}"
    fi
    echo "[stage:train_${tag}_seed${seed}] start=$(date -Is) cuda=${cuda_device}"
    CUDA_VISIBLE_DEVICES="${cuda_device}" \
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
      --datafile_paths "../review_mcts_train_data/$(basename "${data_file}")" \
      --output_dir "${out_dir}" \
      --max_steps 160 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --max_training_seq_length 1152 \
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
      --seed "${seed}" \
      "${RESUME_ARGS[@]}"
    echo "[stage:train_${tag}_seed${seed}] done=$(date -Is)"
  ) >"${log_file}" 2>&1 &
}

eval_one() {
  local tag="$1"
  local cuda_device="$2"
  local out_dir="$3"
  local eval_dir="$4"
  local log_file="$5"
  mkdir -p "${eval_dir}"

  if [[ -f "${eval_dir}/summary.json" ]]; then
    echo "[stage:eval_${tag}] summary exists, skipping: ${eval_dir}/summary.json"
    return
  fi

  (
    set -euo pipefail
    cd "${ROOT}/model_training/src"
    echo "[stage:eval_${tag}] start=$(date -Is) cuda=${cuda_device}"
    CUDA_VISIBLE_DEVICES="${cuda_device}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
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
      --temperature 0 \
      --top_p 1.0 \
      --final_temperature 0 \
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

wait_for_existing_deterministic_eval() {
  CURRENT_STAGE="eval_existing_det"
  eval_one "existing_baseline_det" 0 "${EXISTING_BASE_OUT}" "${EVAL_ROOT}/existing_baseline_det" "${LOG_DIR}/eval_existing_baseline_det.log"
  pid_base=$!
  eval_one "existing_valueonly_det" 1 "${EXISTING_VALUE_OUT}" "${EVAL_ROOT}/existing_valueonly_det" "${LOG_DIR}/eval_existing_valueonly_det.log"
  pid_value=$!
  echo "[stage:${CURRENT_STAGE}] waiting baseline pid=${pid_base}, valueonly pid=${pid_value}"
  wait "${pid_base}"
  wait "${pid_value}"
  echo "[stage:${CURRENT_STAGE}] done=$(date -Is)"
}

run_seed_pair() {
  local seed="$1"
  local base_out="${ROOT}/model_training/src/output/review-lora-verifier-valueonly-repeat3-baseline-seed${seed}-160step"
  local value_out="${ROOT}/model_training/src/output/review-lora-verifier-valueonly-repeat3-augmented-seed${seed}-160step"

  CURRENT_STAGE="train_seed${seed}"
  train_one "baseline" 0 "${BASE_DATA}" "${base_out}" "${seed}"
  pid_base=$!
  train_one "valueonly" 1 "${VALUE_DATA}" "${value_out}" "${seed}"
  pid_value=$!
  echo "[stage:${CURRENT_STAGE}] waiting baseline pid=${pid_base}, valueonly pid=${pid_value}"
  wait "${pid_base}"
  wait "${pid_value}"
  echo "[stage:${CURRENT_STAGE}] done=$(date -Is)"

  CURRENT_STAGE="eval_seed${seed}"
  eval_one "baseline_seed${seed}" 0 "${base_out}" "${EVAL_ROOT}/baseline_seed${seed}" "${LOG_DIR}/eval_baseline_seed${seed}.log"
  pid_base=$!
  eval_one "valueonly_seed${seed}" 1 "${value_out}" "${EVAL_ROOT}/valueonly_seed${seed}" "${LOG_DIR}/eval_valueonly_seed${seed}.log"
  pid_value=$!
  echo "[stage:${CURRENT_STAGE}] waiting baseline pid=${pid_base}, valueonly pid=${pid_value}"
  wait "${pid_base}"
  wait "${pid_value}"
  echo "[stage:${CURRENT_STAGE}] done=$(date -Is)"
}

write_comparison() {
  CURRENT_STAGE="summarize"
  PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import json
import statistics
from pathlib import Path

root = Path("/data1/xianzhiwei/mcts-code-review/model_training/src/output/review-eval-verifier_valueonly_repeat3_20260420")
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
for summary_path in sorted(root.glob("*/summary.json")):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows[summary_path.parent.name] = {metric: summary.get(metric) for metric in metrics}

groups = {"baseline": [], "valueonly": []}
for name, row in rows.items():
    if name.startswith("existing_"):
        continue
    if name.startswith("baseline_"):
        groups["baseline"].append(row)
    elif name.startswith("valueonly_"):
        groups["valueonly"].append(row)

aggregate = {}
for group_name, group_rows in groups.items():
    aggregate[group_name] = {}
    for metric in metrics:
        values = [row.get(metric) for row in group_rows if isinstance(row.get(metric), (int, float))]
        if values:
            aggregate[group_name][metric] = {
                "mean": round(statistics.fmean(values), 6),
                "min": round(min(values), 6),
                "max": round(max(values), 6),
                "n": len(values),
            }

comparison = {"rows": rows, "aggregate": aggregate}
out = root / "comparison.json"
out.write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
PY
}

echo "[pipeline] start=$(date -Is)"
echo "[pipeline] log=${LOG_FILE}"
require_inputs
wait_for_existing_deterministic_eval
for seed in "${SEEDS[@]}"; do
  run_seed_pair "${seed}"
  write_comparison
done
write_comparison

CURRENT_STAGE="finished"
echo "[pipeline] finished=$(date -Is)"
notify "COMPLETED"
