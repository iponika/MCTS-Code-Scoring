#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/xianzhiwei/mcts-code-review"
RUN_NAME="report_mean_mcts_repeat_20260421"
MODEL_PATH="${ROOT}/model_training/src/output/review-lora-report-static-mcts-valueonly-480step"
INDICES_FILE="${ROOT}/data_collection/review_mcts_runs/verifier_correction_overnight_20260419/heldout_indices.json"
EVAL_ROOT="${ROOT}/model_training/src/output/review-eval-${RUN_NAME}"
RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"

mkdir -p "${EVAL_ROOT}" "${RUN_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

NOTIFY_URL="${NTFY_URL:-https://ntfy.sh/iponika_mcts}"
CURRENT_STAGE="init"

notify() {
  local status="$1"
  local message="${RUN_NAME} ${status}: stage=${CURRENT_STAGE}, host=$(hostname), time=$(date -Is), log=${LOG_FILE}"
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    curl --noproxy '*' -fsS -d "${message}" "${NOTIFY_URL}" >/dev/null 2>&1 || true
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
  for path in "${MODEL_PATH}" "${INDICES_FILE}" "${ROOT}/datasets/CodeCriticBench/data/CodeCriticBench.jsonl"; do
    if [[ ! -e "${path}" ]]; then
      echo "Missing required input: ${path}" >&2
      exit 1
    fi
  done
}

eval_repeat() {
  local tag="$1"
  local cuda_device="$2"
  local seed="$3"
  local eval_dir="${EVAL_ROOT}/${tag}"
  local log_file="${LOG_DIR}/eval_${tag}.log"
  mkdir -p "${eval_dir}"
  (
    set -euo pipefail
    cd "${ROOT}/model_training/src"
    echo "[stage:eval_${tag}] start=$(date -Is) cuda=${cuda_device} seed=${seed}"
    CUDA_VISIBLE_DEVICES="${cuda_device}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_DATASETS_CACHE=/tmp/hf-datasets-cache \
    HF_HUB_OFFLINE=1 \
    TRL_EXPERIMENTAL_SILENCE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.batch_review_evaluator \
      --policy_model_path "${MODEL_PATH}" \
      --value_model_path "${MODEL_PATH}" \
      --share_policy_value_model \
      --input_record ../../datasets/CodeCriticBench/data/CodeCriticBench.jsonl \
      --record_indices_file "${INDICES_FILE}" \
      --output_dir "${eval_dir}" \
      --dimensions "Correctness Verification" \
      --device cuda \
      --dtype bf16 \
      --max_steps 3 \
      --num_candidates 2 \
      --max_new_tokens 256 \
      --final_max_new_tokens 384 \
      --temperature 0.7 \
      --top_p 0.95 \
      --score_key response_mean_value \
      --seed "${seed}" \
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

write_comparison() {
  CURRENT_STAGE="summarize"
  PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import json
import math
from pathlib import Path

root = Path("/data1/xianzhiwei/mcts-code-review")
out_root = root / "model_training/src/output/review-eval-report_mean_mcts_repeat_20260421"
summary_paths = {
    "base_direct_det": root / "model_training/src/output/review-eval-report_pilot_training_20260420/base_direct_det/summary.json",
    "trained_direct_det": root / "model_training/src/output/review-eval-report_pilot_training_20260420/report_direct_det/summary.json",
    "value_guided_last_value": root / "model_training/src/output/review-eval-report_pilot_training_20260420/report_value_guided_mcts/summary.json",
    "value_guided_mean_repeat0": root / "model_training/src/output/review-eval-value_score_key_ablation_20260421/response_mean_value/summary.json",
    "value_guided_mean_repeat1": out_root / "value_guided_mean_seed202604211/summary.json",
    "value_guided_mean_repeat2": out_root / "value_guided_mean_seed202604212/summary.json",
}
metrics = [
    "valid_rate",
    "grade_mae",
    "grade_median_abs_error",
    "boundary_acc",
    "low_grade_false_positive_rate",
    "high_grade_false_negative_rate",
    "unsupported_evidence_rate",
]

def pick(summary):
    return {metric: summary.get(metric) for metric in metrics} | {
        "valid_count": summary.get("valid_count"),
        "dimension_count": summary.get("dimension_count"),
    }

comparison = {}
mean_runs = []
for name, path in summary_paths.items():
    if not path.exists():
        continue
    summary = json.loads(path.read_text(encoding="utf-8"))
    comparison[name] = pick(summary)
    if name.startswith("value_guided_mean_repeat"):
        mean_runs.append(comparison[name])

aggregate = {"repeat_count": len(mean_runs)}
for metric in metrics:
    values = [row[metric] for row in mean_runs if isinstance(row.get(metric), (int, float))]
    if values:
        avg = sum(values) / len(values)
        var = sum((value - avg) ** 2 for value in values) / len(values)
        aggregate[f"{metric}_mean"] = round(avg, 6)
        aggregate[f"{metric}_std"] = round(math.sqrt(var), 6)
comparison["value_guided_mean_repeat_aggregate"] = aggregate

(out_root / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
PY
}

main() {
  require_inputs
  CURRENT_STAGE="eval_repeats"
  eval_repeat "value_guided_mean_seed202604211" 0 202604211
  pid_repeat1=$!
  eval_repeat "value_guided_mean_seed202604212" 1 202604212
  pid_repeat2=$!
  echo "[stage:${CURRENT_STAGE}] waiting repeat1 pid=${pid_repeat1}, repeat2 pid=${pid_repeat2}"
  wait "${pid_repeat1}"
  wait "${pid_repeat2}"
  write_comparison
  CURRENT_STAGE="finished"
  notify "FINISHED"
  echo "[finished] output=${EVAL_ROOT} log=${LOG_FILE}"
}

main "$@"
