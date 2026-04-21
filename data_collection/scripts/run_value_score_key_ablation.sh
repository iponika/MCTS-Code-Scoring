#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/xianzhiwei/mcts-code-review"
RUN_NAME="value_score_key_ablation_20260421"
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

eval_score_key() {
  local tag="$1"
  local score_key="$2"
  local cuda_device="$3"
  local eval_dir="${EVAL_ROOT}/${tag}"
  local log_file="${LOG_DIR}/eval_${tag}.log"
  mkdir -p "${eval_dir}"
  (
    set -euo pipefail
    cd "${ROOT}/model_training/src"
    echo "[stage:eval_${tag}] start=$(date -Is) cuda=${cuda_device} score_key=${score_key}"
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
      --score_key "${score_key}" \
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
from pathlib import Path

root = Path("/data1/xianzhiwei/mcts-code-review/model_training/src/output/review-eval-value_score_key_ablation_20260421")
metrics = [
    "valid_rate",
    "grade_mae",
    "grade_median_abs_error",
    "boundary_acc",
    "low_grade_false_positive_rate",
    "high_grade_false_negative_rate",
    "unsupported_evidence_rate",
]
comparison = {}
for name in ["response_mean_value", "response_conservative_value"]:
    summary_path = root / name / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comparison[name] = {metric: summary.get(metric) for metric in metrics}
    comparison[name]["valid_count"] = summary.get("valid_count")
    comparison[name]["dimension_count"] = summary.get("dimension_count")
(root / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
PY
}

main() {
  require_inputs
  CURRENT_STAGE="evals"
  eval_score_key "response_mean_value" "response_mean_value" 0
  pid_mean=$!
  eval_score_key "response_conservative_value" "response_conservative_value" 1
  pid_conservative=$!
  echo "[stage:${CURRENT_STAGE}] waiting mean pid=${pid_mean}, conservative pid=${pid_conservative}"
  wait "${pid_mean}"
  wait "${pid_conservative}"
  write_comparison
  CURRENT_STAGE="finished"
  notify "FINISHED"
  echo "[finished] output=${EVAL_ROOT} log=${LOG_FILE}"
}

main "$@"
