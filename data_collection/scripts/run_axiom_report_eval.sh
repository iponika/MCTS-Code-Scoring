#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/xianzhiwei/mcts-code-review"
RUN_NAME="axiom_report_eval_v3_20260421"
BASE_MODEL_PATH="/data1/xianzhiwei/model/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
REPORT_MODEL_PATH="${ROOT}/model_training/src/output/review-lora-report-static-mcts-valueonly-480step"
AXIOM_DIR="${ROOT}/datasets/axiom-llm-judge/axiombench"
TRAIN_DATA="${ROOT}/model_training/review_mcts_train_data/report_static_mcts_valueonly_20260420.jsonl"
RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
EVAL_ROOT="${ROOT}/model_training/src/output/review-eval-${RUN_NAME}"
EVAL_FILE="${RUN_DIR}/axiom_heldout_balanced_60.jsonl"
INDICES_FILE="${RUN_DIR}/axiom_heldout_indices.json"

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${EVAL_ROOT}"
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

prepare_eval_set() {
  CURRENT_STAGE="prepare_eval_set"
  PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import json
import random
from collections import defaultdict
from pathlib import Path

root = Path("/data1/xianzhiwei/mcts-code-review")
axiom_dir = root / "datasets/axiom-llm-judge/axiombench"
train_data = root / "model_training/review_mcts_train_data/report_static_mcts_valueonly_20260420.jsonl"
out_file = root / "data_collection/review_mcts_runs/axiom_report_eval_v3_20260421/axiom_heldout_balanced_30.jsonl"
indices_file = root / "data_collection/review_mcts_runs/axiom_report_eval_v3_20260421/axiom_heldout_indices.json"

used = set()
if train_data.exists():
    for line in train_data.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("source") == "axiom":
            used.add(str(row.get("dataset_index")))

by_grade = defaultdict(list)
for file_path in sorted(axiom_dir.glob("*.jsonl")):
    subset = file_path.stem
    for row_idx, line in enumerate(file_path.open(encoding="utf-8")):
        row = json.loads(line)
        dataset_index = f"{subset}:{row_idx}"
        if dataset_index in used:
            continue
        row["source"] = "axiom"
        row["subset"] = subset
        row["dataset_index"] = dataset_index
        by_grade[int(row["score"])].append(row)

rng = random.Random(20260421)
selected = []
for grade in range(6):
    candidates = by_grade[grade]
    rng.shuffle(candidates)
    selected.extend(candidates[:5])
rng.shuffle(selected)

out_file.parent.mkdir(parents=True, exist_ok=True)
with out_file.open("w", encoding="utf-8") as handle:
    for row in selected:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
indices_file.write_text(json.dumps({"indices": list(range(len(selected)))}, indent=2) + "\n", encoding="utf-8")

print(json.dumps({
    "output": str(out_file),
    "indices": str(indices_file),
    "count": len(selected),
    "used_axiom_training_items": len(used),
    "grade_counts": {grade: sum(1 for row in selected if int(row["score"]) == grade) for grade in range(6)},
}, ensure_ascii=False, indent=2))
PY
}

eval_one() {
  local tag="$1"
  local cuda_device="$2"
  local policy_path="$3"
  local value_path="$4"
  local max_steps="$5"
  local num_candidates="$6"
  local temperature="$7"
  local top_p="$8"
  local max_rethinks="$9"
  local seed="${10}"
  local skip_value="${11}"
  local final_retries="${12}"
  local eval_dir="${EVAL_ROOT}/${tag}"
  local log_file="${LOG_DIR}/eval_${tag}.log"
  mkdir -p "${eval_dir}"
  (
    set -euo pipefail
    cd "${ROOT}/model_training/src"
    echo "[stage:eval_${tag}] start=$(date -Is) cuda=${cuda_device} seed=${seed}"
    SKIP_VALUE_ARGS=()
    if [[ "${skip_value}" == "skip_value" ]]; then
      SKIP_VALUE_ARGS=(--skip_value_scoring)
    fi
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
      --share_policy_value_model \
      "${SKIP_VALUE_ARGS[@]}" \
      --input_record "${EVAL_FILE}" \
      --record_indices_file "${INDICES_FILE}" \
      --output_dir "${eval_dir}" \
      --dimensions "Correctness Verification" \
      --device cuda \
      --dtype bf16 \
      --max_steps "${max_steps}" \
      --num_candidates "${num_candidates}" \
      --max_new_tokens 192 \
      --final_max_new_tokens 256 \
      --temperature "${temperature}" \
      --top_p "${top_p}" \
      --score_key response_mean_value \
      --seed "${seed}" \
      --final_temperature 0 \
      --rethink_threshold -0.2 \
      --max_rethinks "${max_rethinks}" \
      --max_final_retries "${final_retries}"
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

root = Path("/data1/xianzhiwei/mcts-code-review/model_training/src/output/review-eval-axiom_report_eval_v3_20260421")
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
for name in ["base_direct", "trained_direct", "trained_value_guided_mean"]:
    summary = json.loads((root / name / "summary.json").read_text(encoding="utf-8"))
    comparison[name] = {metric: summary.get(metric) for metric in metrics}
    comparison[name]["valid_count"] = summary.get("valid_count")
    comparison[name]["dimension_count"] = summary.get("dimension_count")
(root / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
PY
}

main() {
  prepare_eval_set
  CURRENT_STAGE="direct_evals"
  eval_one "base_direct" 0 "${BASE_MODEL_PATH}" "${BASE_MODEL_PATH}" 1 1 0 1.0 0 202604213 skip_value 2
  pid_base=$!
  eval_one "trained_direct" 1 "${REPORT_MODEL_PATH}" "${REPORT_MODEL_PATH}" 1 1 0 1.0 0 202604214 skip_value 2
  pid_trained=$!
  echo "[stage:${CURRENT_STAGE}] waiting base pid=${pid_base}, trained pid=${pid_trained}"
  wait "${pid_base}"
  wait "${pid_trained}"

  CURRENT_STAGE="value_guided_eval"
  eval_one "trained_value_guided_mean" 0 "${REPORT_MODEL_PATH}" "${REPORT_MODEL_PATH}" 3 2 0.7 0.95 1 202604215 value 0
  pid_mcts=$!
  echo "[stage:${CURRENT_STAGE}] waiting mcts pid=${pid_mcts}"
  wait "${pid_mcts}"

  write_comparison
  CURRENT_STAGE="finished"
  notify "FINISHED"
  echo "[finished] output=${EVAL_ROOT} log=${LOG_FILE}"
}

main "$@"
