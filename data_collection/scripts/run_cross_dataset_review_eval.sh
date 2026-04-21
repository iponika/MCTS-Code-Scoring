#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data1/xianzhiwei/mcts-code-review}"
RUN_NAME="${RUN_NAME:-cross_dataset_review_eval_20260421}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/data1/xianzhiwei/model/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
REPORT_MODEL_PATH="${REPORT_MODEL_PATH:-${ROOT}/model_training/src/output/review-lora-report-static-mcts-valueonly-480step}"
RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}"
MANIFEST_DIR="${RUN_DIR}/manifests"
LOG_DIR="${RUN_DIR}/logs"
EVAL_ROOT="${ROOT}/model_training/src/output/review-eval-${RUN_NAME}"
ALL_MANIFEST="${MANIFEST_DIR}/all.jsonl"
ALL_INDICES="${MANIFEST_DIR}/all_indices.json"
PER_GRADE="${PER_GRADE:-5}"
PER_BINARY_LABEL="${PER_BINARY_LABEL:-10}"
PAIR_COUNT="${PAIR_COUNT:-10}"
NTFY_URL="${NTFY_URL:-https://ntfy.sh/iponika_mcts}"

mkdir -p "${MANIFEST_DIR}" "${LOG_DIR}" "${EVAL_ROOT}"
LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE="init"

notify() {
  local status="$1"
  local message="${RUN_NAME} ${status}: stage=${CURRENT_STAGE}, host=$(hostname), time=$(date -Is), log=${LOG_FILE}, output=${EVAL_ROOT}"
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    curl --noproxy '*' -fsS -d "${message}" "${NTFY_URL}" >/dev/null 2>&1 || true
}

on_exit() {
  local code=$?
  if [[ "${CURRENT_STAGE}" != "finished" ]]; then
    notify "FAILED(code=${code})"
  fi
  exit "${code}"
}
trap on_exit EXIT

build_manifests() {
  CURRENT_STAGE="build_manifests"
  cd "${ROOT}"
  PYTHONPATH="${ROOT}/model_training/src:${ROOT}/data_collection" \
  PYTHONDONTWRITEBYTECODE=1 \
  UV_CACHE_DIR=/tmp/uv-cache \
  uv run python "${ROOT}/data_collection/scripts/build_multidataset_eval_manifest.py" \
    --output_dir "${MANIFEST_DIR}" \
    --per_grade "${PER_GRADE}" \
    --per_binary_label "${PER_BINARY_LABEL}" \
    --pair_count "${PAIR_COUNT}" \
    --seed 20260421
}

combine_manifests() {
  CURRENT_STAGE="combine_manifests"
  PYTHONDONTWRITEBYTECODE=1 python - <<PY
import json
from pathlib import Path

manifest_dir = Path("${MANIFEST_DIR}")
all_manifest = Path("${ALL_MANIFEST}")
all_indices = Path("${ALL_INDICES}")
rows = []
for name in ["codecritic", "axiom", "code_diting", "codejudgebench"]:
    path = manifest_dir / f"{name}.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
all_manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
all_indices.write_text(json.dumps({"indices": list(range(len(rows)))}, indent=2) + "\n", encoding="utf-8")
counts = {}
for row in rows:
    counts[row.get("source", "unknown")] = counts.get(row.get("source", "unknown"), 0) + 1
summary = {"path": str(all_manifest), "indices": str(all_indices), "count": len(rows), "source_counts": counts}
(manifest_dir / "all_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

eval_one() {
  local tag="$1"
  local cuda_device="$2"
  local policy_path="$3"
  local value_path="$4"
  local share_mode="$5"
  local skip_value="$6"
  local max_steps="$7"
  local num_candidates="$8"
  local temperature="$9"
  local top_p="${10}"
  local seed="${11}"
  local eval_dir="${EVAL_ROOT}/${tag}"
  local log_file="${LOG_DIR}/eval_${tag}.log"
  mkdir -p "${eval_dir}"

  (
    set -euo pipefail
    cd "${ROOT}/model_training/src"
    echo "[stage:eval_${tag}] start=$(date -Is) cuda=${cuda_device} seed=${seed}"
    extra_args=()
    if [[ "${share_mode}" == "share" ]]; then
      extra_args+=(--share_policy_value_model)
    fi
    if [[ "${skip_value}" == "skip_value" ]]; then
      extra_args+=(--skip_value_scoring)
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
      "${extra_args[@]}" \
      --input_record "${ALL_MANIFEST}" \
      --record_indices_file "${ALL_INDICES}" \
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
      --final_only_json \
      --format_penalty 1.0 \
      --low_grade_no_evidence_penalty 0.4 \
      --final_temperature 0 \
      --max_final_retries 2
    PYTHONPATH="${ROOT}/model_training/src:${ROOT}/data_collection" \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python "${ROOT}/data_collection/scripts/summarize_review_eval_outputs.py" \
      --eval_dir "${eval_dir}" \
      --output "${eval_dir}/summary.json"
    echo "[stage:eval_${tag}] done=$(date -Is)"
  ) >"${log_file}" 2>&1
}

write_comparison() {
  CURRENT_STAGE="write_comparison"
  PYTHONPATH="${ROOT}:${ROOT}/model_training/src:${ROOT}/data_collection" \
  PYTHONDONTWRITEBYTECODE=1 \
  UV_CACHE_DIR=/tmp/uv-cache \
  uv run python - <<PY
import json
from collections import defaultdict
from pathlib import Path
from data_collection.scripts.summarize_review_eval_outputs import mean, median, pairwise_stats

eval_root = Path("${EVAL_ROOT}")
tags = ["base_direct_final", "trained_direct_final", "trained_value_rerank_final"]
metrics = [
    "valid_rate",
    "grade_mae",
    "grade_median_abs_error",
    "boundary_acc",
    "low_grade_false_positive_rate",
    "high_grade_false_negative_rate",
    "lenient_rate",
    "lenient_grade_mae",
    "lenient_boundary_acc",
    "interval_acc",
    "lenient_interval_acc",
    "pairwise_acc",
    "lenient_pairwise_acc",
    "unsupported_evidence_rate",
]

def source_breakdown(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("source") or "unknown"].append(row)
    output = {}
    for source, source_rows in sorted(grouped.items()):
        valid_rows = [row for row in source_rows if row.get("valid")]
        grade_rows = [row for row in valid_rows if "abs_grade_error" in row]
        lenient_rows = [row for row in source_rows if "lenient_abs_grade_error" in row]
        interval_rows = [row for row in valid_rows if "interval_correct" in row]
        lenient_interval_rows = [row for row in source_rows if "lenient_interval_correct" in row]
        pairwise = pairwise_stats(source_rows, grade_key="parsed_grade")
        lenient_pairwise = pairwise_stats(source_rows, grade_key="lenient_grade")
        output[source] = {
            "dimension_count": len(source_rows),
            "valid_rate": round(len(valid_rows) / max(1, len(source_rows)), 6),
            "grade_mae": mean([row["abs_grade_error"] for row in grade_rows]),
            "grade_median_abs_error": median([row["abs_grade_error"] for row in grade_rows]),
            "boundary_acc": mean([1.0 if row["boundary_correct"] else 0.0 for row in grade_rows]),
            "lenient_rate": round(len(lenient_rows) / max(1, len(source_rows)), 6),
            "lenient_grade_mae": mean([row["lenient_abs_grade_error"] for row in lenient_rows]),
            "lenient_boundary_acc": mean([1.0 if row["lenient_boundary_correct"] else 0.0 for row in lenient_rows]),
            "interval_acc": mean([1.0 if row["interval_correct"] else 0.0 for row in interval_rows]),
            "lenient_interval_acc": mean([1.0 if row["lenient_interval_correct"] else 0.0 for row in lenient_interval_rows]),
            "pairwise_acc": pairwise["pairwise_acc"],
            "pairwise_count": pairwise["pairwise_count"],
            "lenient_pairwise_acc": lenient_pairwise["pairwise_acc"],
            "lenient_pairwise_count": lenient_pairwise["pairwise_count"],
        }
    return output

comparison = {}
for tag in tags:
    summary_path = eval_root / tag / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comparison[tag] = {metric: summary.get(metric) for metric in metrics}
    comparison[tag]["dimension_count"] = summary.get("dimension_count")
    comparison[tag]["valid_count"] = summary.get("valid_count")
    comparison[tag]["by_source"] = source_breakdown(summary.get("rows") or [])
(eval_root / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
PY
}

main() {
  echo "[start] run=${RUN_NAME} time=$(date -Is) log=${LOG_FILE}"
  build_manifests
  combine_manifests
  if [[ "${STOP_AFTER_MANIFEST:-0}" == "1" ]]; then
    CURRENT_STAGE="finished"
    echo "[finished] STOP_AFTER_MANIFEST=1 output=${MANIFEST_DIR} log=${LOG_FILE}"
    exit 0
  fi

  CURRENT_STAGE="direct_evals"
  eval_one "base_direct_final" 0 "${BASE_MODEL_PATH}" "${REPORT_MODEL_PATH}" "no_share" "skip_value" 1 1 0 1.0 202604211 &
  pid_base=$!
  eval_one "trained_direct_final" 1 "${REPORT_MODEL_PATH}" "${REPORT_MODEL_PATH}" "share" "skip_value" 1 1 0 1.0 202604212 &
  pid_trained=$!
  echo "[stage:${CURRENT_STAGE}] waiting base=${pid_base} trained=${pid_trained}"
  wait "${pid_base}"
  wait "${pid_trained}"

  CURRENT_STAGE="value_rerank_eval"
  eval_one "trained_value_rerank_final" 0 "${REPORT_MODEL_PATH}" "${REPORT_MODEL_PATH}" "share" "use_value" 1 2 0.7 0.95 202604213

  write_comparison
  CURRENT_STAGE="finished"
  notify "FINISHED"
  echo "[finished] output=${EVAL_ROOT} log=${LOG_FILE}"
}

main "$@"
