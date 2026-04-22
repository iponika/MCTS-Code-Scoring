#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data1/xianzhiwei/mcts-code-review}"
RUN_NAME="${RUN_NAME:-axiom_clean_eval_no0_20260422}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
TRAINED_MODEL_PATH="${TRAINED_MODEL_PATH:-}"
AXIOM_DIR="${AXIOM_DIR:-${ROOT}/datasets/axiom-llm-judge/axiombench}"
TRAIN_DATA="${TRAIN_DATA:-${ROOT}/model_training/review_mcts_train_data/principle_generalization_qwen3_4b_full_context_no_axiom0_20260422.jsonl}"
RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
EVAL_ROOT="${ROOT}/model_training/src/output/review-eval-${RUN_NAME}"
EVAL_FILE="${RUN_DIR}/axiom_clean_heldout.jsonl"
INDICES_FILE="${RUN_DIR}/axiom_clean_indices.json"
PER_GRADE="${PER_GRADE:-5}"
DROP_AXIOM_GRADE_ZERO="${DROP_AXIOM_GRADE_ZERO:-1}"
MAX_PROBLEM_CHARS="${MAX_PROBLEM_CHARS:-12000}"
MAX_CODE_CHARS="${MAX_CODE_CHARS:-12000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
FINAL_MAX_NEW_TOKENS="${FINAL_MAX_NEW_TOKENS:-320}"
NTFY_URL="${NTFY_URL:-https://ntfy.sh/iponika_mcts}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${EVAL_ROOT}"
LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE="init"

notify() {
  local status="$1"
  local message="${RUN_NAME} ${status}: stage=${CURRENT_STAGE}, host=$(hostname), time=$(date -Is), log=${LOG_FILE}, output=${EVAL_ROOT}"
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

prepare_eval_set() {
  CURRENT_STAGE="prepare_eval_set"
  PYTHONDONTWRITEBYTECODE=1 python - <<PY
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

axiom_dir = Path("${AXIOM_DIR}")
train_data = Path("${TRAIN_DATA}")
out_file = Path("${EVAL_FILE}")
indices_file = Path("${INDICES_FILE}")
per_grade = int("${PER_GRADE}")
drop_zero = "${DROP_AXIOM_GRADE_ZERO}".strip().lower() not in {"0", "false", "no", "off"}

used = set()
if train_data.exists():
    for line in train_data.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("source") == "axiom" and row.get("dataset_index") is not None:
            used.add(str(row.get("dataset_index")))

by_grade = defaultdict(list)
for file_path in sorted(axiom_dir.glob("*.jsonl")):
    subset = file_path.stem
    for row_idx, line in enumerate(file_path.open(encoding="utf-8")):
        if not line.strip():
            continue
        row = json.loads(line)
        grade = int(row["score"])
        if drop_zero and grade == 0:
            continue
        dataset_index = f"{subset}:{row_idx}"
        if dataset_index in used:
            continue
        row["source"] = "axiom"
        row["subset"] = subset
        row["dataset_index"] = dataset_index
        by_grade[grade].append(row)

rng = random.Random(20260422)
selected = []
grades = [grade for grade in range(6) if not (drop_zero and grade == 0)]
for grade in grades:
    candidates = by_grade[grade]
    rng.shuffle(candidates)
    selected.extend(candidates[:per_grade])
rng.shuffle(selected)

out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\\n" for row in selected), encoding="utf-8")
indices_file.write_text(json.dumps({"indices": list(range(len(selected)))}, indent=2) + "\\n", encoding="utf-8")

summary = {
    "output": str(out_file),
    "indices": str(indices_file),
    "count": len(selected),
    "drop_axiom_grade_zero": drop_zero,
    "per_grade": per_grade,
    "used_axiom_training_items": len(used),
    "available_grade_counts": {str(grade): len(by_grade[grade]) for grade in grades},
    "selected_grade_counts": dict(Counter(str(int(row["score"])) for row in selected)),
}
(out_file.parent / "axiom_clean_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
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
  if [[ -f "${eval_dir}/summary.json" ]]; then
    echo "[stage:eval_${tag}] summary exists, skipping"
    return
  fi
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
      --input_record "${EVAL_FILE}" \
      --record_indices_file "${INDICES_FILE}" \
      --output_dir "${eval_dir}" \
      --dimensions "Correctness Verification" \
      --device cuda \
      --dtype bf16 \
      --max_steps "${max_steps}" \
      --num_candidates "${num_candidates}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --final_max_new_tokens "${FINAL_MAX_NEW_TOKENS}" \
      --temperature "${temperature}" \
      --top_p "${top_p}" \
      --score_key response_mean_value \
      --seed "${seed}" \
      --final_only_json \
      --max_problem_chars "${MAX_PROBLEM_CHARS}" \
      --max_code_chars "${MAX_CODE_CHARS}" \
      --no-mark_code_truncation_inside_block \
      --format_penalty 1.0 \
      --low_grade_no_evidence_penalty 0.4 \
      --final_temperature 0 \
      --max_final_retries 2
    PYTHONPATH="${ROOT}:${ROOT}/model_training/src:${ROOT}/data_collection" \
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
  PYTHONDONTWRITEBYTECODE=1 python - <<PY
import json
from pathlib import Path

eval_root = Path("${EVAL_ROOT}")
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
    "unsupported_evidence_rate",
]
comparison = {}
for summary_path in sorted(eval_root.glob("*/summary.json")):
    tag = summary_path.parent.name
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comparison[tag] = {metric: summary.get(metric) for metric in metrics}
    comparison[tag]["valid_count"] = summary.get("valid_count")
    comparison[tag]["dimension_count"] = summary.get("dimension_count")
(eval_root / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
PY
}

main() {
  echo "[start] run=${RUN_NAME} time=$(date -Is) log=${LOG_FILE}"
  prepare_eval_set
  if [[ "${STOP_AFTER_MANIFEST:-0}" == "1" ]]; then
    CURRENT_STAGE="finished"
    echo "[finished] STOP_AFTER_MANIFEST=1 output=${RUN_DIR} log=${LOG_FILE}"
    exit 0
  fi

  CURRENT_STAGE="eval_base_direct"
  eval_one "base_direct_clean" 0 "${BASE_MODEL_PATH}" "${BASE_MODEL_PATH}" "share" "skip_value" 1 1 0 1.0 202604221

  if [[ -n "${TRAINED_MODEL_PATH}" && -d "${TRAINED_MODEL_PATH}" ]]; then
    CURRENT_STAGE="eval_trained"
    eval_one "trained_direct_clean" 0 "${TRAINED_MODEL_PATH}" "${TRAINED_MODEL_PATH}" "share" "skip_value" 1 1 0 1.0 202604222
    eval_one "trained_value_rerank_clean" 0 "${TRAINED_MODEL_PATH}" "${TRAINED_MODEL_PATH}" "share" "use_value" 1 2 0.7 0.95 202604223
  else
    echo "[stage:eval_trained] TRAINED_MODEL_PATH is empty or missing; skipping trained evaluations."
  fi

  write_comparison
  CURRENT_STAGE="finished"
  notify "FINISHED"
  echo "[finished] output=${EVAL_ROOT} log=${LOG_FILE}"
}

main "$@"
