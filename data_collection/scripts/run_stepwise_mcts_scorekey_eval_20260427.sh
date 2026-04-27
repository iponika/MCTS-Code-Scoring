#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data1/xianzhiwei/mcts-code-review}"
BASE_RUN="${BASE_RUN:-bootstrap_cmp_stageq_branch2_overnight_v2_20260426}"
RUN_PREFIX="${RUN_PREFIX:-stepwise_mcts_scorekey_20260427}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
MCTS_MODEL_PATH="${MCTS_MODEL_PATH:-${ROOT}/model_training/src/output/review-lora-${BASE_RUN}-mcts-240step}"
TRAIN_DATA="${TRAIN_DATA:-${ROOT}/data_collection/review_mcts_runs/${BASE_RUN}/seed_codecritic_axiom.jsonl}"
PER_GRADE="${PER_GRADE:-12}"
STEPWISE_MAX_STEPS="${STEPWISE_MAX_STEPS:-3}"
STEPWISE_NUM_CANDIDATES="${STEPWISE_NUM_CANDIDATES:-3}"
MAX_RETHINKS="${MAX_RETHINKS:-1}"
RETHINK_THRESHOLD="${RETHINK_THRESHOLD:--0.2}"
RETHINK_SPREAD_THRESHOLD="${RETHINK_SPREAD_THRESHOLD:-0.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
FINAL_MAX_NEW_TOKENS="${FINAL_MAX_NEW_TOKENS:-320}"
NTFY_URL="${NTFY_URL:-https://ntfy.sh/iponika_mcts}"
OUTER_LOG="${OUTER_LOG:-/tmp/stepwise_mcts_scorekey_eval_20260427.log}"

exec > >(tee -a "${OUTER_LOG}") 2>&1

cd "${ROOT}"
echo "[outer-start] run_prefix=${RUN_PREFIX} time=$(date -Is) log=${OUTER_LOG}"

run_one_score_key() {
  local score_key="$1"
  local run_name="${RUN_PREFIX}_${score_key}"
  echo "[score-key-start] ${score_key} run=${run_name} time=$(date -Is)"
  RUN_NAME="${run_name}" \
  BASE_MODEL_PATH="${MODEL_PATH}" \
  TRAINED_MODEL_PATH="${MCTS_MODEL_PATH}" \
  TRAIN_DATA="${TRAIN_DATA}" \
  PER_GRADE="${PER_GRADE}" \
  DROP_AXIOM_GRADE_ZERO=1 \
  FINAL_ONLY_JSON=0 \
  SCORE_KEY="${score_key}" \
  EVAL_BASE_DIRECT=0 \
  EVAL_TRAINED_DIRECT=0 \
  EVAL_TRAINED_VALUE=1 \
  TRAINED_VALUE_MAX_STEPS="${STEPWISE_MAX_STEPS}" \
  TRAINED_VALUE_NUM_CANDIDATES="${STEPWISE_NUM_CANDIDATES}" \
  MAX_RETHINKS="${MAX_RETHINKS}" \
  RETHINK_THRESHOLD="${RETHINK_THRESHOLD}" \
  RETHINK_SPREAD_THRESHOLD="${RETHINK_SPREAD_THRESHOLD}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  FINAL_MAX_NEW_TOKENS="${FINAL_MAX_NEW_TOKENS}" \
  NTFY_URL="${NTFY_URL}" \
  bash data_collection/scripts/run_axiom_clean_eval.sh
  echo "[score-key-finished] ${score_key} run=${run_name} time=$(date -Is)"
}

run_one_score_key response_mean_value
run_one_score_key last_value

python - <<PY
import json
from pathlib import Path

root = Path("${ROOT}")
base_run = "${BASE_RUN}"
run_prefix = "${RUN_PREFIX}"
score_keys = ["response_mean_value", "last_value"]

rows = []
base_summary_path = root / "data_collection/review_mcts_runs" / base_run / "summary.json"
if base_summary_path.exists():
    base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))
    direct = ((base_summary.get("evaluations") or {}).get("direct") or {}).get("trained_direct_clean")
    if direct:
        rows.append({
            "method": "direct_bootstrap_final_only",
            "score_key": None,
            "grade_mae": direct.get("grade_mae"),
            "boundary_acc": direct.get("boundary_acc"),
            "valid_rate": direct.get("valid_rate"),
            "median_err": direct.get("grade_median_abs_error"),
            "unsupported_evidence_rate": direct.get("unsupported_evidence_rate"),
        })

for score_key in score_keys:
    comparison_path = root / "model_training/src/output" / f"review-eval-{run_prefix}_{score_key}" / "comparison.json"
    if not comparison_path.exists():
        rows.append({"method": "mcts_stepwise_value_rerank", "score_key": score_key, "missing": str(comparison_path)})
        continue
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    row = comparison.get("trained_value_rerank_clean") or {}
    rows.append({
        "method": "mcts_stepwise_value_rerank",
        "score_key": score_key,
        "grade_mae": row.get("grade_mae"),
        "boundary_acc": row.get("boundary_acc"),
        "valid_rate": row.get("valid_rate"),
        "median_err": row.get("grade_median_abs_error"),
        "unsupported_evidence_rate": row.get("unsupported_evidence_rate"),
        "valid_count": row.get("valid_count"),
        "dimension_count": row.get("dimension_count"),
    })

out = root / "data_collection/review_mcts_runs" / f"{run_prefix}_compact_results.json"
out.write_text(json.dumps({"run_prefix": run_prefix, "base_run": base_run, "rows": rows}, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps({"compact_results": str(out), "rows": rows}, ensure_ascii=False, indent=2))
PY

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  curl --noproxy '*' --connect-timeout 5 --max-time 20 -fsS \
  -d "stepwise MCTS score-key eval completed: ${RUN_PREFIX}" \
  "${NTFY_URL}" >/dev/null 2>&1 || true

echo "[outer-finished] run_prefix=${RUN_PREFIX} time=$(date -Is)"
