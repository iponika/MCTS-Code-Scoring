#!/usr/bin/env bash
set -euo pipefail

cd /data1/xianzhiwei/mcts-code-review

RUN_NAME_VALUE="${RUN_NAME:-bootstrap_cmp_stageq_branch2_overnight_20260426}"
OUTER_LOG="${OUTER_LOG:-/tmp/overnight_stageq_branch2_cmp_20260426.log}"

exec > >(tee -a "${OUTER_LOG}") 2>&1

echo "[outer-start] run=${RUN_NAME_VALUE} time=$(date -Is) log=${OUTER_LOG}"

export ROOT="${ROOT:-/data1/xianzhiwei/mcts-code-review}"
export RUN_NAME="${RUN_NAME_VALUE}"
export NTFY_URL="${NTFY_URL:-https://ntfy.sh/iponika_mcts}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

export SEED_PER_GRADE="${SEED_PER_GRADE:-16}"
export SEED_MIN_GRADE="${SEED_MIN_GRADE:-1}"
export SEED_MAX_GRADE="${SEED_MAX_GRADE:-5}"
export SEED_MAX_OBJECTIVE_ASSERTIONS_PER_SPLIT="${SEED_MAX_OBJECTIVE_ASSERTIONS_PER_SPLIT:-8}"
export SEED_ASSERTION_TIMEOUT_SECONDS="${SEED_ASSERTION_TIMEOUT_SECONDS:-1.5}"

export DIRECT_REPEATS="${DIRECT_REPEATS:-4}"
export MAX_STEPS="${MAX_STEPS:-240}"
export EVAL_PER_GRADE="${EVAL_PER_GRADE:-12}"
export MAX_TRAINING_SEQ_LENGTH="${MAX_TRAINING_SEQ_LENGTH:-3072}"
export POLICY_MIN_Q="${POLICY_MIN_Q:-0.5}"
export MAX_VALUE_PATHS_PER_DIMENSION="${MAX_VALUE_PATHS_PER_DIMENSION:-0}"
export DIRECT_KEEP_ALL_VALUE_PATHS="${DIRECT_KEEP_ALL_VALUE_PATHS:-1}"
export MCTS_KEEP_ALL_VALUE_PATHS="${MCTS_KEEP_ALL_VALUE_PATHS:-1}"
export DIRECT_POLICY_RESPONSE_MODE="${DIRECT_POLICY_RESPONSE_MODE:-path}"
export MCTS_POLICY_RESPONSE_MODE="${MCTS_POLICY_RESPONSE_MODE:-path}"
export BOOTSTRAP_STRATIFY_BY_DATASET="${BOOTSTRAP_STRATIFY_BY_DATASET:-1}"
export BOOTSTRAP_STRATIFY_BY_DELTA_BUCKET="${BOOTSTRAP_STRATIFY_BY_DELTA_BUCKET:-1}"
export EVAL_FINAL_ONLY_JSON="${EVAL_FINAL_ONLY_JSON:-0}"
export STEPWISE_MAX_STEPS="${STEPWISE_MAX_STEPS:-3}"
export STEPWISE_NUM_CANDIDATES="${STEPWISE_NUM_CANDIDATES:-3}"
export STEPWISE_MAX_RETHINKS="${STEPWISE_MAX_RETHINKS:-1}"

bash data_collection/scripts/run_bootstrap_comparison_qwen3_4b.sh

python - <<PY
import json
from pathlib import Path

root = Path("/data1/xianzhiwei/mcts-code-review")
run = "${RUN_NAME_VALUE}"
summary = root / "data_collection/review_mcts_runs" / run / "summary.json"
if summary.exists():
    payload = json.loads(summary.read_text(encoding="utf-8"))
    rows = []
    for method, evals in (payload.get("evaluations") or {}).items():
        for tag in ["trained_direct_clean", "trained_value_rerank_clean", "base_direct_clean"]:
            row = (evals or {}).get(tag) or {}
            if row:
                rows.append(
                    {
                        "method": method,
                        "eval_tag": tag,
                        "grade_mae": row.get("grade_mae"),
                        "boundary_acc": row.get("boundary_acc"),
                        "valid_rate": row.get("valid_rate"),
                        "median_err": row.get("grade_median_abs_error"),
                        "unsupported_evidence_rate": row.get("unsupported_evidence_rate"),
                    }
                )
    out = root / "data_collection/review_mcts_runs" / f"{run}_compact_results.json"
    out.write_text(json.dumps({"run": run, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"compact_results": str(out), "rows": rows}, ensure_ascii=False, indent=2))
else:
    print(json.dumps({"missing_summary": str(summary)}, ensure_ascii=False))
PY

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  curl --noproxy '*' --connect-timeout 5 --max-time 20 -fsS \
  -d "overnight stageq branch2 bootstrap comparison completed: ${RUN_NAME_VALUE}" \
  "${NTFY_URL}" >/dev/null 2>&1 || true

echo "[outer-finished] run=${RUN_NAME_VALUE} time=$(date -Is)"
