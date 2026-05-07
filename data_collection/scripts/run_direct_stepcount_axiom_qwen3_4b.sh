#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

export RUN_NAME="${RUN_NAME:-direct_stepcount_axiom_qwen3_4b_nothink_20260506}"
export CFG="${CFG:-${ROOT}/data_collection/configs/mcts_code_review_qwen3_4b_no_think.yaml}"
export MODEL_KEY="${MODEL_KEY:-Qwen/Qwen3-4B}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
export STEP_COUNTS="${STEP_COUNTS:-1,2,3}"
export DIRECT_REPEATS="${DIRECT_REPEATS:-4}"
export MAX_STEPS="${MAX_STEPS:-240}"
export MAX_TRAINING_SEQ_LENGTH="${MAX_TRAINING_SEQ_LENGTH:-3072}"
export EVAL_PER_GRADE="${EVAL_PER_GRADE:-10}"
export POLICY_MIN_Q="${POLICY_MIN_Q:-0.8}"

AXIOM_PER_GRADE="${AXIOM_PER_GRADE:-30}"
AXIOM_MIN_GRADE="${AXIOM_MIN_GRADE:-1}"
AXIOM_MAX_GRADE="${AXIOM_MAX_GRADE:-5}"
AXIOM_SEED="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}/seed_codecritic_axiom.jsonl"
AXIOM_META="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}/seed_codecritic_axiom.metadata.json"

mkdir -p "$(dirname "${AXIOM_SEED}")"

if [[ ! -f "${AXIOM_SEED}" || ! -f "${AXIOM_META}" ]]; then
  cd "${ROOT}"
  PYTHONPATH="${ROOT}/data_collection" UV_CACHE_DIR=/tmp/uv-cache \
    uv run python data_collection/prepare_axiom_seedset.py \
      --output "${AXIOM_SEED}" \
      --metadata "${AXIOM_META}" \
      --per_grade "${AXIOM_PER_GRADE}" \
      --min_grade "${AXIOM_MIN_GRADE}" \
      --max_grade "${AXIOM_MAX_GRADE}"
fi

exec "${SCRIPT_DIR}/run_direct_stepcount_vs_review_qwen3_4b.sh" "$@"
