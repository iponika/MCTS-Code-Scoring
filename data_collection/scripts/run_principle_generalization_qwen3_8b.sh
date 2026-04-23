#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data1/xianzhiwei/mcts-code-review}"
MODEL_KEY="${MODEL_KEY:-Qwen/Qwen3-8B}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
RUN_NAME="${RUN_NAME:-principle_generalization_qwen3_8b_full_context_no_axiom0_20260423}"
MAX_TRAINING_SEQ_LENGTH="${MAX_TRAINING_SEQ_LENGTH:-4096}"
MAX_STEPS="${MAX_STEPS:-600}"
EXACT_PER_GRADE="${EXACT_PER_GRADE:-220}"
WEAK_INTERVAL_ITEMS="${WEAK_INTERVAL_ITEMS:-120}"
CODEJUDGE_PAIRS="${CODEJUDGE_PAIRS:-60}"
DROP_AXIOM_GRADE_ZERO="${DROP_AXIOM_GRADE_ZERO:-1}"
AXIOM_EXACT_FRACTION="${AXIOM_EXACT_FRACTION:-0.7}"

export ROOT
export MODEL_KEY
export MODEL_PATH
export RUN_NAME
export MAX_TRAINING_SEQ_LENGTH
export MAX_STEPS
export EXACT_PER_GRADE
export WEAK_INTERVAL_ITEMS
export CODEJUDGE_PAIRS
export DROP_AXIOM_GRADE_ZERO
export AXIOM_EXACT_FRACTION

exec "${ROOT}/data_collection/scripts/run_principle_generalization_eval.sh"
