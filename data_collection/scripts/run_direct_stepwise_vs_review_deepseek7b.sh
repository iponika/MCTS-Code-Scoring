#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

source "${SCRIPT_DIR}/deepseek7b_defaults.sh"

export RUN_NAME="${RUN_NAME:-direct_stepwise_vs_review_deepseek7b}"
export CFG="${CFG:-${ROOT}/data_collection/configs/mcts_code_review_deepseek_r1_distill_qwen_7b.yaml}"

exec "${SCRIPT_DIR}/run_direct_stepwise_vs_review_qwen3_4b.sh" "$@"
