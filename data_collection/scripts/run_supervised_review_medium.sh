#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${RUN_DIR:-data_collection/review_mcts_runs/supervised_medium_$(date +%Y%m%d_%H%M%S)}"
DATASET="$RUN_DIR/review_scoring_medium.jsonl"
DIRECT_OUT="$RUN_DIR/direct_local.jsonl"
MCTS_OUT="$RUN_DIR/supervised_mcts_local.jsonl"
SUMMARY="$RUN_DIR/summary.json"
LOG="$RUN_DIR/run.log"
NOTIFY_LOG="$RUN_DIR/notify.log"
CFG="${CFG:-data_collection/configs/mcts_code_review_supervised_medium.yaml}"
NTFY_TOPIC="${NTFY_TOPIC:-https://ntfy.sh/iponika_mcts}"

line_count() {
  if [[ -f "$1" ]]; then
    wc -l < "$1" | tr -d ' '
  else
    echo 0
  fi
}

notify() {
  local message="$1"
  local attempt
  mkdir -p "$RUN_DIR"
  for attempt in 1 2 3; do
    if curl -fsS -d "$message" "$NTFY_TOPIC" >>"$NOTIFY_LOG" 2>&1; then
      echo "[$(date '+%F %T')] notify ok attempt=$attempt message=$message" >>"$NOTIFY_LOG"
      return 0
    fi
    echo "[$(date '+%F %T')] notify failed attempt=$attempt message=$message" >>"$NOTIFY_LOG"
    sleep $((attempt * 5))
  done
  return 0
}

mkdir -p "$RUN_DIR/samples" "$RUN_DIR/logs"
trap 'notify "supervised review medium failed: '"$RUN_DIR"', see '"$LOG"'"' ERR

{
  echo "[$(date '+%F %T')] supervised review medium run started"
  echo "run_dir=$RUN_DIR"
  echo "config=$CFG"

  if [[ ! -f "$DATASET" ]]; then
    echo "[$(date '+%F %T')] preparing unified AXIOM+CodeCritic dataset"
    PYTHONPATH=data_collection UV_CACHE_DIR=/tmp/uv-cache \
      uv run python data_collection/prepare_review_scoring_dataset.py \
        --output "$DATASET" \
        --metadata "$RUN_DIR/dataset_metadata.json"
  else
    echo "[$(date '+%F %T')] dataset already exists; skipping preparation"
  fi
  LIMIT="$(line_count "$DATASET")"
  echo "limit=$LIMIT"

  if [[ "$(line_count "$DIRECT_OUT")" -ge "$LIMIT" ]]; then
    echo "[$(date '+%F %T')] direct baseline already complete; skipping"
  else
    echo "[$(date '+%F %T')] running direct baseline"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" PYTHONPATH=data_collection UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 \
      uv run python data_collection/direct_review_local.py \
        --custom_cfg "$CFG" \
        --dataset "$DATASET" \
        --start 0 \
        --limit "$LIMIT" \
        --output "$DIRECT_OUT" \
        --dimension "Correctness Verification" \
        --batch_size 6 \
        --repeats 1
  fi

  if [[ "$(line_count "$MCTS_OUT")" -ge "$LIMIT" ]]; then
    echo "[$(date '+%F %T')] supervised MCTS already complete; skipping"
  else
    echo "[$(date '+%F %T')] running supervised MCTS"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" PYTHONPATH=data_collection UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 \
      uv run python data_collection/solver_review.py \
        --custom_cfg "$CFG" \
        --dataset "$DATASET" \
        --start 0 \
        --limit "$LIMIT" \
        --output "$MCTS_OUT" \
        --output_dir "$RUN_DIR/samples"
  fi

  echo "[$(date '+%F %T')] summarizing"
  PYTHONPATH=data_collection UV_CACHE_DIR=/tmp/uv-cache \
    uv run python data_collection/summarize_review_scores.py \
      --direct "$DIRECT_OUT" \
      --mcts "$MCTS_OUT" \
      --output "$SUMMARY"

  echo "[$(date '+%F %T')] supervised review medium run finished"
  cat "$SUMMARY"
} > "$LOG" 2>&1

notify "supervised review medium completed: $SUMMARY"
