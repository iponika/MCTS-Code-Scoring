#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_NAME="${RUN_NAME:-qwen35_codecritic_easy_correctness_optimistic_20260514}"
RUN_DIR="${RUN_DIR:-${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}}"
SEED_DATA="${SEED_DATA:-${RUN_DIR}/seed_all_nonqa_easy.jsonl}"
SHARD_GLOB="${SHARD_GLOB:-${RUN_DIR}/mcts_shard_*.jsonl}"
LOG_DIR="${RUN_DIR}/logs"
LOG_FILE="${LOG_DIR}/finalize_optimistic_mcts.log"

OPT_RECORDS="${OPT_RECORDS:-${RUN_DIR}/mcts_optimistic_over_gt_under_records.jsonl}"
OPT_SEEDS="${OPT_SEEDS:-${RUN_DIR}/seed_optimistic_over_gt_under.jsonl}"
OPT_STATS="${OPT_STATS:-${RUN_DIR}/optimistic_selection_stats.json}"
TRAIN_JSONL="${TRAIN_JSONL:-${RUN_DIR}/mcts_optimistic_policy_pm2_value_pm2_aligned_train.jsonl}"
TRAIN_STATS="${TRAIN_STATS:-${RUN_DIR}/mcts_optimistic_policy_pm2_value_pm2_aligned_train_stats.json}"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

{
  echo "[finalize] host=$(hostname) time=$(date -Is)"
  echo "[finalize] run_dir=${RUN_DIR}"
  echo "[finalize] seed_data=${SEED_DATA}"
  echo "[finalize] shard_glob=${SHARD_GLOB}"
} | tee -a "${LOG_FILE}"

RUN_DIR="${RUN_DIR}" \
SEED_DATA="${SEED_DATA}" \
SHARD_GLOB="${SHARD_GLOB}" \
OPT_RECORDS="${OPT_RECORDS}" \
OPT_SEEDS="${OPT_SEEDS}" \
OPT_STATS="${OPT_STATS}" \
python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import glob
import json
import os
from collections import Counter
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def terminal_score_deltas(record: dict) -> list[int]:
    deltas: list[int] = []
    for node in (record.get("react") or {}).values():
        if not isinstance(node, dict) or not node.get("final_answer"):
            continue
        raw_details = node.get("reward_details")
        try:
            details = json.loads(raw_details) if isinstance(raw_details, str) else (raw_details or {})
        except json.JSONDecodeError:
            continue
        if details.get("score_scale") != "codecritic_correctness_1_10":
            continue
        pred = details.get("predicted_correctness_score")
        target = details.get("target_correctness_score")
        if pred is None or target is None:
            target = record.get("target_correctness_score")
        if pred is None or target is None:
            continue
        try:
            deltas.append(int(round(float(pred) - float(target))))
        except (TypeError, ValueError):
            continue
    return deltas


seed_data = Path(os.environ["SEED_DATA"])
shards = [Path(path) for path in sorted(glob.glob(os.environ["SHARD_GLOB"]))]
if not shards:
    raise SystemExit(f"No MCTS shards matched: {os.environ['SHARD_GLOB']}")

records_out = Path(os.environ["OPT_RECORDS"])
seeds_out = Path(os.environ["OPT_SEEDS"])
stats_out = Path(os.environ["OPT_STATS"])
records_out.parent.mkdir(parents=True, exist_ok=True)

selected_records = []
selected_keys = set()
stats = Counter()
delta_hist = Counter()
for shard in shards:
    for record in iter_jsonl(shard):
        stats["records"] += 1
        deltas = terminal_score_deltas(record)
        over = sum(1 for delta in deltas if delta > 0)
        under = sum(1 for delta in deltas if delta < 0)
        exact = sum(1 for delta in deltas if delta == 0)
        for delta in deltas:
            delta_hist[delta] += 1
        if over > under:
            stats["optimistic_records"] += 1
            record["optimistic_selection"] = {
                "over_count": over,
                "under_count": under,
                "exact_count": exact,
                "terminal_review_count": len(deltas),
            }
            selected_records.append(record)
            selected_keys.add((record.get("source"), record.get("subset"), record.get("dataset_index")))
        elif under > over:
            stats["pessimistic_records"] += 1
        else:
            stats["tied_records"] += 1

with records_out.open("w", encoding="utf-8") as writer:
    for record in selected_records:
        writer.write(json.dumps(record, ensure_ascii=False) + "\n")

seed_rows = []
for row in iter_jsonl(seed_data):
    key = (row.get("source"), row.get("subset"), row.get("dataset_index"))
    if key in selected_keys:
        row["seed_split"] = "mcts_optimistic_over_gt_under"
        seed_rows.append(row)

with seeds_out.open("w", encoding="utf-8") as writer:
    for row in seed_rows:
        writer.write(json.dumps(row, ensure_ascii=False) + "\n")

payload = {
    "input_shards": [str(path) for path in shards],
    "records_output": str(records_out),
    "seeds_output": str(seeds_out),
    "records": stats["records"],
    "optimistic_records": stats["optimistic_records"],
    "pessimistic_records": stats["pessimistic_records"],
    "tied_records": stats["tied_records"],
    "seed_rows": len(seed_rows),
    "delta_hist": dict(sorted(delta_hist.items())),
}
stats_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

PYTHONPATH="${ROOT}/model_training/src:${ROOT}" \
uv run python -m magicoder.preprocess_review_mcts_data \
  --input "${OPT_RECORDS}" \
  --output_file "${TRAIN_JSONL}" \
  --policy_min_q -1.0 \
  --policy_max_grade_delta 2 \
  --max_value_paths_per_dimension 0 \
  --max_value_paths_per_sample_label 3 \
  --value_only_from_policy_records_only \
  --value_sampling_seed 20260513 \
  --no-shuffle \
  > "${TRAIN_STATS}"

{
  echo "[finalize] preprocess complete time=$(date -Is)"
  cat "${TRAIN_STATS}"
  wc -l "${TRAIN_JSONL}"
} | tee -a "${LOG_FILE}"
