from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mcts_math.review_utils import prepare_codecriticbench_sample


CODEGEN_SOURCES = {"mbpp", "codeforce", "live-code-bench", "debug"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build eligible CodeCriticBench CodeGen correctness seeds.")
    parser.add_argument("--input", type=Path, default=Path("datasets/CodeCriticBench/data/CodeCriticBench.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--difficulty",
        action="append",
        default=[],
        help="Difficulty value to include. Repeatable. Use Meidum for the dataset's misspelled medium label.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                yield index, json.loads(line)


def correctness_index(raw: dict[str, Any]) -> int | None:
    dimensions = raw.get("checklist_dimensions") or []
    scores = raw.get("checklist_scores") or []
    checklists = raw.get("checklists") or []
    if "Correctness Verification" not in dimensions:
        return None
    index = dimensions.index("Correctness Verification")
    if index >= len(scores) or scores[index] is None:
        return None
    if index >= len(checklists) or checklists[index] is None:
        return None
    return index


def layer_for_score(score: int) -> str:
    if score <= 3:
        return "low"
    if score <= 6:
        return "mid"
    return "high"


def should_keep_codecritic_correctness_sample(sample: dict[str, Any]) -> bool:
    label = str(sample.get("correctness_label") or "").strip().lower()
    try:
        target_score = float(sample.get("target_correctness_score"))
    except (TypeError, ValueError):
        return False
    if label == "error" and target_score > 5:
        return False
    if label == "correct" and target_score < 5:
        return False
    return True


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    difficulties = set(args.difficulty or ["Easy"])

    rows: list[dict[str, Any]] = []
    raw_counts = Counter()
    eligible_counts = Counter()
    skipped = Counter()

    for original_index, raw in iter_jsonl(args.input):
        source = raw.get("source")
        difficulty = raw.get("difficulty")
        if source not in CODEGEN_SOURCES or difficulty not in difficulties:
            continue
        score = int(raw.get("score"))
        layer = layer_for_score(score)
        raw_counts[(difficulty, layer)] += 1
        if correctness_index(raw) is None:
            skipped[(difficulty, "missing_correctness_verification")] += 1
            continue
        sample = prepare_codecriticbench_sample(raw, dataset_index=None)
        if not should_keep_codecritic_correctness_sample(sample):
            skipped[(difficulty, "label_score_conflict")] += 1
            continue
        sample["prepared_review_sample"] = True
        sample["dataset_family"] = "codecritic_codegen_correctness"
        sample["source_dataset"] = str(args.input)
        sample["original_dataset_index"] = original_index
        sample["score_layer"] = layer
        sample["seed_selection_score"] = score
        sample["seed_split"] = "mcts_all_eligible"
        rows.append(sample)
        eligible_counts[(difficulty, layer)] += 1

    rows.sort(key=lambda row: int(row["original_dataset_index"]))
    for dataset_index, sample in enumerate(rows):
        sample["dataset_index"] = dataset_index

    write_jsonl(args.output, rows)

    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "difficulty": sorted(difficulties),
        "codegen_sources": sorted(CODEGEN_SOURCES),
        "filter": {
            "requires_correctness_verification_score_and_checklist": True,
            "drop_error_with_target_score_gt_5": True,
            "drop_correct_with_target_score_lt_5": True,
        },
        "layer_definition": {"low": "score 0-3", "mid": "score 4-6", "high": "score 7-10"},
        "raw_counts_by_difficulty_layer": {f"{k[0]}:{k[1]}": v for k, v in sorted(raw_counts.items())},
        "eligible_counts_by_difficulty_layer": {f"{k[0]}:{k[1]}": v for k, v in sorted(eligible_counts.items())},
        "skipped": {f"{k[0]}:{k[1]}": v for k, v in sorted(skipped.items())},
        "total": len(rows),
    }
    metadata_path = args.metadata or args.output.with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
