from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize MCTS record dataset_index values to an all-seed file.")
    parser.add_argument("--all_seed", type=Path, required=True)
    parser.add_argument("--record_seed", type=Path, required=True)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=None)
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def seed_key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (row.get("source"), row.get("subset"), row.get("original_dataset_index"))


def copy_seed_fields(record: dict[str, Any], seed: dict[str, Any]) -> None:
    for key in (
        "dataset_index",
        "original_dataset_index",
        "source",
        "subset",
        "target_correctness_score",
        "reference_scores",
        "dimension_rubrics",
        "overall_score",
        "correctness_label",
        "score_layer",
        "seed_selection_score",
        "dataset_family",
        "source_dataset",
    ):
        if key in seed:
            record[key] = seed[key]


def main() -> None:
    args = parse_args()
    all_seed_by_key = {seed_key(row): row for row in iter_jsonl(args.all_seed)}
    record_seed_by_index = {index: row for index, row in enumerate(iter_jsonl(args.record_seed))}

    stats = {
        "all_seed": str(args.all_seed),
        "record_seed": str(args.record_seed),
        "output": str(args.output),
        "input_records": 0,
        "written_records": 0,
        "dropped_without_record_seed": 0,
        "dropped_not_in_all_seed": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as writer:
        for path in expand_inputs(args.input):
            for record in iter_jsonl(path):
                stats["input_records"] += 1
                record_seed = record_seed_by_index.get(record.get("dataset_index"))
                if record_seed is None:
                    stats["dropped_without_record_seed"] += 1
                    continue
                all_seed = all_seed_by_key.get(seed_key(record_seed))
                if all_seed is None:
                    stats["dropped_not_in_all_seed"] += 1
                    continue
                copy_seed_fields(record, all_seed)
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["written_records"] += 1

    stats_path = args.stats or args.output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
