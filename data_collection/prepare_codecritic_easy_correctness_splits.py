from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from mcts_math.review_utils import prepare_codecriticbench_sample


CODEGEN_SOURCES = {"mbpp", "codeforce", "live-code-bench", "debug"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CodeCriticBench CodeGen-Easy correctness train/eval splits.")
    parser.add_argument("--input", type=Path, default=Path("datasets/CodeCriticBench/data/CodeCriticBench.jsonl"))
    parser.add_argument("--train_output", type=Path, required=True)
    parser.add_argument("--eval_output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--target_train", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260513)
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    all_easy_counts = Counter()
    eligible_by_layer: dict[str, list[dict[str, Any]]] = {"low": [], "mid": [], "high": []}
    skipped = Counter()

    for original_index, raw in iter_jsonl(args.input):
        if raw.get("source") not in CODEGEN_SOURCES or raw.get("difficulty") != "Easy":
            continue
        score = int(raw.get("score"))
        layer = layer_for_score(score)
        all_easy_counts[layer] += 1
        if correctness_index(raw) is None:
            skipped["missing_correctness_verification"] += 1
            continue
        sample = prepare_codecriticbench_sample(raw, dataset_index=None)
        sample["prepared_review_sample"] = True
        sample["dataset_family"] = "codecritic_codegen_easy"
        sample["source_dataset"] = str(args.input)
        sample["original_dataset_index"] = original_index
        sample["score_layer"] = layer
        sample["seed_selection_score"] = score
        eligible_by_layer[layer].append(sample)

    total_easy = sum(all_easy_counts.values())
    selected: list[dict[str, Any]] = []
    selection_counts: dict[str, int] = {}
    for layer in ("low", "mid", "high"):
        pool = list(eligible_by_layer[layer])
        rng.shuffle(pool)
        if layer == "mid":
            take = len(pool)
        else:
            take = math.floor(args.target_train * all_easy_counts[layer] / total_easy)
            take = min(take, len(pool))
        selected.extend(pool[:take])
        eligible_by_layer[layer] = pool[take:]
        selection_counts[layer] = take

    eval_rows = [sample for layer in ("low", "mid", "high") for sample in eligible_by_layer[layer]]
    rng.shuffle(selected)
    rng.shuffle(eval_rows)
    for dataset_index, sample in enumerate(selected):
        sample["dataset_index"] = dataset_index
        sample["seed_split"] = "mcts_train"
    for dataset_index, sample in enumerate(eval_rows):
        sample["dataset_index"] = dataset_index
        sample["seed_split"] = "eval"

    write_jsonl(args.train_output, selected)
    write_jsonl(args.eval_output, eval_rows)

    metadata = {
        "input": str(args.input),
        "train_output": str(args.train_output),
        "eval_output": str(args.eval_output),
        "seed": args.seed,
        "target_train": args.target_train,
        "codegen_sources": sorted(CODEGEN_SOURCES),
        "difficulty": "Easy",
        "layer_definition": {"low": "score 0-3", "mid": "score 4-6", "high": "score 7-10"},
        "all_easy_layer_counts": dict(all_easy_counts),
        "eligible_layer_counts_before_selection": {key: len(value) + selection_counts.get(key, 0) for key, value in eligible_by_layer.items()},
        "selected_layer_counts": selection_counts,
        "train_total": len(selected),
        "eval_total": len(eval_rows),
        "skipped": dict(skipped),
        "selection_note": "Low/high train counts use floor(target_train * original_layer_count / total_codegen_easy); mid uses all eligible samples.",
    }
    metadata_path = args.metadata or args.train_output.with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
