from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mcts_math.axiom_scoring import axiom_grade_from_codecritic
from mcts_math.review_utils import prepare_codecriticbench_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced CodeCritic-only AXIOM-aligned seed set.")
    parser.add_argument("--codecritic", type=Path, default=Path("datasets/CodeCriticBench/data/CodeCriticBench.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--per_grade", type=int, default=60)
    parser.add_argument("--min_grade", type=int, default=1)
    parser.add_argument("--max_grade", type=int, default=5)
    parser.add_argument("--max_code_chars", type=int, default=12000)
    parser.add_argument("--oversample_factor", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if line:
                yield index, json.loads(line)


def usable_candidate(raw: dict[str, Any], max_code_chars: int) -> bool:
    answer = str(raw.get("answer") or "")
    public_tests = list(raw.get("public_test", {}).get("input") or [])
    private_tests = list(raw.get("private_test", {}).get("input") or [])
    all_tests = public_tests + private_tests
    if not answer.strip():
        return False
    if len(answer) > max_code_chars:
        return False
    if not all_tests:
        return False
    if not raw.get("checklist_dimensions"):
        return False
    return True


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    coarse_by_grade: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    by_grade: dict[int, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()

    for original_index, raw in iter_jsonl(args.codecritic):
        rough_grade = axiom_grade_from_codecritic(raw.get("correctness"), raw.get("score"))
        if rough_grade is None:
            skipped["missing_grade"] += 1
            continue
        if not usable_candidate(raw, args.max_code_chars):
            skipped["unusable"] += 1
            continue
        if rough_grade < args.min_grade or rough_grade > args.max_grade:
            skipped[f"outside_rough_grade_range_{rough_grade}"] += 1
            continue
        coarse_by_grade[rough_grade].append((original_index, raw))

    shortlisted: list[tuple[int, dict[str, Any]]] = []
    shortlist_counts = {}
    for grade in range(args.min_grade, args.max_grade + 1):
        pool = list(coarse_by_grade.get(grade, []))
        rng.shuffle(pool)
        take = min(len(pool), max(args.per_grade, args.per_grade * max(1, args.oversample_factor)))
        shortlist_counts[str(grade)] = take
        shortlisted.extend(pool[:take])

    for original_index, raw in shortlisted:
        sample = prepare_codecriticbench_sample(raw, dataset_index=None)
        final_grade = int(sample["axiom_target_grade"])
        if final_grade < args.min_grade or final_grade > args.max_grade:
            skipped[f"outside_final_grade_range_{final_grade}"] += 1
            continue
        sample["prepared_review_sample"] = True
        sample["dataset_family"] = "codecritic_seed"
        sample["source_dataset"] = str(args.codecritic)
        sample["original_dataset_index"] = original_index
        sample["seed_split"] = "bootstrap_train"
        by_grade[final_grade].append(sample)

    selected: list[dict[str, Any]] = []
    available_by_grade = {str(grade): len(by_grade.get(grade, [])) for grade in range(args.min_grade, args.max_grade + 1)}
    for grade in range(args.min_grade, args.max_grade + 1):
        pool = [dict(sample) for sample in by_grade.get(grade, [])]
        if len(pool) < args.per_grade and args.strict:
            raise SystemExit(f"grade {grade} has {len(pool)} candidates, requested {args.per_grade}")
        rng.shuffle(pool)
        for sample in pool[: min(args.per_grade, len(pool))]:
            selected.append(sample)

    rng.shuffle(selected)
    for dataset_index, sample in enumerate(selected):
        sample["dataset_index"] = dataset_index

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as writer:
        for sample in selected:
            writer.write(json.dumps(sample, ensure_ascii=False) + "\n")

    metadata = {
        "output": str(args.output),
        "seed": args.seed,
        "per_grade_requested": args.per_grade,
        "min_grade": args.min_grade,
        "max_grade": args.max_grade,
        "oversample_factor": args.oversample_factor,
        "total": len(selected),
        "available_rough_grade_counts": {str(grade): len(coarse_by_grade.get(grade, [])) for grade in range(args.min_grade, args.max_grade + 1)},
        "shortlist_counts": shortlist_counts,
        "available_final_grade_counts": available_by_grade,
        "selected_grade_counts": dict(sorted(Counter(sample["axiom_target_grade"] for sample in selected).items())),
        "skipped": dict(sorted(skipped.items())),
    }
    metadata_path = args.metadata or args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
