from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mcts_math.axiom_scoring import axiom_grade_from_codecritic, axiom_scalar_score
from mcts_math.review_utils import (
    DEFAULT_DIMENSION_RUBRIC,
    build_review_question,
    format_public_tests,
    prepare_codecriticbench_sample,
)


CORRECTNESS_DIMENSION = "Correctness Verification"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified AXIOM-style review-scoring dataset.")
    parser.add_argument("--codecritic", type=Path, default=Path("datasets/CodeCriticBench/data/CodeCriticBench.jsonl"))
    parser.add_argument("--axiom", type=Path, nargs="*", default=sorted(Path("datasets/axiom-llm-judge/axiombench").glob("*.jsonl")))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260418)
    parser.add_argument("--codecritic_per_grade", type=int, default=13)
    parser.add_argument("--axiom_per_grade", type=int, default=50)
    parser.add_argument("--max_code_chars", type=int, default=8000)
    parser.add_argument("--strict", action="store_true", help="Fail if any source has fewer candidates than requested.")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if line:
                yield index, json.loads(line)


def codecritic_candidates(path: Path, max_code_chars: int) -> dict[int, list[tuple[int, dict[str, Any]]]]:
    by_grade: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, raw in iter_jsonl(path):
        grade = axiom_grade_from_codecritic(raw.get("correctness"), raw.get("score"))
        if grade is None:
            continue
        answer = str(raw.get("answer") or "")
        tests = (raw.get("public_test", {}).get("input") or []) + (raw.get("private_test", {}).get("input") or [])
        if not answer.strip() or not tests or len(answer) > max_code_chars:
            continue
        by_grade[grade].append((index, raw))
    return by_grade


def axiom_candidates(paths: list[Path], max_code_chars: int) -> dict[int, list[tuple[str, int, dict[str, Any]]]]:
    by_grade: dict[int, list[tuple[str, int, dict[str, Any]]]] = defaultdict(list)
    for path in paths:
        for index, raw in iter_jsonl(path):
            try:
                grade = int(raw["score"])
            except (KeyError, TypeError, ValueError):
                continue
            code = str(raw.get("code") or "")
            inst = str(raw.get("inst") or "")
            if grade < 0 or grade > 5 or not code.strip() or not inst.strip() or len(code) > max_code_chars:
                continue
            by_grade[grade].append((path.name, index, raw))
    return by_grade


def sample_by_grade(
    by_grade: dict[int, list[Any]],
    per_grade: int,
    rng: random.Random,
    strict: bool,
    source_name: str,
) -> list[Any]:
    selected = []
    for grade in range(6):
        pool = list(by_grade.get(grade, []))
        if len(pool) < per_grade and strict:
            raise SystemExit(f"{source_name} grade {grade} has {len(pool)} candidates, requested {per_grade}")
        rng.shuffle(pool)
        selected.extend(pool[: min(per_grade, len(pool))])
    return selected


def prepared_axiom_sample(dataset_name: str, original_index: int, raw: dict[str, Any]) -> dict[str, Any]:
    grade = int(raw["score"])
    scalar = axiom_scalar_score(grade)
    pass_proxy = 1.0 if grade >= 3 else 0.0
    sample = {
        "prepared_review_sample": True,
        "dataset_family": "axiom",
        "source_dataset": dataset_name,
        "original_dataset_index": original_index,
        "problem": raw["inst"],
        "candidate_code": raw["code"],
        "code_language": raw.get("lang") or "python",
        "tests": [],
        "tests_for_prompt": "No executable tests are available. Use the AXIOM refinement-effort label as the objective scoring anchor.",
        "difficulty": None,
        "source": "axiom",
        "subset": dataset_name.replace(".jsonl", ""),
        "reference_scores": {CORRECTNESS_DIMENSION: scalar / 10.0},
        "dimension_rubrics": {CORRECTNESS_DIMENSION: DEFAULT_DIMENSION_RUBRIC[CORRECTNESS_DIMENSION]},
        "dimension_target_scores": {CORRECTNESS_DIMENSION: scalar / 10.0},
        "axiom_target_grade": grade,
        "axiom_target_score": scalar,
        "objective": {
            "public_test_pass_rate": pass_proxy,
            "private_test_pass_rate": pass_proxy,
            "full_test_pass_rate": pass_proxy,
            "source": "axiom_grade_boundary_proxy",
        },
        "overall_score": raw["score"],
        "correctness_label": "Correct" if grade >= 3 else "Error",
    }
    sample["question"] = build_review_question(sample)
    return sample


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    codecritic_selected = sample_by_grade(
        codecritic_candidates(args.codecritic, args.max_code_chars),
        args.codecritic_per_grade,
        rng,
        args.strict,
        "CodeCriticBench",
    )
    axiom_selected = sample_by_grade(
        axiom_candidates(args.axiom, args.max_code_chars),
        args.axiom_per_grade,
        rng,
        args.strict,
        "AXIOMBench",
    )

    samples: list[dict[str, Any]] = []
    for original_index, raw in codecritic_selected:
        sample = prepare_codecriticbench_sample(raw, dataset_index=None)
        sample["prepared_review_sample"] = True
        sample["dataset_family"] = "codecritic"
        sample["source_dataset"] = str(args.codecritic)
        sample["original_dataset_index"] = original_index
        samples.append(sample)
    for dataset_name, original_index, raw in axiom_selected:
        samples.append(prepared_axiom_sample(dataset_name, original_index, raw))

    rng.shuffle(samples)
    for dataset_index, sample in enumerate(samples):
        sample["dataset_index"] = dataset_index

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as writer:
        for sample in samples:
            writer.write(json.dumps(sample, ensure_ascii=False) + "\n")

    metadata = {
        "output": str(args.output),
        "seed": args.seed,
        "total": len(samples),
        "codecritic_per_grade_requested": args.codecritic_per_grade,
        "axiom_per_grade_requested": args.axiom_per_grade,
        "source_dist": dict(sorted(Counter(sample["dataset_family"] for sample in samples).items())),
        "target_grade_dist": dict(sorted(Counter(sample["axiom_target_grade"] for sample in samples).items())),
        "subset_dist": dict(sorted(Counter(sample["subset"] for sample in samples).items())),
    }
    metadata_path = args.metadata or args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
