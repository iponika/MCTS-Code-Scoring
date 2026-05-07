from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mcts_math.axiom_scoring import axiom_scalar_score
from mcts_math.review_utils import DEFAULT_DIMENSION_RUBRIC, build_review_question


CORRECTNESS_DIMENSION = "Correctness Verification"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced AXIOM-only prepared review seed dataset.")
    parser.add_argument("--axiom_dir", type=Path, default=Path("datasets/axiom-llm-judge/axiombench"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--per_grade", type=int, default=30)
    parser.add_argument("--min_grade", type=int, default=1)
    parser.add_argument("--max_grade", type=int, default=5)
    parser.add_argument("--max_code_chars", type=int, default=12000)
    parser.add_argument("--strict", action="store_true", help="Fail if any requested grade has fewer candidates than requested.")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if line:
                yield index, json.loads(line)


def prepared_axiom_sample(dataset_name: str, original_index: int, raw: dict[str, Any]) -> dict[str, Any]:
    grade = int(raw["score"])
    scalar = axiom_scalar_score(grade)
    pass_proxy = 1.0 if grade >= 3 else 0.0
    sample = {
        "prepared_review_sample": True,
        "dataset_family": "axiom",
        "source_dataset": dataset_name,
        "original_dataset_index": original_index,
        "dataset_index": f"{dataset_name.replace('.jsonl', '')}:{original_index}",
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

    by_grade: dict[int, list[tuple[str, int, dict[str, Any]]]] = defaultdict(list)
    for file_path in sorted(args.axiom_dir.glob("*.jsonl")):
        for index, raw in iter_jsonl(file_path):
            try:
                grade = int(raw["score"])
            except (KeyError, TypeError, ValueError):
                continue
            code = str(raw.get("code") or "")
            inst = str(raw.get("inst") or "")
            if (
                grade < args.min_grade
                or grade > args.max_grade
                or not code.strip()
                or not inst.strip()
                or (args.max_code_chars > 0 and len(code) > args.max_code_chars)
            ):
                continue
            by_grade[grade].append((file_path.name, index, raw))

    selected: list[dict[str, Any]] = []
    available_grade_counts: dict[str, int] = {}
    for grade in range(args.min_grade, args.max_grade + 1):
        pool = list(by_grade.get(grade, []))
        available_grade_counts[str(grade)] = len(pool)
        if len(pool) < args.per_grade and args.strict:
            raise SystemExit(f"AXIOM grade {grade} has {len(pool)} candidates, requested {args.per_grade}")
        rng.shuffle(pool)
        for dataset_name, original_index, raw in pool[: min(args.per_grade, len(pool))]:
            selected.append(prepared_axiom_sample(dataset_name, original_index, raw))

    rng.shuffle(selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as writer:
        for sample in selected:
            writer.write(json.dumps(sample, ensure_ascii=False) + "\n")

    metadata = {
        "output": str(args.output),
        "seed": args.seed,
        "total": len(selected),
        "per_grade_requested": args.per_grade,
        "grade_range": [args.min_grade, args.max_grade],
        "available_grade_counts": available_grade_counts,
        "selected_grade_counts": dict(sorted(Counter(str(sample["axiom_target_grade"]) for sample in selected).items())),
        "subset_dist": dict(sorted(Counter(sample["subset"] for sample in selected).items())),
    }
    metadata_path = args.metadata or args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
