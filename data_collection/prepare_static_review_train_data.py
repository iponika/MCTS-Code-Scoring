from __future__ import annotations

import argparse
import json
from pathlib import Path

from magicoder.axiom_scoring import AXIOM_GRADE_DESCRIPTIONS, axiom_scalar_score, axiom_value_target, axiom_verdict, clamp_axiom_grade
from magicoder.preprocess_review_mcts_data import build_instruction
from mcts_math.review_utils import load_codecriticbench_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert prepared review samples into exact-label train_multi review data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimension", default="Correctness Verification")
    parser.add_argument("--value_loss_weight", type=float, default=1.0)
    parser.add_argument("--lm_loss_weight", type=float, default=0.15)
    return parser.parse_args()


def repair_effort_for_grade(grade: int) -> str:
    if grade == 5:
        return "none"
    if grade == 4:
        return "minor_quality"
    if grade == 3:
        return "major_quality"
    if grade == 2:
        return "minor_functional"
    if grade == 1:
        return "major_functional"
    return "rewrite"


def review_response_for_grade(grade: int) -> str:
    grade = clamp_axiom_grade(grade)
    payload = {
        "dimension": "Correctness Verification",
        "axiom_grade": grade,
        "score": axiom_scalar_score(grade),
        "verdict": axiom_verdict(grade),
        "functional_correctness": grade >= 3,
        "repair_effort": repair_effort_for_grade(grade),
        "evidence_type": "uncertain",
        "summary": AXIOM_GRADE_DESCRIPTIONS[grade],
        "evidence": [
            f"Use AXIOM grade {grade}: {AXIOM_GRADE_DESCRIPTIONS[grade]}",
            "Static baseline item derived from the reference target label rather than model-generated reasoning.",
        ],
    }
    return "<review>\n" + json.dumps(payload, ensure_ascii=False) + "\n</review>"


def main() -> None:
    args = parse_args()
    samples = load_codecriticbench_dataset(str(args.input), start=0, limit=None)
    items = []
    for sample in samples:
        grade = clamp_axiom_grade(sample["axiom_target_grade"])
        item = {
            "instruction": build_instruction(sample, args.dimension),
            "response": [review_response_for_grade(grade)],
            "q_value": [axiom_value_target(grade)],
            "train_lm": True,
            "dataset_index": sample.get("dataset_index"),
            "source": sample.get("source"),
            "subset": sample.get("subset"),
            "target_dimension": args.dimension,
            "terminal_tag": "static_exact",
            "terminal_q_value": axiom_value_target(grade),
            "terminal_error": None,
            "parsed_score": axiom_scalar_score(grade),
            "parsed_axiom_grade": grade,
            "target_score": axiom_scalar_score(grade),
            "target_axiom_grade": grade,
            "is_best_path": True,
            "value_loss_weight": args.value_loss_weight,
            "lm_loss_weight": args.lm_loss_weight,
            "data_split": "static",
            "synthetic_type": "static_exact",
        }
        items.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as writer:
        for item in items:
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "items": len(items),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
