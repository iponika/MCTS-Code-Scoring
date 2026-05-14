from __future__ import annotations

import argparse
import json
from pathlib import Path

from magicoder.axiom_scoring import AXIOM_GRADE_DESCRIPTIONS, axiom_scalar_score, axiom_value_target, axiom_verdict, clamp_axiom_grade
from magicoder.preprocess_review_mcts_data import (
    attach_qwen_messages,
    build_assistant_parts,
    build_instruction,
    qwen_assistant_content_from_responses,
)
from mcts_math.review_utils import load_codecriticbench_dataset
from shared.prompt_contract import build_base_static_user_content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert prepared review samples into exact-label train_multi review data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimension", default="Correctness Verification")
    parser.add_argument("--value_loss_weight", type=float, default=1.0)
    parser.add_argument("--lm_loss_weight", type=float, default=0.15)
    parser.add_argument(
        "--prompt_variant",
        choices=["default", "base_static"],
        default="default",
        help="default preserves the shared Direct/MCTS training prompt; base_static uses a direct final-only user prompt.",
    )
    parser.add_argument(
        "--output_schema",
        choices=["axiom", "codecritic_correctness"],
        default="axiom",
        help="Structured review schema to emit. Use codecritic_correctness for CodeCriticBench 1-10 correctness experiments.",
    )
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


def attach_base_static_messages(item: dict) -> None:
    responses = [str(segment or "").strip() for segment in item.get("response", []) if str(segment or "").strip()]
    assistant_content = qwen_assistant_content_from_responses(responses)
    item["messages"] = [
        {"role": "user", "content": build_base_static_user_content(str(item.get("instruction") or ""))},
        {"role": "assistant", "content": assistant_content},
    ]
    item["assistant_parts"] = build_assistant_parts(item)
    item["training_format"] = "qwen_messages_base_static_review"


def codecritic_value_target(score: float | int) -> float:
    return round((float(score) - 1.0) / 4.5 - 1.0, 4)


def codecritic_review_response_for_score(score: float | int) -> str:
    value = max(1, min(10, int(round(float(score)))))
    payload = {
        "correctness_score": value,
        "dimension": "Correctness Verification",
        "evidence_type": "uncertain",
        "summary": f"Reference static label assigns Correctness Verification score {value}.",
        "evidence": [
            f"Use CodeCriticBench Correctness Verification score {value}.",
            "Static baseline item derived from the reference target label rather than model-generated reasoning.",
        ],
    }
    return "<review>\n" + json.dumps(payload, ensure_ascii=False) + "\n</review>"


def attach_codecritic_static_messages(item: dict) -> None:
    responses = [str(segment or "").strip() for segment in item.get("response", []) if str(segment or "").strip()]
    assistant_content = qwen_assistant_content_from_responses(responses)
    item["messages"] = [
        {
            "role": "user",
            "content": (
                "You are a direct CodeCriticBench correctness scoring model.\n"
                "Evaluate only Correctness Verification on a 1-10 integer scale using the problem, answer/code, "
                "visible tests, and provided correctness checklist.\n"
                "Return exactly one <review> JSON block containing correctness_score. Do not produce intermediate "
                "reasoning, <think> blocks, <step> blocks, markdown fences, code fixes, q_value, reward, or training metadata.\n\n"
                + str(item.get("instruction") or "").strip()
            ),
        },
        {"role": "assistant", "content": assistant_content},
    ]
    item["assistant_parts"] = build_assistant_parts(item)
    item["training_format"] = "qwen_messages_codecritic_static_review"


def main() -> None:
    args = parse_args()
    samples = load_codecriticbench_dataset(str(args.input), start=0, limit=None)
    items = []
    for sample in samples:
        instruction = build_instruction(sample, args.dimension)
        if args.output_schema == "codecritic_correctness":
            score = float(sample.get("target_correctness_score") or sample.get("reference_scores", {}).get(args.dimension) or 0)
            target_value = codecritic_value_target(score)
            item = {
                "instruction": instruction,
                "response": [codecritic_review_response_for_score(score)],
                "q_value": [target_value],
                "train_lm": True,
                "dataset_index": sample.get("dataset_index"),
                "source": sample.get("source"),
                "subset": sample.get("subset"),
                "target_dimension": args.dimension,
                "terminal_tag": "static_exact",
                "terminal_q_value": target_value,
                "terminal_error": None,
                "score_scale": "codecritic_correctness_1_10",
                "parsed_score": int(round(score)),
                "target_score": score,
                "target_correctness_score": score,
                "is_best_path": True,
                "value_loss_weight": args.value_loss_weight,
                "lm_loss_weight": args.lm_loss_weight,
                "data_split": "static",
                "synthetic_type": "static_exact_codecritic_correctness",
            }
            attach_codecritic_static_messages(item)
        else:
            grade = clamp_axiom_grade(sample["axiom_target_grade"])
            item = {
                "instruction": instruction,
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
            if args.prompt_variant == "base_static":
                attach_base_static_messages(item)
            else:
                attach_qwen_messages(item)
        items.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as writer:
        for item in items:
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "items": len(items),
        "prompt_variant": args.prompt_variant,
        "output_schema": args.output_schema,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
