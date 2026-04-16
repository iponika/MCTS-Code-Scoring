from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from magicoder.preprocess_review_mcts_data import build_instruction, iter_records
from magicoder.review_policy_value_inference import (
    generate_response,
    load_policy,
    load_value_model,
    resolve_model_path,
    score_response,
)
from magicoder.review_value_guided_evaluator import candidate_sort_key
from magicoder.prompt_template import QWEN_REVIEW_STEP_PROMPT
from magicoder.axiom_scoring import (
    AXIOM_SCALE_TEXT,
    axiom_grade_from_codecritic,
    axiom_scalar_score,
    parse_axiom_grade,
)


DEFAULT_DIMENSIONS = [
    "Correctness Verification",
    "Time Complexity Optimization",
    "Space Complexity Control",
    "Code Readability Enhancement",
    "Robustness Validation",
    "Algorithm Optimization",
    "Comprehensive Testing",
    "Output Format Compliance",
    "Code Style Consistency",
    "Maintainability",
]


def load_record(path: str, index: int) -> dict[str, Any]:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        with input_path.open("r", encoding="utf-8") as handle:
            for current_index, line in enumerate(handle):
                if current_index == index:
                    return json.loads(line)
        raise IndexError(f"{path} has no record at index {index}")

    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        try:
            return payload[index]
        except IndexError as exc:
            raise IndexError(f"{path} has no record at index {index}") from exc
    if index != 0:
        raise IndexError(f"{path} contains a single JSON record; index must be 0")
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported JSON payload in {path}")


def sample_from_record(record: dict[str, Any]) -> dict[str, Any]:
    if "candidate_code" in record:
        return record
    if "answer" in record and "question" in record:
        axiom_grade = axiom_grade_from_codecritic(record.get("correctness"), record.get("score"))
        return {
            "problem": record["question"],
            "candidate_code": record["answer"],
            "tests": list(record.get("public_test", {}).get("input", []) or [])
            + list(record.get("private_test", {}).get("input", []) or []),
            "reference_scores": {
                dimension: score
                for dimension, score in zip(record.get("checklist_dimensions", []), record.get("checklist_scores", []))
            },
            "source": record.get("source"),
            "subset": record.get("subset"),
            "dataset_index": record.get("dataset_index"),
            "axiom_target_grade": axiom_grade,
            "axiom_target_score": axiom_scalar_score(axiom_grade) if axiom_grade is not None else None,
        }
    raise ValueError("Input record must be a review MCTS sample or a CodeCriticBench-style raw sample.")


def fill_sample_metadata(sample: dict[str, Any], record_index: int) -> dict[str, Any]:
    if sample.get("dataset_index") is None:
        sample["dataset_index"] = record_index
    return sample


def dimensions_for_sample(sample: dict[str, Any], requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    if sample.get("reference_scores"):
        return list(sample["reference_scores"].keys())
    return DEFAULT_DIMENSIONS


def prompt_for_dimension(
    sample: dict[str, Any],
    dimension: str,
    partial_response: str = "",
    force_final: bool = False,
    parse_error: dict[str, Any] | None = None,
) -> str:
    instruction = build_instruction(sample, dimension)
    if force_final:
        instruction += (
            "\n\nCurrent generation mode: finish now. Start your next output with <review> and end it with </review>. "
            "Do not add more <step> blocks. Output exactly one valid JSON object inside the review tags. "
            "Required JSON keys: dimension, axiom_grade, score, verdict, functional_correctness, repair_effort, summary, evidence. "
            "The evidence value must be a JSON array of strings using square brackets only. "
            "If a previous review block exists, ignore it and output a corrected final review. "
            "Use any <value_feedback> blocks as private guidance; do not quote or repeat them in the review."
        )
        if parse_error:
            instruction += (
                "\nPrevious final review parse error: "
                f"{parse_error.get('error')}: {parse_error.get('message', '')}. "
                "Correct the JSON syntax in the next review block."
            )
    else:
        instruction += (
            "\n\nCurrent generation mode: output exactly one next <step>...</step> reasoning block unless the review is already ready. "
            "Use any <value_feedback> blocks as private guidance; do not quote or repeat them."
        )
    return QWEN_REVIEW_STEP_PROMPT.format(instruction=instruction, response=partial_response)


def rethink_feedback(best: dict[str, Any], threshold: float, score_key: str) -> str:
    value = best["value_score"][score_key]
    return (
        "<value_feedback>\n"
        f"Value feedback: the previous continuation scored {value:.4f} on {score_key}, "
        f"below the rethink threshold {threshold:.4f}. Rethink the dimension with more concrete evidence and avoid unsupported claims.\n"
        "</value_feedback>\n"
    )


def should_finish(response: str) -> bool:
    return "<review>" in response and "</review>" in response


def parse_final_review(text: str) -> dict[str, Any]:
    if "<review>" not in text or "</review>" not in text:
        return {"ok": False, "error": "missing_review_tags"}
    review_text = text.rsplit("<review>", 1)[1].split("</review>", 1)[0].strip()
    if "{" in review_text and "}" in review_text:
        review_text = review_text[review_text.find("{") : review_text.rfind("}") + 1]
    try:
        parsed = json.loads(review_text)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": "invalid_review_json", "raw_review": review_text, "message": str(exc)}
    return {"ok": True, "parsed": parsed}


def parsed_review_score(final_review_parse: dict[str, Any]) -> float | None:
    if not final_review_parse.get("ok"):
        return None
    parsed = final_review_parse.get("parsed", {})
    grade = parse_axiom_grade(parsed)
    if grade is not None:
        return axiom_scalar_score(grade)
    score = parsed.get("score")
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def parsed_review_grade(final_review_parse: dict[str, Any]) -> int | None:
    if not final_review_parse.get("ok"):
        return None
    return parse_axiom_grade(final_review_parse.get("parsed", {}))


def score_delta(parsed_score: float | None, reference_score: Any) -> float | None:
    if parsed_score is None or reference_score is None:
        return None
    try:
        return round(parsed_score - float(reference_score), 6)
    except (TypeError, ValueError):
        return None


def retry_partial_response(partial_response: str, final_review_parse: dict[str, Any]) -> str:
    if final_review_parse.get("error") != "invalid_review_json":
        return partial_response
    return ""


def value_spread(candidates: list[dict[str, Any]], score_key: str) -> float:
    values = [float(candidate["value_score"][score_key]) for candidate in candidates]
    if not values:
        return 0.0
    return max(values) - min(values)


def evaluate_dimension(
    sample: dict[str, Any],
    dimension: str,
    policy_model,
    value_model,
    tokenizer,
    args,
) -> dict[str, Any]:
    partial_response = ""
    trace: list[dict[str, Any]] = []
    rethink_count = 0

    for step_index in range(args.max_steps):
        force_final = step_index == args.max_steps - 1
        max_new_tokens = (args.final_max_new_tokens or args.max_new_tokens) if force_final else args.max_new_tokens
        stop = "</review>" if force_final else ["</step>", "</review>"]
        prompt = prompt_for_dimension(sample, dimension, partial_response, force_final=force_final)
        candidates = []
        for candidate_index in range(args.num_candidates):
            with torch.no_grad():
                continuation = generate_response(
                    policy_model=policy_model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop=stop,
                )
                value_score = score_response(value_model, tokenizer, prompt, continuation)
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "continuation": continuation,
                    "value_score": value_score,
                }
            )

        best = max(candidates, key=lambda candidate: candidate_sort_key(candidate, args.score_key))
        selected_value = float(best["value_score"][args.score_key])
        spread = value_spread(candidates, args.score_key)
        candidate_rethink_reasons = []
        if selected_value < args.rethink_threshold:
            candidate_rethink_reasons.append("low_selected_value")
        if args.rethink_spread_threshold > 0 and spread >= args.rethink_spread_threshold:
            candidate_rethink_reasons.append("high_candidate_value_spread")
        rethink = bool(candidate_rethink_reasons) and rethink_count < args.max_rethinks and not force_final
        accepted = not rethink
        if rethink:
            rethink_count += 1
            partial_response += rethink_feedback(best, args.rethink_threshold, args.score_key)
        else:
            partial_response += best["continuation"].strip() + "\n"

        trace.append(
            {
                "step_index": step_index,
                "force_final": force_final,
                "selected_candidate_index": best["candidate_index"],
                "selected_value": best["value_score"],
                "value_spread": round(spread, 6),
                "accepted": accepted,
                "rethink": rethink,
                "rethink_reasons": candidate_rethink_reasons if rethink else [],
                "blocked_rethink_reasons": candidate_rethink_reasons if candidate_rethink_reasons and not rethink else [],
                "candidates": candidates,
            }
        )

        if accepted and should_finish(best["continuation"]):
            break

    final_review_parse = parse_final_review(partial_response)
    final_retries = []
    for retry_index in range(args.max_final_retries):
        if final_review_parse["ok"]:
            break
        clean_partial_response = retry_partial_response(partial_response, final_review_parse)
        retry_prompt = prompt_for_dimension(
            sample,
            dimension,
            clean_partial_response,
            force_final=True,
            parse_error=final_review_parse,
        )
        with torch.no_grad():
            continuation = generate_response(
                policy_model=policy_model,
                tokenizer=tokenizer,
                prompt=retry_prompt,
                max_new_tokens=args.final_max_new_tokens or args.max_new_tokens,
                temperature=args.final_temperature,
                top_p=args.top_p,
                stop="</review>",
            )
            retry_value_score = score_response(value_model, tokenizer, retry_prompt, continuation)
        partial_response += continuation.strip() + "\n"
        final_review_parse = parse_final_review(partial_response)
        final_retries.append(
            {
                "retry_index": retry_index,
                "continuation": continuation,
                "value_score": retry_value_score,
                "final_review_parse": final_review_parse,
            }
        )

    final_prompt = prompt_for_dimension(sample, dimension, "", force_final=True)
    final_value_score = score_response(value_model, tokenizer, final_prompt, partial_response)
    reference_score = sample.get("axiom_target_score")
    reference_grade = sample.get("axiom_target_grade")
    parsed_score = parsed_review_score(final_review_parse)
    parsed_grade = parsed_review_grade(final_review_parse)
    parsed_score_delta = score_delta(parsed_score, reference_score)
    return {
        "dimension": dimension,
        "score_scale": "axiom_0_5_scalar_0_100",
        "score_semantics": AXIOM_SCALE_TEXT,
        "reference_axiom_grade": reference_grade,
        "reference_score": reference_score,
        "legacy_dimension_score": sample.get("reference_scores", {}).get(dimension),
        "parsed_axiom_grade": parsed_grade,
        "parsed_score": parsed_score,
        "score_delta": parsed_score_delta,
        "abs_score_delta": abs(parsed_score_delta) if parsed_score_delta is not None else None,
        "final_review": partial_response,
        "final_review_parse": final_review_parse,
        "final_value_score": final_value_score,
        "final_retries": final_retries,
        "rethink_count": rethink_count,
        "trace": trace,
    }


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def summarize_dimension_results(dimension_results: list[dict[str, Any]]) -> dict[str, Any]:
    parsed_scores = [result["parsed_score"] for result in dimension_results if result.get("parsed_score") is not None]
    reference_scores = [
        float(result["reference_score"])
        for result in dimension_results
        if result.get("reference_score") is not None
    ]
    abs_score_deltas = [
        result["abs_score_delta"]
        for result in dimension_results
        if result.get("abs_score_delta") is not None
    ]
    valid_review_count = sum(1 for result in dimension_results if result.get("final_review_parse", {}).get("ok"))
    return {
        "dimension_count": len(dimension_results),
        "valid_review_count": valid_review_count,
        "valid_review_rate": round(valid_review_count / len(dimension_results), 6) if dimension_results else 0.0,
        "mean_parsed_score": mean(parsed_scores),
        "mean_reference_score": mean(reference_scores),
        "mean_abs_score_delta": mean(abs_score_deltas),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run value-guided code review evaluation.")
    parser.add_argument("--policy_model_path", required=True)
    parser.add_argument("--value_model_path")
    parser.add_argument(
        "--share_policy_value_model",
        action="store_true",
        help="Load one value-head model and use its pretrained_model for policy generation.",
    )
    parser.add_argument("--input_record", required=True, help="Review MCTS sample JSON/JSONL or raw CodeCriticBench JSON/JSONL.")
    parser.add_argument("--record_index", type=int, default=0)
    parser.add_argument("--output_file")
    parser.add_argument("--dimensions", nargs="*", help="Optional dimension subset. Defaults to dataset dimensions.")
    parser.add_argument("--max_dimensions", type=int, default=0, help="0 means all selected dimensions.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--max_steps", type=int, default=3)
    parser.add_argument("--num_candidates", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--final_max_new_tokens", type=int, default=0, help="0 means reuse --max_new_tokens for final review generation and retries.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--score_key", choices=["last_value", "response_mean_value", "response_min_value"], default="last_value")
    parser.add_argument("--rethink_threshold", type=float, default=-0.2)
    parser.add_argument("--rethink_spread_threshold", type=float, default=0.0, help="0 disables spread-based rethink.")
    parser.add_argument("--max_rethinks", type=int, default=1)
    parser.add_argument("--max_final_retries", type=int, default=1)
    parser.add_argument("--final_temperature", type=float, default=0.0)
    args = parser.parse_args()

    record = load_record(args.input_record, args.record_index)
    sample = fill_sample_metadata(sample_from_record(record), args.record_index)
    dimensions = dimensions_for_sample(sample, args.dimensions)
    if args.max_dimensions > 0:
        dimensions = dimensions[: args.max_dimensions]

    policy_model_path = resolve_model_path(args.policy_model_path)
    value_model_path = resolve_model_path(args.value_model_path or args.policy_model_path)
    tokenizer = AutoTokenizer.from_pretrained(policy_model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.share_policy_value_model or value_model_path == policy_model_path:
        value_model = load_value_model(value_model_path, args.device, args.dtype)
        policy_model = value_model.pretrained_model
        policy_model.eval()
    else:
        policy_model = load_policy(policy_model_path, args.device, args.dtype)
        value_model = load_value_model(value_model_path, args.device, args.dtype)

    dimension_results = [
        evaluate_dimension(sample, dimension, policy_model, value_model, tokenizer, args)
        for dimension in dimensions
    ]
    result = {
        "input_record": args.input_record,
        "record_index": args.record_index,
        "source": sample.get("source"),
        "subset": sample.get("subset"),
        "dataset_index": sample.get("dataset_index"),
        "problem": sample.get("problem"),
        "candidate_code": sample.get("candidate_code"),
        "evaluation_summary": summarize_dimension_results(dimension_results),
        "dimensions": dimension_results,
    }

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
