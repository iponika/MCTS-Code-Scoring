from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, set_seed

from magicoder.preprocess_review_mcts_data import build_instruction, iter_records
from magicoder.review_policy_value_inference import (
    generate_response,
    load_policy,
    load_value_model,
    resolve_model_path,
    score_response,
)
from magicoder.review_value_guided_evaluator import VALUE_SCORE_KEYS
from magicoder.prompt_template import QWEN_REVIEW_STEP_PROMPT
from magicoder.axiom_scoring import (
    AXIOM_SCALE_TEXT,
    axiom_grade_from_scalar,
    axiom_grade_from_codecritic,
    axiom_scalar_score,
    clamp_axiom_grade,
    parse_axiom_grade,
)


QWEN_REVIEW_FINAL_ONLY_PROMPT = """You are an exceptionally intelligent code scoring model.
@@ Instruction
Score the candidate code using the AXIOM 0-5 ordinal code-quality scale.
Textual critique is only supporting evidence; the primary output is a stable scalar grade.
Functionality is the primary boundary: grades 3-5 are functionally correct, grades 0-2 are not.
Calibration rule: do not assign grades 0-2 merely because an issue is suspected or because no tests are available. Low grades require concrete visible evidence such as a syntax/runtime error, missing required I/O, unrelated or empty code, a direct contradiction of the task, or a simple counterexample grounded in the prompt/tests. If the implementation is complete and plausibly functional but you cannot prove a functional defect, keep the grade in 3-5 and use repair_effort to express quality/refactoring concerns.

{instruction}

Output exactly one JSON object wrapped in <review> tags.
Do not output <step> blocks, markdown, code fixes, or prose outside the tags.
Required JSON keys: axiom_grade, score, verdict, functional_correctness, repair_effort, summary, evidence.
Use a compact evidence array with at most 2 short strings.

@@ Response
"""


DEFAULT_DIMENSIONS = [
    "Correctness Verification",
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
    if "inst" in record and "code" in record and "score" in record:
        axiom_grade = clamp_axiom_grade(record.get("score", 0))
        return {
            "problem": record["inst"],
            "candidate_code": record["code"],
            "tests": [],
            "reference_scores": {},
            "source": record.get("source") or "axiom",
            "subset": record.get("subset"),
            "dataset_index": record.get("dataset_index"),
            "language": record.get("lang") or "unknown",
            "axiom_target_grade": axiom_grade,
            "axiom_target_score": axiom_scalar_score(axiom_grade),
        }
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
            "language": record.get("language") or "python",
            "axiom_target_grade": axiom_grade,
            "axiom_target_score": axiom_scalar_score(axiom_grade) if axiom_grade is not None else None,
        }
    raise ValueError("Input record must be a review MCTS sample, AXIOM raw sample, or CodeCriticBench-style raw sample.")


def fill_sample_metadata(sample: dict[str, Any], record_index: int) -> dict[str, Any]:
    if sample.get("dataset_index") is None:
        sample["dataset_index"] = record_index
    return sample


def dimensions_for_sample(sample: dict[str, Any], requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return DEFAULT_DIMENSIONS


def prompt_for_dimension(
    sample: dict[str, Any],
    dimension: str,
    partial_response: str = "",
    force_final: bool = False,
    parse_error: dict[str, Any] | None = None,
    final_only: bool = False,
    max_problem_chars: int = 3500,
    max_code_chars: int = 3500,
    mark_code_truncation_inside_block: bool = True,
    show_tests_in_prompt: bool = False,
) -> str:
    instruction = build_instruction(
        sample,
        dimension,
        max_problem_chars=max_problem_chars,
        max_code_chars=max_code_chars,
        mark_code_truncation_inside_block=mark_code_truncation_inside_block,
        show_tests_in_prompt=show_tests_in_prompt,
    )
    if final_only:
        if parse_error:
            instruction += (
                "\n\nPrevious final review parse error: "
                f"{parse_error.get('error')}: {parse_error.get('message', '')}. "
                "Correct the JSON syntax in the next review block."
            )
        return QWEN_REVIEW_FINAL_ONLY_PROMPT.format(instruction=instruction)
    if force_final:
        instruction += (
            "\n\nCurrent generation mode: finish now. Start your next output with <review> and end it with </review>. "
            "Do not add more <step> blocks. Output exactly one valid JSON object inside the review tags. "
            "Required JSON keys: axiom_grade, score, verdict, functional_correctness, repair_effort, summary, evidence. "
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
        f"below the rethink threshold {threshold:.4f}. Rethink the code score with more concrete evidence and avoid unsupported claims.\n"
        "</value_feedback>\n"
    )


def should_finish(response: str) -> bool:
    return "<review>" in response and "</review>" in response


def _escape_control_chars_inside_json_strings(text: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if escaped:
            result.append(char)
            escaped = False
            continue
        if char == "\\" and in_string:
            result.append(char)
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            result.append(char)
            continue
        if in_string and char == "\n":
            result.append("\\n")
            continue
        if in_string and char == "\r":
            result.append("\\r")
            continue
        if in_string and char == "\t":
            result.append("\\t")
            continue
        result.append(char)
    return "".join(result)


def _balanced_json_prefix(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _minimal_review_payload(text: str) -> dict[str, Any] | None:
    grade = lenient_axiom_grade(text)
    if grade is None:
        return None
    score_matches = re.findall(r'"score"\s*:\s*([0-9]+(?:\.\d+)?)', text)
    if score_matches:
        score = float(score_matches[-1])
    else:
        score = axiom_scalar_score(grade)
    verdict_match = re.findall(r'"verdict"\s*:\s*"([^"\n\r]{0,120})"', text)
    summary_match = re.findall(r'"summary"\s*:\s*"([^"\n\r]{0,240})"', text)
    return {
        "axiom_grade": grade,
        "score": score,
        "verdict": verdict_match[-1] if verdict_match else "recovered_from_malformed_review",
        "functional_correctness": None,
        "repair_effort": None,
        "summary": summary_match[-1] if summary_match else "Recovered grade from malformed final review.",
        "evidence": [],
    }


def _parse_review_json(review_text: str) -> dict[str, Any]:
    candidates = [review_text.strip()]
    balanced = _balanced_json_prefix(review_text)
    if balanced and balanced not in candidates:
        candidates.append(balanced)

    for candidate_index, candidate in enumerate(candidates):
        try:
            result = {"ok": True, "parsed": json.loads(candidate)}
            if candidate_index > 0:
                result["recovered"] = True
                result["recovery_method"] = "balanced_json_prefix"
            return result
        except json.JSONDecodeError:
            pass
        sanitized = _escape_control_chars_inside_json_strings(candidate)
        if sanitized != candidate:
            try:
                return {"ok": True, "parsed": json.loads(sanitized), "recovered": True, "recovery_method": "escaped_control_chars"}
            except json.JSONDecodeError:
                pass

    fallback = _minimal_review_payload(review_text)
    if fallback is not None:
        return {"ok": True, "parsed": fallback, "recovered": True, "recovery_method": "grade_fallback"}

    try:
        json.loads(review_text)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": "invalid_review_json", "raw_review": review_text, "message": str(exc)}
    return {"ok": False, "error": "invalid_review_json", "raw_review": review_text, "message": "parsed JSON was not an object"}


def parse_final_review(text: str) -> dict[str, Any]:
    if "<review>" not in text or "</review>" not in text:
        return {"ok": False, "error": "missing_review_tags"}
    review_text = text.rsplit("<review>", 1)[1].split("</review>", 1)[0].strip()
    if "{" in review_text and "}" in review_text:
        review_text = review_text[review_text.find("{") : review_text.rfind("}") + 1]
    return _parse_review_json(review_text)


def lenient_axiom_grade(text: str) -> int | None:
    if not text:
        return None
    matches = re.findall(r'"(?:axiom_grade|grade)"\s*:\s*([0-5](?:\.\d+)?)', text)
    if not matches:
        matches = re.findall(r'\b(?:axiom_grade|grade)\b[^0-9]{0,20}([0-5](?:\.\d+)?)', text, flags=re.IGNORECASE)
    if matches:
        return clamp_axiom_grade(float(matches[-1]))
    score_matches = re.findall(r'"score"\s*:\s*([0-9]+(?:\.\d+)?)', text)
    if score_matches:
        return axiom_grade_from_scalar(float(score_matches[-1]), max_score=100.0)
    return None


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


def neutral_value_score() -> dict[str, float | int]:
    return {
        "prompt_tokens": 0,
        "total_tokens": 0,
        "last_value": 0.0,
        "response_mean_value": 0.0,
        "response_min_value": 0.0,
        "response_conservative_value": 0.0,
        "response_max_value": 0.0,
    }


def final_candidate_sort_key(candidate: dict[str, Any], args, force_final: bool) -> tuple[float, float]:
    value_score = candidate["value_score"]
    primary = float(value_score[args.score_key])
    tie_break = float(value_score["last_value"])
    if not force_final:
        return primary, tie_break

    continuation = str(candidate.get("continuation") or "")
    grade = lenient_axiom_grade(continuation)
    if grade is None:
        primary -= float(args.format_penalty)
    elif grade < 3 and not concrete_low_grade_evidence(continuation):
        primary -= float(args.low_grade_no_evidence_penalty)
    return primary, tie_break


def concrete_low_grade_evidence(text: str) -> bool:
    lowered = text.lower()
    evidence_markers = (
        "fails",
        "incorrect",
        "wrong",
        "exception",
        "runtime error",
        "syntax error",
        "does not",
        "missing",
        "mismatch",
        "counterexample",
        "test",
        "expected",
        "actual",
    )
    return any(marker in lowered for marker in evidence_markers)


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
        force_final = args.final_only_json or step_index == args.max_steps - 1
        max_new_tokens = (args.final_max_new_tokens or args.max_new_tokens) if force_final else args.max_new_tokens
        stop = "</review>" if force_final else ["</step>", "</review>"]
        prompt = prompt_for_dimension(
            sample,
            dimension,
            partial_response,
            force_final=force_final,
            final_only=args.final_only_json,
            max_problem_chars=args.max_problem_chars,
            max_code_chars=args.max_code_chars,
            mark_code_truncation_inside_block=args.mark_code_truncation_inside_block,
            show_tests_in_prompt=args.show_tests_in_prompt,
        )
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
                value_score = score_response(value_model, tokenizer, prompt, continuation) if value_model is not None else neutral_value_score()
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "continuation": continuation,
                    "value_score": value_score,
                }
            )

        best = max(candidates, key=lambda candidate: final_candidate_sort_key(candidate, args, force_final))
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
        if args.final_only_json:
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
            final_only=args.final_only_json,
            max_problem_chars=args.max_problem_chars,
            max_code_chars=args.max_code_chars,
            mark_code_truncation_inside_block=args.mark_code_truncation_inside_block,
            show_tests_in_prompt=args.show_tests_in_prompt,
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
            retry_value_score = score_response(value_model, tokenizer, retry_prompt, continuation) if value_model is not None else neutral_value_score()
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

    final_prompt = prompt_for_dimension(
        sample,
        dimension,
        "",
        force_final=True,
        final_only=args.final_only_json,
        max_problem_chars=args.max_problem_chars,
        max_code_chars=args.max_code_chars,
        mark_code_truncation_inside_block=args.mark_code_truncation_inside_block,
        show_tests_in_prompt=args.show_tests_in_prompt,
    )
    final_value_score = score_response(value_model, tokenizer, final_prompt, partial_response) if value_model is not None else neutral_value_score()
    reference_score = sample.get("axiom_target_score")
    reference_grade = sample.get("axiom_target_grade")
    reference_interval = sample.get("axiom_target_interval")
    parsed_score = parsed_review_score(final_review_parse)
    parsed_grade = parsed_review_grade(final_review_parse)
    lenient_grade = lenient_axiom_grade(partial_response)
    parsed_score_delta = score_delta(parsed_score, reference_score)
    return {
        "dimension": dimension,
        "score_scale": "axiom_0_5_scalar_0_100",
        "score_semantics": AXIOM_SCALE_TEXT,
        "reference_axiom_grade": reference_grade,
        "reference_axiom_interval": reference_interval,
        "reference_score": reference_score,
        "label_type": sample.get("label_type"),
        "pair_id": sample.get("pair_id"),
        "pair_role": sample.get("pair_role"),
        "legacy_dimension_score": sample.get("reference_scores", {}).get(dimension),
        "parsed_axiom_grade": parsed_grade,
        "lenient_axiom_grade": lenient_grade,
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
    parser.add_argument("--skip_value_scoring", action="store_true", help="Do not load/use a value model; useful for direct-generation baselines.")
    parser.add_argument("--input_record", required=True, help="Review MCTS sample JSON/JSONL or raw CodeCriticBench JSON/JSONL.")
    parser.add_argument("--record_index", type=int, default=0)
    parser.add_argument("--output_file")
    parser.add_argument("--dimensions", nargs="*", help="Optional legacy dimension subset. Defaults to correctness only.")
    parser.add_argument("--max_dimensions", type=int, default=0, help="0 means all selected dimensions.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--max_steps", type=int, default=3)
    parser.add_argument("--num_candidates", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--final_max_new_tokens", type=int, default=0, help="0 means reuse --max_new_tokens for final review generation and retries.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--score_key", choices=VALUE_SCORE_KEYS, default="last_value")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--final_only_json", action="store_true", help="Generate only one compact final <review> JSON block; no step reasoning.")
    parser.add_argument("--show_tests_in_prompt", action="store_true", help="Expose dataset tests to the reviewer prompt for oracle diagnostics. Default hides tests.")
    parser.add_argument("--max_problem_chars", type=int, default=3500, help="Maximum task-description characters included in review prompts. 0 keeps full text.")
    parser.add_argument("--max_code_chars", type=int, default=3500, help="Maximum candidate-code characters included in review prompts. 0 keeps full text.")
    parser.add_argument(
        "--mark_code_truncation_inside_block",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When truncating candidate code, insert the truncation marker inside the code block. Disable for clean evaluation to avoid treating truncation as a code defect.",
    )
    parser.add_argument("--format_penalty", type=float, default=1.0, help="Final-candidate value penalty when no AXIOM grade can be parsed.")
    parser.add_argument("--low_grade_no_evidence_penalty", type=float, default=0.4, help="Final-candidate value penalty for grades 0-2 without concrete defect evidence.")
    parser.add_argument("--rethink_threshold", type=float, default=-0.2)
    parser.add_argument("--rethink_spread_threshold", type=float, default=0.0, help="0 disables spread-based rethink.")
    parser.add_argument("--max_rethinks", type=int, default=1)
    parser.add_argument("--max_final_retries", type=int, default=1)
    parser.add_argument("--final_temperature", type=float, default=0.0)
    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)

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
    if args.skip_value_scoring and (args.share_policy_value_model or value_model_path == policy_model_path):
        value_wrapper = load_value_model(value_model_path, args.device, args.dtype)
        policy_model = value_wrapper.pretrained_model
        policy_model.eval()
        value_model = None
    elif args.skip_value_scoring:
        policy_model = load_policy(policy_model_path, args.device, args.dtype)
        value_model = None
    elif args.share_policy_value_model or value_model_path == policy_model_path:
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
