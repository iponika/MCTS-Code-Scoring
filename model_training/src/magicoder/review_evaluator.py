from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, set_seed

from magicoder.preprocess_review_mcts_data import iter_records
from magicoder.review_policy_value_inference import (
    generate_response,
    load_policy,
    load_value_model,
    resolve_model_path,
    score_response,
)
from magicoder.review_value_guided_evaluator import VALUE_SCORE_KEYS
try:
    from shared.prompt_contract import (
        FINAL_REVIEW_PREFILL,
        build_base_static_prompt_from_sample,
        build_review_prompt_from_sample,
        prompt_to_chat_messages,
        render_eval_prompt,
    )
    _HAS_SHARED = True
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from shared.prompt_contract import (
        FINAL_REVIEW_PREFILL,
        build_base_static_prompt_from_sample,
        build_review_prompt_from_sample,
        prompt_to_chat_messages,
        render_eval_prompt,
    )
    _HAS_SHARED = True
from magicoder.axiom_scoring import (
    AXIOM_SCALE_TEXT,
    axiom_grade_from_scalar,
    axiom_grade_from_codecritic,
    axiom_scalar_score,
    clamp_axiom_grade,
    parse_axiom_grade,
)


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
        reference_scores = {
            dimension: score
            for dimension, score in zip(record.get("checklist_dimensions", []), record.get("checklist_scores", []))
        }
        correctness_score = float(reference_scores.get("Correctness Verification", record.get("score") or 0))
        dimension_rubrics = {
            dimension: checklist
            for dimension, checklist in zip(record.get("checklist_dimensions", []), record.get("checklists", []))
            if dimension == "Correctness Verification"
        }
        return {
            "scoring_target": "codecritic_correctness",
            "score_scale": "codecritic_correctness_1_10",
            "target_correctness_score": correctness_score,
            "problem": record["question"],
            "candidate_code": record["answer"],
            "tests": list(record.get("public_test", {}).get("input", []) or [])
            + list(record.get("private_test", {}).get("input", []) or []),
            "tests_for_prompt": "\n".join(str(item) for item in list(record.get("public_test", {}).get("input", []) or [])[:5]),
            "reference_scores": {"Correctness Verification": correctness_score},
            "dimension_rubrics": dimension_rubrics,
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
    step_context_mode: str = "assistant_prefix",
    max_problem_chars: int = 3500,
    max_code_chars: int = 3500,
    mark_code_truncation_inside_block: bool = True,
    show_tests_in_prompt: bool = False,
    prompt_variant: str = "default",
) -> str:
    if prompt_variant == "base_static":
        return build_base_static_prompt_from_sample(
            sample,
            dimension=dimension,
            partial_solution=partial_response,
            max_problem_chars=max_problem_chars,
            max_code_chars=max_code_chars,
            mark_code_truncation_inside_block=mark_code_truncation_inside_block,
            show_tests_in_prompt=show_tests_in_prompt,
        )
    return build_review_prompt_from_sample(
        sample,
        dimension=dimension,
        partial_solution=partial_response,
        force_final=force_final,
        final_only=final_only,
        step_context_mode=step_context_mode,
        parse_error=parse_error,
        freeform_final_review=False,
        max_problem_chars=max_problem_chars,
        max_code_chars=max_code_chars,
        mark_code_truncation_inside_block=mark_code_truncation_inside_block,
        show_tests_in_prompt=show_tests_in_prompt,
        prompt_variant=prompt_variant,
    )


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


def extract_reasoning_artifacts(
    text: str,
    *,
    structured_reasoning: str | None = None,
) -> dict[str, str]:
    raw_text = str(text or "")
    cleaned = raw_text.strip()
    reasoning = str(structured_reasoning or "").strip()
    if reasoning:
        return {
            "raw_text": raw_text,
            "content": cleaned,
            "reasoning": reasoning,
            "reasoning_source": "structured_field",
        }

    if not cleaned:
        return {
            "raw_text": raw_text,
            "content": "",
            "reasoning": "",
            "reasoning_source": "none",
        }

    if "<think>" in cleaned:
        prefix, suffix = cleaned.split("<think>", 1)
        think_body = suffix
        trailing = ""
        if "</think>" in think_body:
            think_body, trailing = think_body.split("</think>", 1)
        content = (prefix + trailing).strip()
        return {
            "raw_text": raw_text,
            "content": content,
            "reasoning": think_body.strip(),
            "reasoning_source": "text_think_block",
        }

    if "</think>" in cleaned:
        think_body, trailing = cleaned.split("</think>", 1)
        return {
            "raw_text": raw_text,
            "content": trailing.strip(),
            "reasoning": think_body.strip(),
            "reasoning_source": "text_think_suffix",
        }

    return {
        "raw_text": raw_text,
        "content": cleaned,
        "reasoning": "",
        "reasoning_source": "none",
    }


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
    if "<review>" not in text and "</review>" in text:
        stripped = str(text or "").strip()
        review_text = stripped.split("</review>", 1)[0].strip()
        if review_text.startswith("{"):
            return _parse_review_json(review_text)
        if re.match(r"^[0-5]\s*,", review_text):
            return _parse_review_json('{"axiom_grade": ' + review_text)
        return {"ok": False, "error": "missing_review_tags"}
    if "<review>" not in text or "</review>" not in text:
        return {"ok": False, "error": "missing_review_tags"}
    review_text = text.rsplit("<review>", 1)[1].split("</review>", 1)[0].strip()
    if "{" in review_text and "}" in review_text:
        review_text = review_text[review_text.find("{") : review_text.rfind("}") + 1]
    return _parse_review_json(review_text)


def merge_final_review_continuation(continuation: str, *, anchored: bool) -> str:
    text = str(continuation or "").strip()
    if not anchored or not text or "<review>" in text:
        return text
    if text.startswith("{"):
        return "<review>\n" + text
    if re.match(r'^[0-5]\s*,', text):
        return FINAL_REVIEW_PREFILL + text
    if (
        "</review>" in text
        and (
            re.match(r'^[0-5]\s*,', text)
            or '"functional_correctness"' in text
            or '"repair_effort"' in text
            or '"evidence"' in text
        )
    ):
        return FINAL_REVIEW_PREFILL + text
    return text


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
    correctness_score = parsed.get("correctness_score")
    try:
        if correctness_score is not None:
            return float(correctness_score)
    except (TypeError, ValueError):
        pass
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


# ---------------------------------------------------------------------------
# Chat-template-aware prompt building (matches training format)
# ---------------------------------------------------------------------------


def _build_assistant_prefix(
    partial_response: str,
    force_final: bool,
) -> str:
    """Build the assistant continuation prefix for chat-template prompts.

    The model was trained with ``<think>`` blocks wrapping step content and
    ``<review>`` blocks immediately after ``</think>``.
    """
    text = partial_response.strip()
    if not text:
        return ""
    # Strip any leading <think> so we can re-add it uniformly.
    inner = text
    if inner.startswith("<think>"):
        inner = inner[len("<think>"):].lstrip("\n")
    if force_final:
        return f"<think>\n{inner}\n</think>\n\n"
    return f"<think>\n{inner}\n"


def build_chat_eval_prompt(
    tokenizer,
    sample: dict[str, Any],
    dimension: str,
    partial_response: str = "",
    *,
    force_final: bool = False,
    final_only: bool = False,
    parse_error: dict[str, Any] | None = None,
    step_context_mode: str = "instruction_context",
    max_problem_chars: int = 3500,
    max_code_chars: int = 3500,
    mark_code_truncation_inside_block: bool = True,
    show_tests_in_prompt: bool = False,
    enable_thinking: bool | None = None,
    prompt_variant: str = "default",
) -> str:
    """Render the existing raw-text prompt contract through the tokenizer chat template.

    This keeps chat-template evaluation behavior aligned with prompt_for_dimension(),
    including step_context_mode semantics, instead of maintaining a second prompt path.
    """
    raw_prompt = prompt_for_dimension(
        sample,
        dimension,
        partial_response,
        force_final=force_final,
        parse_error=parse_error,
        final_only=final_only,
        step_context_mode=step_context_mode,
        max_problem_chars=max_problem_chars,
        max_code_chars=max_code_chars,
        mark_code_truncation_inside_block=mark_code_truncation_inside_block,
        show_tests_in_prompt=show_tests_in_prompt,
        prompt_variant=prompt_variant,
    )
    messages = prompt_to_chat_messages(raw_prompt)
    user_content = messages[0]["content"]
    assistant_prefix = messages[1]["content"] if len(messages) > 1 and messages[1]["role"] == "assistant" else ""
    return render_eval_prompt(
        tokenizer,
        user_content,
        assistant_prefix=assistant_prefix,
        enable_thinking=enable_thinking,
    )


def _chat_template_enable_thinking(args) -> bool | None:
    raw = str(getattr(args, "chat_template_enable_thinking", "auto") or "auto").strip().lower()
    if raw in {"1", "true", "yes", "on", "think"}:
        return True
    if raw in {"0", "false", "no", "off", "no_think", "nothink"}:
        return False
    return None


def _effective_max_steps(args) -> int:
    """Return the total number of generation rounds (intermediate + final).

    If ``reasoning_steps`` is set, use ``reasoning_steps + 1`` to account
    for the final forced-review round.  Otherwise fall back to the legacy
    ``max_steps`` value (which conflates intermediate and final).
    """
    reasoning_steps = getattr(args, "reasoning_steps", None)
    if reasoning_steps is not None and reasoning_steps >= 0:
        return reasoning_steps + 1
    return args.max_steps


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
    use_chat_template = getattr(args, "use_chat_template", True)
    chat_template_enable_thinking = _chat_template_enable_thinking(args)
    total_steps = _effective_max_steps(args)

    for step_index in range(total_steps):
        force_final = args.final_only_json or step_index == total_steps - 1
        max_new_tokens = (args.final_max_new_tokens or args.max_new_tokens) if force_final else args.max_new_tokens
        stop = "</review>" if force_final else ["</think>", "</review>"]

        if use_chat_template:
            prompt = build_chat_eval_prompt(
                tokenizer,
                sample,
                dimension,
                partial_response,
                force_final=force_final,
                final_only=args.final_only_json,
                step_context_mode=args.step_context_mode,
                max_problem_chars=args.max_problem_chars,
                max_code_chars=args.max_code_chars,
                mark_code_truncation_inside_block=args.mark_code_truncation_inside_block,
                show_tests_in_prompt=args.show_tests_in_prompt,
                enable_thinking=chat_template_enable_thinking,
                prompt_variant=args.prompt_variant,
            )
        else:
            prompt = prompt_for_dimension(
                sample,
                dimension,
                partial_response,
                force_final=force_final,
                final_only=args.final_only_json,
                step_context_mode=args.step_context_mode,
                max_problem_chars=args.max_problem_chars,
                max_code_chars=args.max_code_chars,
                mark_code_truncation_inside_block=args.mark_code_truncation_inside_block,
                show_tests_in_prompt=args.show_tests_in_prompt,
                prompt_variant=args.prompt_variant,
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
                    chat_template_prompt=use_chat_template,
                )
                value_score = score_response(value_model, tokenizer, prompt, continuation) if value_model is not None else neutral_value_score()
            artifacts = extract_reasoning_artifacts(continuation)
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "continuation": continuation,
                    "content": artifacts["content"],
                    "reasoning": artifacts["reasoning"],
                    "reasoning_source": artifacts["reasoning_source"],
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
            if force_final:
                final_continuation = merge_final_review_continuation(best["continuation"], anchored=True)
                partial_response += final_continuation.strip() + "\n"
            else:
                reasoning_note = (best.get("reasoning") or best.get("content") or best["continuation"]).strip()
                partial_response += reasoning_note + "\n"

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
        if use_chat_template:
            retry_prompt = build_chat_eval_prompt(
                tokenizer,
                sample,
                dimension,
                clean_partial_response,
                force_final=True,
                final_only=args.final_only_json,
                parse_error=final_review_parse,
                step_context_mode=args.step_context_mode,
                max_problem_chars=args.max_problem_chars,
                max_code_chars=args.max_code_chars,
                mark_code_truncation_inside_block=args.mark_code_truncation_inside_block,
                show_tests_in_prompt=args.show_tests_in_prompt,
                enable_thinking=chat_template_enable_thinking,
                prompt_variant=args.prompt_variant,
            )
        else:
            retry_prompt = prompt_for_dimension(
                sample,
                dimension,
                clean_partial_response,
                force_final=True,
                parse_error=final_review_parse,
                final_only=args.final_only_json,
                step_context_mode=args.step_context_mode,
                max_problem_chars=args.max_problem_chars,
                max_code_chars=args.max_code_chars,
                mark_code_truncation_inside_block=args.mark_code_truncation_inside_block,
                show_tests_in_prompt=args.show_tests_in_prompt,
                prompt_variant=args.prompt_variant,
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
                chat_template_prompt=use_chat_template,
            )
            retry_value_score = score_response(value_model, tokenizer, retry_prompt, continuation) if value_model is not None else neutral_value_score()
        retry_artifacts = extract_reasoning_artifacts(continuation)
        merged_retry = merge_final_review_continuation(continuation, anchored=True)
        partial_response += merged_retry.strip() + "\n"
        final_review_parse = parse_final_review(partial_response)
        final_retries.append(
            {
                "retry_index": retry_index,
                "continuation": continuation,
                "content": retry_artifacts["content"],
                "reasoning": retry_artifacts["reasoning"],
                "reasoning_source": retry_artifacts["reasoning_source"],
                "value_score": retry_value_score,
                "final_review_parse": final_review_parse,
            }
        )

    if use_chat_template:
        final_prompt = build_chat_eval_prompt(
            tokenizer,
            sample,
            dimension,
            "",
            force_final=True,
            final_only=args.final_only_json,
            step_context_mode=args.step_context_mode,
            max_problem_chars=args.max_problem_chars,
            max_code_chars=args.max_code_chars,
            mark_code_truncation_inside_block=args.mark_code_truncation_inside_block,
            show_tests_in_prompt=args.show_tests_in_prompt,
            enable_thinking=chat_template_enable_thinking,
            prompt_variant=args.prompt_variant,
        )
    else:
        final_prompt = prompt_for_dimension(
            sample,
            dimension,
            "",
            force_final=True,
            final_only=args.final_only_json,
            step_context_mode=args.step_context_mode,
            max_problem_chars=args.max_problem_chars,
            max_code_chars=args.max_code_chars,
            mark_code_truncation_inside_block=args.mark_code_truncation_inside_block,
            show_tests_in_prompt=args.show_tests_in_prompt,
            prompt_variant=args.prompt_variant,
        )
    final_value_score = score_response(value_model, tokenizer, final_prompt, partial_response) if value_model is not None else neutral_value_score()
    reference_score = sample.get("axiom_target_score")
    reference_grade = sample.get("axiom_target_grade")
    reference_interval = sample.get("axiom_target_interval")
    parsed_score = parsed_review_score(final_review_parse)
    parsed_grade = parsed_review_grade(final_review_parse)
    lenient_grade = lenient_axiom_grade(partial_response)
    parsed_score_delta = score_delta(parsed_score, reference_score)
    accepted_reasoning_segments = [
        str(step.get("candidates", [])[step.get("selected_candidate_index", 0)].get("reasoning", "")).strip()
        for step in trace
        if step.get("accepted") and step.get("candidates")
    ]
    accepted_reasoning_segments = [segment for segment in accepted_reasoning_segments if segment]
    retry_reasoning_segments = [str(retry.get("reasoning", "")).strip() for retry in final_retries if retry.get("reasoning")]
    final_reasoning = "\n\n".join([*accepted_reasoning_segments, *retry_reasoning_segments]).strip()
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
        "final_reasoning": final_reasoning,
        "final_reasoning_source": "trace_compose" if final_reasoning else "none",
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
    parser.add_argument("--final_max_new_tokens", type=int, default=1536, help="Maximum new tokens for final review generation and retries.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--score_key", choices=VALUE_SCORE_KEYS, default="last_value")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--final_only_json", action="store_true", help="Generate only one compact final <review> JSON block; no step reasoning.")
    parser.add_argument(
        "--prompt_variant",
        choices=["default", "base_static", "codecritic_correctness"],
        default="default",
        help="Prompt contract for evaluation. default preserves Direct/MCTS prompts; base_static is a direct final-only control prompt; codecritic_correctness uses CodeCriticBench 1-10 correctness scoring.",
    )
    parser.add_argument(
        "--step_context_mode",
        choices=["assistant_prefix", "instruction_context"],
        default="instruction_context",
        help="How prior reasoning notes are exposed during stepwise reasoning. instruction_context avoids assistant-prefix continuation.",
    )
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
