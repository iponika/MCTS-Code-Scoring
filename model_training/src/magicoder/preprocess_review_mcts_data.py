from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from magicoder.axiom_scoring import (
    AXIOM_GRADE_DESCRIPTIONS,
    axiom_functionally_correct,
    axiom_grade_from_scalar,
    axiom_scalar_score,
    axiom_verdict,
    clamp_axiom_grade,
    parse_axiom_grade,
)


IGNORED_INDEX = -100


def iter_input_files(paths: list[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(item for item in path.rglob("*.json") if item.is_file())
            yield from sorted(item for item in path.rglob("*.jsonl") if item.is_file())
        elif path.is_file():
            yield path


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if isinstance(record, dict) and "react" in record:
                        yield record
        return

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "react" in payload:
        yield payload
    elif isinstance(payload, list):
        for record in payload:
            if isinstance(record, dict) and "react" in record:
                yield record


def node_lineage(tag: str) -> list[str]:
    lineage = []
    current = ""
    for part in tag.split("."):
        current = f"{current}.{part}" if current else part
        lineage.append(current)
    return lineage


def normalize_response_segment(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    if "<review>" in cleaned:
        review_body = cleaned.split("<review>", 1)[1]
        if "</review>" in review_body:
            review_body = review_body.split("</review>", 1)[0]
        return f"<review>\n{review_body.strip()}\n</review>"

    if cleaned.startswith("<step>"):
        if cleaned.endswith("</step>"):
            return cleaned
        return f"{cleaned}\n</step>"

    return f"<step>\n{cleaned}\n</step>"


def truncate_for_review(value: object, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n... [truncated]"


def build_instruction(record: dict[str, Any], dimension: str) -> str:
    tests = record.get("tests") or []
    if tests:
        tests_text = "\n".join(str(test) for test in tests[:5])
        if len(tests) > 5:
            tests_text += f"\n... ({len(tests) - 5} more assertions omitted)"
    else:
        tests_text = "No tests are available."
    language = str(record.get("language") or record.get("lang") or "python").strip() or "python"

    return (
        f"Target review dimension: {dimension}\n\n"
        "Scoring target: assign the overall AXIOM code grade; use the target dimension as supporting evidence.\n\n"
        f"Task description:\n{truncate_for_review(record.get('problem') or record.get('question') or '', 5000)}\n\n"
        "Candidate code:\n"
        f"```{language}\n"
        f"{truncate_for_review(record.get('candidate_code') or '', 6000)}\n"
        "```\n\n"
        f"Available tests:\n{tests_text}\n\n"
        "Review only the target dimension. Use concrete evidence from the task, code, and tests."
    )


def terminal_error(node: dict[str, Any]) -> str | None:
    details_text = node.get("reward_details")
    if not details_text:
        return None
    try:
        details = json.loads(details_text)
    except json.JSONDecodeError:
        return "invalid_reward_details"
    return details.get("error")


def terminal_reward_details(node: dict[str, Any]) -> dict[str, Any]:
    details_text = node.get("reward_details")
    if not details_text:
        return {}
    try:
        details = json.loads(details_text)
    except json.JSONDecodeError:
        return {"error": "invalid_reward_details"}
    return details if isinstance(details, dict) else {}


def repair_effort_for_grade(grade: int) -> str:
    grade = clamp_axiom_grade(grade)
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


def verifier_issue_messages(details: dict[str, Any]) -> list[str]:
    evidence_details = details.get("evidence_details") if isinstance(details.get("evidence_details"), dict) else {}
    fact_checks = evidence_details.get("fact_checks") if isinstance(evidence_details.get("fact_checks"), list) else []
    messages: list[str] = []
    seen: set[str] = set()

    for check in fact_checks:
        if not isinstance(check, dict) or check.get("supported", True):
            continue
        kind = str(check.get("kind") or "fact_check")
        claim = str(check.get("claim") or "").strip()
        actual = check.get("actual") if isinstance(check.get("actual"), dict) else {}
        if kind == "unused_identifier":
            identifier = actual.get("identifier")
            usage = actual.get("usage") if isinstance(actual.get("usage"), dict) else {}
            message = (
                f"Unsupported unused-identifier claim: {claim!r}. "
                f"AST usage for {identifier!r}: load={usage.get('load', 0)}, "
                f"store={usage.get('store', 0)}, param={usage.get('param', 0)}."
            )
        elif kind == "provided_test_failure":
            reason = actual.get("reason")
            pass_rate = actual.get("full_test_pass_rate")
            if reason:
                message = f"Unsupported provided-test-failure evidence: {reason}."
            elif pass_rate is not None:
                message = f"Unsupported provided-test-failure evidence: full_test_pass_rate={pass_rate}."
            else:
                message = "Unsupported provided-test-failure evidence: no failing listed test was verified."
        else:
            actual_status = actual.get("status")
            if actual_status == "result":
                message = f"Unsupported executable claim: {claim!r}; actual result was {actual.get('value')!r}."
            elif actual_status == "exception":
                message = f"Unsupported executable claim: {claim!r}; actual exception was {actual.get('exception')!r}."
            elif actual:
                message = f"Unsupported verifier claim ({kind}): {claim!r}; actual={actual!r}."
            else:
                message = f"Unsupported verifier claim ({kind}): {claim!r}."
        if message not in seen:
            seen.add(message)
            messages.append(message)

    for cap in details.get("reward_caps") or []:
        if not isinstance(cap, dict):
            continue
        reason = str(cap.get("reason") or "")
        if not reason.startswith("unsupported_"):
            continue
        message = f"Reward cap triggered by verifier: {reason}."
        if message not in seen:
            seen.add(message)
            messages.append(message)
    return messages


def build_verifier_correction_instruction(
    record: dict[str, Any],
    dimension: str,
    terminal_text: str,
    messages: list[str],
) -> str:
    feedback = "\n".join(f"- {message}" for message in messages)
    return (
        build_instruction(record, dimension)
        + "\n\nA previous review for this same sample was rejected by deterministic verifier checks.\n"
        "Use the verifier feedback to repair the reasoning. Do not repeat unsupported evidence; "
        "if the failed claim was central, re-evaluate the AXIOM grade from the task, code, and verified tests.\n\n"
        f"Previous rejected review:\n{terminal_text.strip()}\n\n"
        f"Verifier feedback:\n{feedback}"
    )


def build_verifier_correction_response(details: dict[str, Any], dimension: str, messages: list[str]) -> list[str] | None:
    target_grade = target_grade_from_details(details)
    if target_grade is None:
        return None
    target_grade = clamp_axiom_grade(target_grade)

    feedback_summary = "; ".join(messages[:3])
    step = (
        "<step>\n"
        "Verifier correction: the previous review used unsupported evidence. "
        f"{feedback_summary} "
        "I will discard that claim and anchor the score to the verified AXIOM correctness boundary.\n"
        "</step>"
    )
    payload = {
        "dimension": dimension,
        "axiom_grade": target_grade,
        "score": axiom_scalar_score(target_grade),
        "verdict": axiom_verdict(target_grade),
        "functional_correctness": axiom_functionally_correct(target_grade),
        "repair_effort": repair_effort_for_grade(target_grade),
        "evidence_type": "verifier_corrected",
        "summary": (
            f"Corrected after verifier rejected unsupported evidence. "
            f"Use AXIOM grade {target_grade}: {AXIOM_GRADE_DESCRIPTIONS[target_grade]}"
        ),
        "evidence": [
            "Unsupported evidence from the previous review was removed.",
            f"The corrected score follows AXIOM grade {target_grade} after removing unsupported evidence.",
        ],
    }
    review = "<review>\n" + json.dumps(payload, ensure_ascii=False) + "\n</review>"
    return [step, review]


def target_grade_from_details(details: dict[str, Any]) -> int | None:
    target_grade = details.get("target_axiom_grade")
    if isinstance(target_grade, (int, float)):
        return clamp_axiom_grade(target_grade)
    target_score = details.get("target_score")
    max_score = 100.0 if isinstance(target_score, (int, float)) and float(target_score) > 10.0 else 10.0
    return axiom_grade_from_scalar(target_score, max_score=max_score)


def verifier_correction_training_item(
    record: dict[str, Any],
    tag: str,
    q_value: float,
    lm_loss_weight: float,
    value_loss_weight: float,
    mode: str,
) -> dict[str, Any] | None:
    terminal = record.get("react", {}).get(tag)
    if not isinstance(terminal, dict):
        return None
    dimension = terminal.get("target_dimension") or ""
    if not dimension:
        return None
    details = terminal_reward_details(terminal)
    messages = verifier_issue_messages(details)
    if not messages:
        return None
    response = build_verifier_correction_response(details, dimension, messages)
    if not response:
        return None
    target_grade = target_grade_from_details(details)
    if target_grade is None:
        return None
    q_value = clamp_reward(q_value)
    train_lm = mode in {"policy", "paired_repair"}
    synthetic_type = "verifier_correction" if mode == "policy" else f"verifier_correction_{mode}"
    terminal_suffix = f"verifier_correction_{mode}"
    return {
        "instruction": build_verifier_correction_instruction(record, dimension, str(terminal.get("text") or ""), messages),
        "response": response,
        "q_value": [q_value for _ in response],
        "train_lm": train_lm,
        "dataset_index": record.get("dataset_index"),
        "source": record.get("source"),
        "subset": record.get("subset"),
        "target_dimension": dimension,
        "terminal_tag": f"{tag}#{terminal_suffix}",
        "terminal_q_value": q_value,
        "terminal_error": None,
        "parsed_score": axiom_scalar_score(target_grade),
        "parsed_axiom_grade": target_grade,
        "target_score": details.get("target_score"),
        "target_axiom_grade": details.get("target_axiom_grade"),
        "is_best_path": train_lm,
        "synthetic_type": synthetic_type,
        "verifier_feedback": messages,
        "value_loss_weight": value_loss_weight,
        "lm_loss_weight": lm_loss_weight if train_lm else 0.0,
        "force_value_only": not train_lm,
        "allow_verifier_policy": train_lm,
    }


def path_to_training_item(
    record: dict[str, Any],
    tag: str,
    train_lm: bool,
) -> dict[str, Any] | None:
    react = record["react"]
    terminal = react.get(tag)
    if not isinstance(terminal, dict):
        return None
    dimension = terminal.get("target_dimension") or ""
    if not dimension:
        return None

    responses: list[str] = []
    q_values: list[float] = []
    for node_tag in node_lineage(tag):
        node = react.get(node_tag)
        if not isinstance(node, dict):
            continue
        segment = normalize_response_segment(str(node.get("text") or ""))
        if not segment:
            continue
        responses.append(segment)
        q_values.append(float(node.get("q_value", IGNORED_INDEX)))

    if not responses:
        return None

    details = terminal_reward_details(terminal)
    verifier_feedback = verifier_issue_messages(details)
    parsed = details.get("parsed") if isinstance(details.get("parsed"), dict) else {}
    parsed_axiom_grade = details.get("predicted_axiom_grade")
    if parsed_axiom_grade is None:
        parsed_axiom_grade = parse_axiom_grade(parsed)

    item = {
        "instruction": build_instruction(record, dimension),
        "response": responses,
        "q_value": q_values,
        "train_lm": train_lm,
        "dataset_index": record.get("dataset_index"),
        "source": record.get("source"),
        "subset": record.get("subset"),
        "target_dimension": dimension,
        "terminal_tag": tag,
        "terminal_q_value": float(terminal.get("q_value", IGNORED_INDEX)),
        "terminal_error": details.get("error"),
        "parsed_score": parsed.get("score"),
        "parsed_axiom_grade": parsed_axiom_grade,
        "target_score": details.get("target_score"),
        "target_axiom_grade": details.get("target_axiom_grade"),
    }
    if verifier_feedback:
        item["verifier_feedback"] = verifier_feedback
        item["force_value_only"] = True
        item["lm_loss_weight"] = 0.0
    return item


def collect_terminal_tags(record: dict[str, Any]) -> list[str]:
    tags = []
    for tag, node in record.get("react", {}).items():
        if isinstance(node, dict) and node.get("final_answer"):
            tags.append(tag)
    return tags


def best_tags(record: dict[str, Any]) -> set[str]:
    tags = set()
    for item in (record.get("best_reviews_by_dimension") or {}).values():
        tag = item.get("tag") if isinstance(item, dict) else None
        if tag:
            tags.add(tag)
    return tags


def clamp_reward(value: float) -> float:
    return max(-1.0, min(1.0, value))


def numeric_q_value(value: Any, default: float = IGNORED_INDEX) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def convert_records(
    records: Iterable[dict[str, Any]],
    policy_min_q: float,
    max_value_paths_per_dimension: int,
    data_split: str = "main",
    emit_verifier_corrections: bool = False,
    verifier_correction_mode: str = "value_only",
    verifier_correction_q: float = 0.8,
    verifier_correction_lm_weight: float = 0.5,
    verifier_correction_value_weight: float = 0.5,
    verifier_correction_repeat: int = 1,
    max_verifier_corrections: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    items: list[dict[str, Any]] = []
    stats = defaultdict(int)
    seen: set[tuple[Any, str]] = set()
    verifier_corrections = 0

    for record in records:
        stats["records"] += 1
        best = best_tags(record)
        per_dimension_counts: dict[str, int] = defaultdict(int)
        for tag in collect_terminal_tags(record):
            node = record["react"][tag]
            dimension = node.get("target_dimension") or ""
            if max_value_paths_per_dimension > 0 and per_dimension_counts[dimension] >= max_value_paths_per_dimension:
                continue
            q_value = float(node.get("q_value", IGNORED_INDEX))
            error = terminal_error(node)
            details = terminal_reward_details(node)
            has_verifier_issues = bool(verifier_issue_messages(details))
            train_lm = tag in best and q_value >= policy_min_q and error is None and not has_verifier_issues
            item = path_to_training_item(record, tag, train_lm=train_lm)
            if item is None:
                stats["skipped_paths"] += 1
                continue
            item["is_best_path"] = tag in best
            item["data_split"] = data_split
            dedupe_key = (record.get("source"), record.get("subset"), record.get("dataset_index"), tag)
            if dedupe_key in seen:
                stats["duplicate_paths"] += 1
                continue
            seen.add(dedupe_key)
            items.append(item)
            per_dimension_counts[dimension] += 1
            stats["paths"] += 1
            stats["policy_paths" if train_lm else "value_only_paths"] += 1
            if error:
                stats[f"error_{error}"] += 1

            if emit_verifier_corrections and (max_verifier_corrections <= 0 or verifier_corrections < max_verifier_corrections):
                correction = verifier_correction_training_item(
                    record,
                    tag,
                    q_value=verifier_correction_q,
                    lm_loss_weight=verifier_correction_lm_weight,
                    value_loss_weight=verifier_correction_value_weight,
                    mode=verifier_correction_mode,
                )
                if correction is not None:
                    correction["data_split"] = data_split
                    repeat_count = max(1, verifier_correction_repeat)
                    base_terminal_tag = str(correction.get("terminal_tag") or "")
                    for repeat_index in range(repeat_count):
                        repeated = correction if repeat_index == 0 else copy.deepcopy(correction)
                        if repeat_count > 1:
                            repeated["terminal_tag"] = f"{base_terminal_tag}#repeat{repeat_index}"
                            repeated["synthetic_repeat_index"] = repeat_index
                            repeated["synthetic_repeat_count"] = repeat_count
                        correction_key = (
                            repeated.get("source"),
                            repeated.get("subset"),
                            repeated.get("dataset_index"),
                            repeated.get("terminal_tag"),
                        )
                        if correction_key in seen:
                            stats["duplicate_verifier_corrections"] += 1
                            continue
                        seen.add(correction_key)
                        items.append(repeated)
                        stats["paths"] += 1
                        if repeated.get("train_lm"):
                            stats["policy_paths"] += 1
                        else:
                            stats["value_only_paths"] += 1
                        stats["verifier_correction_paths"] += 1
                        stats[f"verifier_correction_mode_{verifier_correction_mode}"] += 1
                    verifier_corrections += 1

    return items, dict(stats)


def terminal_q_values_by_dimension(records: Iterable[dict[str, Any]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for node in record.get("react", {}).values():
            if not isinstance(node, dict) or not node.get("final_answer"):
                continue
            if terminal_error(node):
                continue
            dimension = node.get("target_dimension") or ""
            if not dimension:
                continue
            q_value = numeric_q_value(node.get("q_value"))
            if q_value == IGNORED_INDEX:
                continue
            values[dimension].append(q_value)
    return values


def build_q_calibrator(records: Iterable[dict[str, Any]], min_count: int) -> dict[str, Any]:
    by_dimension = terminal_q_values_by_dimension(records)
    all_values = [value for values in by_dimension.values() for value in values]
    if len(all_values) < max(2, min_count):
        return {"enabled": False, "reason": "not_enough_anchor_values", "count": len(all_values)}

    global_std = statistics.pstdev(all_values)
    calibrator = {
        "enabled": True,
        "min_count": min_count,
        "global": {
            "count": len(all_values),
            "mean": statistics.fmean(all_values),
            "std": global_std,
        },
        "dimensions": {},
    }
    for dimension, values in sorted(by_dimension.items()):
        if len(values) < min_count:
            continue
        std = statistics.pstdev(values)
        if std <= 1e-6 or global_std <= 1e-6:
            continue
        calibrator["dimensions"][dimension] = {
            "count": len(values),
            "mean": statistics.fmean(values),
            "std": std,
        }
    if not calibrator["dimensions"]:
        return {"enabled": False, "reason": "not_enough_dimension_values", "count": len(all_values)}
    return calibrator


def calibrate_q_value(q_value: float, dimension: str, calibrator: dict[str, Any], strength: float) -> float:
    if q_value == IGNORED_INDEX:
        return q_value
    dimension_stats = calibrator.get("dimensions", {}).get(dimension)
    global_stats = calibrator.get("global", {})
    if not dimension_stats or not global_stats:
        return q_value
    standardized = (q_value - dimension_stats["mean"]) / dimension_stats["std"]
    mapped = global_stats["mean"] + standardized * global_stats["std"]
    return round(clamp_reward((1.0 - strength) * q_value + strength * mapped), 4)


def apply_q_calibration(items: list[dict[str, Any]], calibrator: dict[str, Any], strength: float) -> dict[str, Any]:
    stats = defaultdict(int)
    if not calibrator.get("enabled"):
        stats["calibration_enabled"] = 0
        stats[f"calibration_skip_{calibrator.get('reason', 'unknown')}"] = 1
        return dict(stats)

    strength = max(0.0, min(1.0, strength))
    for item in items:
        dimension = item.get("target_dimension") or ""
        if not dimension or dimension not in calibrator.get("dimensions", {}):
            stats["calibration_skipped_missing_dimension"] += 1
            continue
        if item.get("terminal_error"):
            stats["calibration_skipped_terminal_error"] += 1
            continue

        raw_q_values = [numeric_q_value(value) for value in item.get("q_value", [])]
        calibrated = [calibrate_q_value(value, dimension, calibrator, strength) for value in raw_q_values]
        if calibrated == raw_q_values:
            stats["calibration_unchanged_items"] += 1
            continue

        item["raw_q_value"] = raw_q_values
        item["raw_terminal_q_value"] = numeric_q_value(item.get("terminal_q_value"))
        item["q_value"] = calibrated
        if calibrated:
            item["terminal_q_value"] = calibrated[-1]
        item["q_calibration"] = {
            "method": "dimension_standardize_v1",
            "strength": strength,
            "anchor_global_mean": round(calibrator["global"]["mean"], 6),
            "anchor_global_std": round(calibrator["global"]["std"], 6),
            "anchor_dimension_mean": round(calibrator["dimensions"][dimension]["mean"], 6),
            "anchor_dimension_std": round(calibrator["dimensions"][dimension]["std"], 6),
        }
        stats["calibrated_items"] += 1
    stats["calibration_enabled"] = 1
    stats["calibration_anchor_values"] = calibrator["global"]["count"]
    stats["calibration_dimensions"] = len(calibrator["dimensions"])
    return dict(stats)


def median_absolute_deviation(values: list[float], median_value: float) -> float:
    return statistics.median(abs(value - median_value) for value in values)


def build_score_consensus(records: Iterable[dict[str, Any]], min_valid: int) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        dataset_index = str(record.get("dataset_index"))
        for node in record.get("react", {}).values():
            if not isinstance(node, dict) or not node.get("final_answer"):
                continue
            dimension = node.get("target_dimension") or ""
            if not dimension:
                continue
            key = (dataset_index, dimension)
            group = grouped.setdefault(key, {"grades": [], "total": 0, "target_grade": None})
            group["total"] += 1
            details = terminal_reward_details(node)
            if group["target_grade"] is None:
                if isinstance(details.get("target_axiom_grade"), (int, float)):
                    group["target_grade"] = float(details["target_axiom_grade"])
                elif isinstance(details.get("target_score"), (int, float)):
                    max_score = 100.0 if float(details["target_score"]) > 10 else 10.0
                    group["target_grade"] = axiom_grade_from_scalar(details["target_score"], max_score=max_score)
            if details.get("error"):
                continue
            parsed = details.get("parsed") if isinstance(details.get("parsed"), dict) else {}
            parsed_grade = details.get("predicted_axiom_grade")
            if parsed_grade is None:
                parsed_grade = parse_axiom_grade(parsed)
            if parsed_grade is None:
                continue
            group["grades"].append(float(parsed_grade))

    consensus = {}
    for key, group in grouped.items():
        grades = group["grades"]
        target_grade = group.get("target_grade")
        if len(grades) < min_valid or not isinstance(target_grade, (int, float)):
            continue
        median_grade = statistics.median(grades)
        mad = median_absolute_deviation(grades, median_grade)
        valid_rate = len(grades) / max(1, int(group["total"]))
        dispersion_confidence = max(0.0, 1.0 - mad / 2.5)
        target_alignment = max(0.0, 1.0 - abs(median_grade - target_grade) / 5.0)
        consensus[key] = {
            "median_score": median_grade * 20.0,
            "median_axiom_grade": median_grade,
            "mad": mad,
            "valid_count": float(len(grades)),
            "total_count": float(group["total"]),
            "valid_rate": valid_rate,
            "target_score": float(target_grade) * 20.0,
            "target_axiom_grade": float(target_grade),
            "target_alignment": target_alignment,
            "dispersion_confidence": dispersion_confidence,
            "confidence": valid_rate * dispersion_confidence,
            "reward": target_alignment * 2.0 - 1.0,
        }
    return consensus


def apply_score_consensus(
    items: list[dict[str, Any]],
    consensus: dict[tuple[str, str], dict[str, float]],
    strength: float,
    mode: str,
) -> dict[str, int]:
    stats = defaultdict(int)
    strength = max(0.0, min(1.0, strength))
    if not consensus:
        stats["score_consensus_enabled"] = 0
        stats["score_consensus_skipped_empty"] = 1
        return dict(stats)

    for item in items:
        key = (str(item.get("dataset_index")), item.get("target_dimension") or "")
        summary = consensus.get(key)
        if not summary:
            stats["score_consensus_skipped_missing"] += 1
            continue

        item["score_consensus"] = {
            "median_score": round(summary["median_score"], 4),
            "median_axiom_grade": round(summary["median_axiom_grade"], 4),
            "mad": round(summary["mad"], 4),
            "valid_count": int(summary["valid_count"]),
            "total_count": int(summary["total_count"]),
            "valid_rate": round(summary["valid_rate"], 4),
            "target_score": round(summary["target_score"], 4),
            "target_axiom_grade": round(summary["target_axiom_grade"], 4),
            "target_alignment": round(summary["target_alignment"], 4),
            "dispersion_confidence": round(summary["dispersion_confidence"], 4),
            "confidence": round(summary["confidence"], 4),
            "reward": round(summary["reward"], 4),
            "strength": strength,
            "mode": mode,
        }

        if item.get("terminal_error"):
            stats["score_consensus_skipped_terminal_error"] += 1
            continue

        grade = item.get("parsed_axiom_grade")
        if grade is None:
            parsed_score = item.get("parsed_score")
            max_score = 100.0
            try:
                if float(parsed_score) <= 10.0:
                    max_score = 10.0
            except (TypeError, ValueError):
                pass
            grade = axiom_grade_from_scalar(parsed_score, max_score=max_score)
        if grade is None:
            item_alignment = 1.0
        else:
            item_alignment = max(0.0, 1.0 - abs(float(grade) - summary["median_axiom_grade"]) / 5.0)
        effective_strength = strength * summary["confidence"] * item_alignment
        item["score_consensus"]["item_alignment"] = round(item_alignment, 4)
        item["score_consensus"]["effective_strength"] = round(effective_strength, 4)

        if mode == "weight":
            sample_weight = round(max(0.0, 1.0 - strength + 2.0 * effective_strength), 4)
            item["value_loss_weight"] = sample_weight
            item["lm_loss_weight"] = sample_weight
            stats["score_consensus_weighted_items"] += 1
        elif mode == "q_adjust":
            if "raw_q_value" not in item:
                item["raw_q_value"] = [numeric_q_value(value) for value in item.get("q_value", [])]
                item["raw_terminal_q_value"] = numeric_q_value(item.get("terminal_q_value"))
            consensus_reward = summary["reward"]
            q_values = [numeric_q_value(value) for value in item.get("q_value", [])]
            adjusted = [
                value if value == IGNORED_INDEX else round(clamp_reward((1.0 - effective_strength) * value + effective_strength * consensus_reward), 4)
                for value in q_values
            ]
            if adjusted != q_values:
                item["pre_consensus_q_value"] = q_values
                item["q_value"] = adjusted
                if adjusted:
                    item["terminal_q_value"] = adjusted[-1]
                stats["score_consensus_adjusted_items"] += 1
            else:
                stats["score_consensus_unchanged_items"] += 1
        else:
            raise ValueError(f"Unsupported score consensus mode: {mode}")

    stats["score_consensus_enabled"] = 1
    stats["score_consensus_groups"] = len(consensus)
    stats["score_consensus_mode_weight" if mode == "weight" else "score_consensus_mode_q_adjust"] = 1
    return dict(stats)


def refresh_policy_flags(items: list[dict[str, Any]], policy_min_q: float) -> dict[str, int]:
    stats = defaultdict(int)
    for item in items:
        old_train_lm = bool(item.get("train_lm"))
        terminal_q = numeric_q_value(item.get("terminal_q_value"))
        new_train_lm = (
            bool(item.get("is_best_path"))
            and terminal_q >= policy_min_q
            and item.get("terminal_error") is None
            and not item.get("force_value_only")
            and (not item.get("verifier_feedback") or item.get("allow_verifier_policy"))
        )
        item["train_lm"] = new_train_lm
        if not new_train_lm and item.get("force_value_only"):
            item["lm_loss_weight"] = 0.0
        if old_train_lm != new_train_lm:
            stats["policy_flag_changed"] += 1
            stats["policy_flag_upgraded" if new_train_lm else "policy_flag_downgraded"] += 1
        stats["policy_paths" if new_train_lm else "value_only_paths"] += 1
    return dict(stats)


def dedupe_items(items: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    deduped = []
    seen: set[tuple[Any, str]] = set()
    duplicate_count = 0
    for item in items:
        key = (item.get("source"), item.get("subset"), item.get("dataset_index"), item.get("terminal_tag"))
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        deduped.append(item)
    return deduped, duplicate_count


def select_replay_items(items: list[dict[str, Any]], replay_ratio: float, new_item_count: int, seed: int) -> list[dict[str, Any]]:
    if replay_ratio < 0 or not items:
        return list(items)
    limit = int(round(replay_ratio * new_item_count))
    if limit >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert review MCTS runs into policy/value training JSONL.")
    parser.add_argument("--input", nargs="+", required=True, help="Review MCTS sample JSON/JSONL files or directories.")
    parser.add_argument(
        "--replay_input",
        nargs="*",
        default=[],
        help="Optional older review MCTS sample JSON/JSONL files or directories mixed into the output as replay data.",
    )
    parser.add_argument(
        "--anchor_input",
        nargs="*",
        default=[],
        help="Optional fixed anchor review MCTS samples used to calibrate q_values. Defaults to replay+new records.",
    )
    parser.add_argument("--output_file", required=True, help="Output JSONL consumed by magicoder.train_multi --task review.")
    parser.add_argument("--policy_min_q", type=float, default=0.5, help="Minimum terminal q_value for LM imitation.")
    parser.add_argument(
        "--max_value_paths_per_dimension",
        type=int,
        default=0,
        help="0 keeps all terminal paths; otherwise cap value-only paths per sample dimension.",
    )
    parser.add_argument(
        "--replay_ratio",
        type=float,
        default=1.0,
        help="Maximum replay items per new item. Set a negative value to keep all replay items.",
    )
    parser.add_argument(
        "--calibrate_q_values",
        action="store_true",
        help="Apply per-dimension anchor standardization to non-error q_values while preserving raw_q_value fields.",
    )
    parser.add_argument(
        "--calibration_strength",
        type=float,
        default=0.35,
        help="Blend strength for calibrated q_values: 0 keeps raw labels, 1 fully maps to the anchor global distribution.",
    )
    parser.add_argument(
        "--min_calibration_count",
        type=int,
        default=8,
        help="Minimum valid terminal leaves required per dimension before applying dimension calibration.",
    )
    parser.add_argument(
        "--apply_score_consensus",
        action="store_true",
        help="Apply per-sample/per-dimension median review score consensus as confidence metadata.",
    )
    parser.add_argument(
        "--score_consensus_mode",
        choices=["weight", "q_adjust"],
        default="weight",
        help="weight keeps q_values unchanged and emits loss weights; q_adjust preserves the older behavior that blends q_values toward consensus reward.",
    )
    parser.add_argument(
        "--score_consensus_strength",
        type=float,
        default=0.25,
        help="Maximum strength for score-consensus weighting or legacy q-value adjustment.",
    )
    parser.add_argument(
        "--score_consensus_min_valid",
        type=int,
        default=3,
        help="Minimum valid review scores needed for a sample-dimension consensus group.",
    )
    parser.add_argument(
        "--emit_verifier_corrections",
        action="store_true",
        help="Emit extra verifier-feedback items. Defaults to value-only supervision, not policy imitation.",
    )
    parser.add_argument(
        "--verifier_correction_mode",
        choices=["value_only", "policy", "paired_repair"],
        default="value_only",
        help=(
            "How emitted verifier corrections are trained. value_only gives value-head supervision with no LM loss; "
            "policy preserves the older behavior; paired_repair is reserved for explicit repair-process imitation."
        ),
    )
    parser.add_argument(
        "--verifier_correction_q",
        type=float,
        default=0.8,
        help="Target q_value assigned to synthetic verifier-correction responses.",
    )
    parser.add_argument(
        "--verifier_correction_lm_weight",
        type=float,
        default=0.5,
        help="LM loss weight for synthetic verifier-correction responses.",
    )
    parser.add_argument(
        "--verifier_correction_value_weight",
        type=float,
        default=0.5,
        help="Value loss weight for synthetic verifier-correction responses.",
    )
    parser.add_argument(
        "--verifier_correction_repeat",
        type=int,
        default=1,
        help="Repeat each synthetic verifier-correction item this many times with unique tags. Useful for short ablations.",
    )
    parser.add_argument(
        "--max_verifier_corrections",
        type=int,
        default=0,
        help="Maximum synthetic verifier-correction items per split. 0 means no cap.",
    )
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Seed for replay sampling and output shuffling.")
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle mixed output items before writing.",
    )
    args = parser.parse_args()

    input_files = list(iter_input_files([Path(path) for path in args.input]))
    new_records = [record for path in input_files for record in iter_records(path)]
    new_items, new_stats = convert_records(
        new_records,
        policy_min_q=args.policy_min_q,
        max_value_paths_per_dimension=args.max_value_paths_per_dimension,
        data_split="new",
        emit_verifier_corrections=args.emit_verifier_corrections,
        verifier_correction_mode=args.verifier_correction_mode,
        verifier_correction_q=args.verifier_correction_q,
        verifier_correction_lm_weight=args.verifier_correction_lm_weight,
        verifier_correction_value_weight=args.verifier_correction_value_weight,
        verifier_correction_repeat=args.verifier_correction_repeat,
        max_verifier_corrections=args.max_verifier_corrections,
    )
    stats = {f"new_{key}": value for key, value in new_stats.items()}

    replay_files = list(iter_input_files([Path(path) for path in args.replay_input]))
    replay_records = [record for path in replay_files for record in iter_records(path)]
    replay_items, replay_stats = convert_records(
        replay_records,
        policy_min_q=args.policy_min_q,
        max_value_paths_per_dimension=args.max_value_paths_per_dimension,
        data_split="replay",
        emit_verifier_corrections=args.emit_verifier_corrections,
        verifier_correction_mode=args.verifier_correction_mode,
        verifier_correction_q=args.verifier_correction_q,
        verifier_correction_lm_weight=args.verifier_correction_lm_weight,
        verifier_correction_value_weight=args.verifier_correction_value_weight,
        verifier_correction_repeat=args.verifier_correction_repeat,
        max_verifier_corrections=args.max_verifier_corrections,
    )
    stats.update({f"replay_{key}": value for key, value in replay_stats.items()})
    selected_replay_items = select_replay_items(
        replay_items,
        replay_ratio=args.replay_ratio,
        new_item_count=len(new_items),
        seed=args.shuffle_seed,
    )
    items, duplicate_count = dedupe_items([*new_items, *selected_replay_items])
    stats["cross_split_duplicate_items"] = duplicate_count
    stats["selected_replay_items"] = len(selected_replay_items)

    if args.calibrate_q_values:
        anchor_files = list(iter_input_files([Path(path) for path in args.anchor_input]))
        anchor_records = [record for path in anchor_files for record in iter_records(path)]
        if not anchor_records:
            anchor_records = [*replay_records, *new_records]
        calibrator = build_q_calibrator(anchor_records, min_count=args.min_calibration_count)
        stats.update(apply_q_calibration(items, calibrator, strength=args.calibration_strength))
    else:
        stats["calibration_enabled"] = 0
    if args.apply_score_consensus:
        consensus = build_score_consensus([*new_records, *replay_records], min_valid=args.score_consensus_min_valid)
        stats.update(apply_score_consensus(items, consensus, strength=args.score_consensus_strength, mode=args.score_consensus_mode))
    else:
        stats["score_consensus_enabled"] = 0
    stats.update({f"final_{key}": value for key, value in refresh_policy_flags(items, args.policy_min_q).items()})

    if args.shuffle:
        random.Random(args.shuffle_seed).shuffle(items)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats["input_files"] = len(input_files)
    stats["replay_input_files"] = len(replay_files)
    stats["output_items"] = len(items)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
