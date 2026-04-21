from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from magicoder.review_evaluator import lenient_axiom_grade, load_record, sample_from_record
from mcts_math.review_utils import validate_review_evidence


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.median(values)), 6)


def sample_for_result(result: dict[str, Any]) -> dict[str, Any]:
    input_record = result.get("input_record")
    record_index = result.get("record_index")
    if input_record is not None and record_index is not None:
        try:
            record = load_record(str(input_record), int(record_index))
            return sample_from_record(record)
        except Exception:
            pass
    if "candidate_code" in result:
        return sample_from_record(result)
    return {
        "problem": result.get("problem", ""),
        "candidate_code": result.get("candidate_code", ""),
        "tests": result.get("tests") or [],
    }


def evidence_stats(parsed: dict[str, Any], sample: dict[str, Any]) -> dict[str, int]:
    _, details = validate_review_evidence(parsed, sample)
    fact_checks = details.get("fact_checks") or []
    unsupported = [check for check in fact_checks if isinstance(check, dict) and not check.get("supported", True)]
    return {
        "fact_check_count": len(fact_checks),
        "unsupported_count": len(unsupported),
        "unsupported_provided_test_failure": sum(1 for check in unsupported if check.get("kind") == "provided_test_failure"),
        "unsupported_unused_identifier": sum(1 for check in unsupported if check.get("kind") == "unused_identifier"),
    }


def summarize(eval_dir: Path) -> dict[str, Any]:
    files = sorted(eval_dir.glob("*.json"))
    rows: list[dict[str, Any]] = []
    for path in files:
        result = json.loads(path.read_text(encoding="utf-8"))
        sample = sample_for_result(result)
        for dimension in result.get("dimensions") or []:
            parse = dimension.get("final_review_parse") or {}
            parsed = parse.get("parsed") if isinstance(parse.get("parsed"), dict) else {}
            valid = bool(parse.get("ok"))
            reference_grade = dimension.get("reference_axiom_grade")
            reference_interval = dimension.get("reference_axiom_interval")
            parsed_grade = dimension.get("parsed_axiom_grade")
            lenient_grade = dimension.get("lenient_axiom_grade")
            if lenient_grade is None:
                text = str(dimension.get("final_review") or "")
                for trace_item in dimension.get("trace") or []:
                    for candidate in trace_item.get("candidates") or []:
                        text += "\n" + str(candidate.get("continuation") or "")
                lenient_grade = lenient_axiom_grade(text)
            row = {
                "file": path.name,
                "record_index": result.get("record_index"),
                "dimension": dimension.get("dimension"),
                "source": result.get("source"),
                "subset": result.get("subset"),
                "dataset_index": result.get("dataset_index"),
                "label_type": dimension.get("label_type"),
                "pair_id": dimension.get("pair_id"),
                "pair_role": dimension.get("pair_role"),
                "valid": valid,
                "reference_grade": reference_grade,
                "reference_interval": reference_interval,
                "parsed_grade": parsed_grade,
                "lenient_grade": lenient_grade,
            }
            if valid and isinstance(reference_grade, (int, float)) and isinstance(parsed_grade, (int, float)):
                row["abs_grade_error"] = abs(float(parsed_grade) - float(reference_grade))
                row["boundary_correct"] = (float(parsed_grade) >= 3) == (float(reference_grade) >= 3)
                row["low_grade_false_positive"] = float(reference_grade) >= 3 and float(parsed_grade) < 3
                row["high_grade_false_negative"] = float(reference_grade) < 3 and float(parsed_grade) >= 3
            if isinstance(reference_grade, (int, float)) and isinstance(lenient_grade, (int, float)):
                row["lenient_abs_grade_error"] = abs(float(lenient_grade) - float(reference_grade))
                row["lenient_boundary_correct"] = (float(lenient_grade) >= 3) == (float(reference_grade) >= 3)
                row["lenient_low_grade_false_positive"] = float(reference_grade) >= 3 and float(lenient_grade) < 3
                row["lenient_high_grade_false_negative"] = float(reference_grade) < 3 and float(lenient_grade) >= 3
            if isinstance(reference_interval, list) and len(reference_interval) == 2 and isinstance(parsed_grade, (int, float)):
                lower, upper = sorted(float(item) for item in reference_interval)
                row["interval_correct"] = lower <= float(parsed_grade) <= upper
                row["interval_boundary_correct"] = (float(parsed_grade) >= 3) == (upper >= 3)
            if isinstance(reference_interval, list) and len(reference_interval) == 2 and isinstance(lenient_grade, (int, float)):
                lower, upper = sorted(float(item) for item in reference_interval)
                row["lenient_interval_correct"] = lower <= float(lenient_grade) <= upper
                row["lenient_interval_boundary_correct"] = (float(lenient_grade) >= 3) == (upper >= 3)
            if valid:
                row.update(evidence_stats(parsed, sample))
            rows.append(row)

    valid_rows = [row for row in rows if row["valid"]]
    grade_rows = [row for row in valid_rows if "abs_grade_error" in row]
    lenient_rows = [row for row in rows if "lenient_abs_grade_error" in row]
    interval_rows = [row for row in valid_rows if "interval_correct" in row]
    lenient_interval_rows = [row for row in rows if "lenient_interval_correct" in row]
    unsupported_rows = [row for row in valid_rows if row.get("unsupported_count", 0) > 0]
    pairwise = pairwise_stats(rows, grade_key="parsed_grade")
    lenient_pairwise = pairwise_stats(rows, grade_key="lenient_grade")
    summary = {
        "eval_dir": str(eval_dir),
        "files": len(files),
        "dimension_count": len(rows),
        "valid_count": len(valid_rows),
        "valid_rate": round(len(valid_rows) / max(1, len(rows)), 6),
        "grade_mae": mean([row["abs_grade_error"] for row in grade_rows]),
        "grade_median_abs_error": median([row["abs_grade_error"] for row in grade_rows]),
        "boundary_acc": mean([1.0 if row["boundary_correct"] else 0.0 for row in grade_rows]),
        "low_grade_false_positive_rate": mean([1.0 if row.get("low_grade_false_positive") else 0.0 for row in grade_rows]),
        "high_grade_false_negative_rate": mean([1.0 if row.get("high_grade_false_negative") else 0.0 for row in grade_rows]),
        "unsupported_evidence_rate": round(len(unsupported_rows) / max(1, len(valid_rows)), 6),
        "unsupported_evidence_count": sum(int(row.get("unsupported_count", 0)) for row in valid_rows),
        "unsupported_provided_test_failure": sum(int(row.get("unsupported_provided_test_failure", 0)) for row in valid_rows),
        "unsupported_unused_identifier": sum(int(row.get("unsupported_unused_identifier", 0)) for row in valid_rows),
        "lenient_count": len(lenient_rows),
        "lenient_rate": round(len(lenient_rows) / max(1, len(rows)), 6),
        "lenient_grade_mae": mean([row["lenient_abs_grade_error"] for row in lenient_rows]),
        "lenient_grade_median_abs_error": median([row["lenient_abs_grade_error"] for row in lenient_rows]),
        "lenient_boundary_acc": mean([1.0 if row["lenient_boundary_correct"] else 0.0 for row in lenient_rows]),
        "lenient_low_grade_false_positive_rate": mean([1.0 if row.get("lenient_low_grade_false_positive") else 0.0 for row in lenient_rows]),
        "lenient_high_grade_false_negative_rate": mean([1.0 if row.get("lenient_high_grade_false_negative") else 0.0 for row in lenient_rows]),
        "interval_acc": mean([1.0 if row["interval_correct"] else 0.0 for row in interval_rows]),
        "interval_boundary_acc": mean([1.0 if row["interval_boundary_correct"] else 0.0 for row in interval_rows]),
        "lenient_interval_acc": mean([1.0 if row["lenient_interval_correct"] else 0.0 for row in lenient_interval_rows]),
        "lenient_interval_boundary_acc": mean([1.0 if row["lenient_interval_boundary_correct"] else 0.0 for row in lenient_interval_rows]),
        "pairwise_acc": pairwise["pairwise_acc"],
        "pairwise_count": pairwise["pairwise_count"],
        "lenient_pairwise_acc": lenient_pairwise["pairwise_acc"],
        "lenient_pairwise_count": lenient_pairwise["pairwise_count"],
        "rows": rows,
    }
    return summary


def pairwise_stats(rows: list[dict[str, Any]], grade_key: str) -> dict[str, Any]:
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        pair_id = row.get("pair_id")
        role = row.get("pair_role")
        grade = row.get(grade_key)
        if not pair_id or role not in {"pos", "neg"} or not isinstance(grade, (int, float)):
            continue
        grouped.setdefault(str(pair_id), {})[str(role)] = float(grade)
    comparable = [pair for pair in grouped.values() if "pos" in pair and "neg" in pair]
    if not comparable:
        return {"pairwise_acc": None, "pairwise_count": 0}
    correct = sum(1 for pair in comparable if pair["pos"] > pair["neg"])
    return {"pairwise_acc": round(correct / len(comparable), 6), "pairwise_count": len(comparable)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize review_evaluator JSON outputs with verifier evidence checks.")
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    summary = summarize(Path(args.eval_dir))
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
