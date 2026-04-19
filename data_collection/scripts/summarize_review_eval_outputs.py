from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from magicoder.review_evaluator import load_record, sample_from_record
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
    record = load_record(result["input_record"], int(result["record_index"]))
    return sample_from_record(record)


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
            parsed_grade = dimension.get("parsed_axiom_grade")
            row = {
                "file": path.name,
                "record_index": result.get("record_index"),
                "dimension": dimension.get("dimension"),
                "valid": valid,
                "reference_grade": reference_grade,
                "parsed_grade": parsed_grade,
            }
            if valid and isinstance(reference_grade, (int, float)) and isinstance(parsed_grade, (int, float)):
                row["abs_grade_error"] = abs(float(parsed_grade) - float(reference_grade))
                row["boundary_correct"] = (float(parsed_grade) >= 3) == (float(reference_grade) >= 3)
                row["low_grade_false_positive"] = float(reference_grade) >= 3 and float(parsed_grade) < 3
                row["high_grade_false_negative"] = float(reference_grade) < 3 and float(parsed_grade) >= 3
            if valid:
                row.update(evidence_stats(parsed, sample))
            rows.append(row)

    valid_rows = [row for row in rows if row["valid"]]
    grade_rows = [row for row in valid_rows if "abs_grade_error" in row]
    unsupported_rows = [row for row in valid_rows if row.get("unsupported_count", 0) > 0]
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
        "rows": rows,
    }
    return summary


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
