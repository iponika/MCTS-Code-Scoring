from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from mcts_math.axiom_scoring import axiom_functionally_correct, clamp_axiom_grade


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize direct/MCTS AXIOM score predictions.")
    parser.add_argument("--direct", type=Path, default=None, help="Direct API baseline JSONL.")
    parser.add_argument("--mcts", type=Path, default=None, help="Review MCTS aggregate JSONL.")
    parser.add_argument("--output", type=Path, default=None, help="Optional summary JSON output.")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record.get("predicted_axiom_grade") is not None]
    errors = [abs(float(record["predicted_axiom_grade"]) - float(record["target_axiom_grade"])) for record in valid]
    boundary = [
        axiom_functionally_correct(record["predicted_axiom_grade"])
        == axiom_functionally_correct(record["target_axiom_grade"])
        for record in valid
    ]
    summary = {
        "count": len(records),
        "valid_count": len(valid),
        "valid_rate": round(len(valid) / max(1, len(records)), 4),
        "mae_grade": round(statistics.fmean(errors), 4) if errors else None,
        "median_abs_error": round(statistics.median(errors), 4) if errors else None,
        "boundary_accuracy": round(sum(boundary) / max(1, len(boundary)), 4) if boundary else None,
        "target_grade_dist": dict(sorted(Counter(record.get("target_axiom_grade") for record in records).items())),
        "pred_grade_dist": dict(sorted(Counter(record.get("predicted_axiom_grade") for record in valid).items())),
    }
    if len(valid) >= 2:
        target = [float(record["target_axiom_grade"]) for record in valid]
        pred = [float(record["predicted_axiom_grade"]) for record in valid]
        summary["pearson_proxy"] = round(correlation(target, pred), 4)
    return summary


def correlation(xs: list[float], ys: list[float]) -> float:
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    denom_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if denom_x <= 1e-9 or denom_y <= 1e-9:
        return 0.0
    return numerator / (denom_x * denom_y)


def direct_records(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def mcts_record_prediction(record: dict[str, Any]) -> dict[str, Any]:
    grades = []
    chosen = []
    for dimension, item in (record.get("best_reviews_by_dimension") or {}).items():
        try:
            details = json.loads(item.get("reward_details") or "{}")
        except json.JSONDecodeError:
            details = {}
        grade = details.get("predicted_axiom_grade")
        if isinstance(grade, (int, float)):
            grades.append(float(grade))
            chosen.append(
                {
                    "dimension": dimension,
                    "tag": item.get("tag"),
                    "q_value": item.get("q_value"),
                    "predicted_axiom_grade": grade,
                    "target_axiom_grade": details.get("target_axiom_grade"),
                    "reward_details": details,
                }
            )
    predicted_grade = clamp_axiom_grade(statistics.median(grades)) if grades else None
    target_grade = record.get("axiom_target_grade")
    return {
        "dataset_index": record.get("dataset_index"),
        "source": record.get("source"),
        "subset": record.get("subset"),
        "target_axiom_grade": target_grade,
        "predicted_axiom_grade": predicted_grade,
        "grade_abs_error": abs(predicted_grade - target_grade) if predicted_grade is not None and target_grade is not None else None,
        "boundary_correct": (
            axiom_functionally_correct(predicted_grade) == axiom_functionally_correct(target_grade)
            if predicted_grade is not None and target_grade is not None
            else False
        ),
        "dimension_predictions": chosen,
    }


def mcts_records(path: Path) -> list[dict[str, Any]]:
    return [mcts_record_prediction(record) for record in iter_jsonl(path)]


def main() -> None:
    args = parse_args()
    output: dict[str, Any] = {}
    if args.direct:
        direct = direct_records(args.direct)
        output["direct"] = summarize_records(direct)
    if args.mcts:
        mcts = mcts_records(args.mcts)
        output["mcts"] = summarize_records(mcts)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
