from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


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


def build_instruction(record: dict[str, Any], dimension: str) -> str:
    tests = record.get("tests") or []
    if tests:
        tests_text = "\n".join(str(test) for test in tests[:5])
        if len(tests) > 5:
            tests_text += f"\n... ({len(tests) - 5} more assertions omitted)"
    else:
        tests_text = "No tests are available."

    return (
        f"Target review dimension: {dimension}\n\n"
        f"Task description:\n{record.get('problem') or record.get('question') or ''}\n\n"
        "Candidate code:\n"
        "```python\n"
        f"{record.get('candidate_code') or ''}\n"
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

    return {
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
        "terminal_error": terminal_error(terminal),
    }


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
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    items: list[dict[str, Any]] = []
    stats = defaultdict(int)
    seen: set[tuple[Any, str]] = set()

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
            train_lm = tag in best and q_value >= policy_min_q and error is None
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


def refresh_policy_flags(items: list[dict[str, Any]], policy_min_q: float) -> dict[str, int]:
    stats = defaultdict(int)
    for item in items:
        old_train_lm = bool(item.get("train_lm"))
        terminal_q = numeric_q_value(item.get("terminal_q_value"))
        new_train_lm = bool(item.get("is_best_path")) and terminal_q >= policy_min_q and item.get("terminal_error") is None
        item["train_lm"] = new_train_lm
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
    )
    stats = {f"new_{key}": value for key, value in new_stats.items()}

    replay_files = list(iter_input_files([Path(path) for path in args.replay_input]))
    replay_records = [record for path in replay_files for record in iter_records(path)]
    replay_items, replay_stats = convert_records(
        replay_records,
        policy_min_q=args.policy_min_q,
        max_value_paths_per_dimension=args.max_value_paths_per_dimension,
        data_split="replay",
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
