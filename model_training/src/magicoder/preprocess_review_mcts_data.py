from __future__ import annotations

import argparse
import json
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


def convert_records(
    records: Iterable[dict[str, Any]],
    policy_min_q: float,
    max_value_paths_per_dimension: int,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert review MCTS runs into policy/value training JSONL.")
    parser.add_argument("--input", nargs="+", required=True, help="Review MCTS sample JSON/JSONL files or directories.")
    parser.add_argument("--output_file", required=True, help="Output JSONL consumed by magicoder.train_multi --task review.")
    parser.add_argument("--policy_min_q", type=float, default=0.5, help="Minimum terminal q_value for LM imitation.")
    parser.add_argument(
        "--max_value_paths_per_dimension",
        type=int,
        default=0,
        help="0 keeps all terminal paths; otherwise cap value-only paths per sample dimension.",
    )
    args = parser.parse_args()

    input_files = list(iter_input_files([Path(path) for path in args.input]))
    records = [record for path in input_files for record in iter_records(path)]
    items, stats = convert_records(
        records,
        policy_min_q=args.policy_min_q,
        max_value_paths_per_dimension=args.max_value_paths_per_dimension,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats["input_files"] = len(input_files)
    stats["output_items"] = len(items)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
