from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebalance review train JSONL by policy/value counts.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target_policy_count", type=int, default=-1)
    parser.add_argument("--target_value_count", type=int, default=-1)
    parser.add_argument("--target_total_count", type=int, default=-1)
    parser.add_argument("--stratify_by_dataset", action="store_true")
    parser.add_argument("--stratify_by_delta_bucket", action="store_true")
    parser.add_argument("--seed", type=int, default=20260424)
    return parser.parse_args()


def load_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_items(items: list[dict]) -> tuple[list[dict], list[dict]]:
    policy = [item for item in items if item.get("train_lm")]
    value_only = [item for item in items if not item.get("train_lm")]
    return policy, value_only


def delta_bucket(item: dict) -> str:
    target = item.get("target_axiom_grade")
    parsed = item.get("parsed_axiom_grade")
    if target is None or parsed is None:
        return "unknown"
    delta = parsed - target
    if delta <= -2:
        return "under_2plus"
    if delta == -1:
        return "under_1"
    if delta == 0:
        return "exact"
    if delta == 1:
        return "over_1"
    return "over_2plus"


def group_key(item: dict, *, stratify_by_dataset: bool, stratify_by_delta_bucket: bool) -> tuple[str, ...]:
    key: list[str] = []
    if stratify_by_dataset:
        key.append(str(item.get("dataset_index") or "__missing_dataset__"))
    if stratify_by_delta_bucket:
        key.append(delta_bucket(item))
    if not key:
        key.append("__all__")
    return tuple(key)


def clone_with_tag(item: dict, *, tag_suffix: str, index: int) -> dict:
    base = copy.deepcopy(item)
    base_tag = str(base.get("terminal_tag") or f"{tag_suffix}_{index}")
    base["terminal_tag"] = f"{base_tag}#rebalance{tag_suffix}{index}"
    base["rebalance_repeat_index"] = index
    return base


def stratified_sample_or_repeat(
    items: list[dict],
    target: int,
    rng: random.Random,
    tag_suffix: str,
    *,
    stratify_by_dataset: bool,
    stratify_by_delta_bucket: bool,
) -> list[dict]:
    if target < 0:
        return [copy.deepcopy(item) for item in items]
    if not items or target == 0:
        return []

    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for item in items:
        groups[group_key(item, stratify_by_dataset=stratify_by_dataset, stratify_by_delta_bucket=stratify_by_delta_bucket)].append(item)

    group_keys = list(groups)
    rng.shuffle(group_keys)
    for key in group_keys:
        rng.shuffle(groups[key])

    selected: list[dict] = []
    take_positions = {key: 0 for key in group_keys}
    repeat_index = 0
    active_keys = [key for key in group_keys if groups[key]]
    while len(selected) < target and active_keys:
        next_active: list[tuple[str, ...]] = []
        for key in active_keys:
            if len(selected) >= target:
                break
            pos = take_positions[key]
            if pos < len(groups[key]):
                selected.append(copy.deepcopy(groups[key][pos]))
                take_positions[key] = pos + 1
            if take_positions[key] < len(groups[key]):
                next_active.append(key)
        active_keys = next_active

    if len(selected) >= target:
        return selected[:target]

    # If filtering/stratification leaves too few items, repeat in the same round-robin order.
    active_keys = list(group_keys)
    while len(selected) < target and active_keys:
        next_active: list[tuple[str, ...]] = []
        for key in active_keys:
            if len(selected) >= target:
                break
            base_items = groups[key]
            if not base_items:
                continue
            base = base_items[repeat_index % len(base_items)]
            selected.append(clone_with_tag(base, tag_suffix=tag_suffix, index=repeat_index))
            repeat_index += 1
            next_active.append(key)
        active_keys = next_active
    return selected[:target]


def summarize_delta(items: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(delta_bucket(item) for item in items).items()))


def summarize_targets(items: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(str(item.get("target_axiom_grade")) for item in items).items()))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    items = load_items(args.input)
    policy_items, value_items = split_items(items)

    out_policy = stratified_sample_or_repeat(
        policy_items,
        args.target_policy_count,
        rng,
        "policy",
        stratify_by_dataset=args.stratify_by_dataset,
        stratify_by_delta_bucket=args.stratify_by_delta_bucket,
    )
    out_value = stratified_sample_or_repeat(
        value_items,
        args.target_value_count,
        rng,
        "value",
        stratify_by_dataset=args.stratify_by_dataset,
        stratify_by_delta_bucket=args.stratify_by_delta_bucket,
    )
    selected = out_policy + out_value

    if args.target_total_count >= 0:
        if len(selected) > args.target_total_count:
            selected = rng.sample(selected, args.target_total_count)
        elif len(selected) < args.target_total_count and selected:
            extra = stratified_sample_or_repeat(
                selected,
                args.target_total_count,
                rng,
                "total",
                stratify_by_dataset=args.stratify_by_dataset,
                stratify_by_delta_bucket=args.stratify_by_delta_bucket,
            )
            selected = extra

    rng.shuffle(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as writer:
        for item in selected:
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "input_policy_count": len(policy_items),
        "input_value_count": len(value_items),
        "target_policy_count": args.target_policy_count,
        "target_value_count": args.target_value_count,
        "target_total_count": args.target_total_count,
        "stratify_by_dataset": args.stratify_by_dataset,
        "stratify_by_delta_bucket": args.stratify_by_delta_bucket,
        "output_total_count": len(selected),
        "output_policy_count": sum(1 for item in selected if item.get("train_lm")),
        "output_value_count": sum(1 for item in selected if not item.get("train_lm")),
        "input_target_grade_distribution": summarize_targets(items),
        "input_delta_bucket_distribution": summarize_delta(items),
        "output_target_grade_distribution": summarize_targets(selected),
        "output_delta_bucket_distribution": summarize_delta(selected),
        "output_sources": dict(sorted(Counter(str(item.get("synthetic_type") or item.get("source") or "unknown") for item in selected).items())),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
