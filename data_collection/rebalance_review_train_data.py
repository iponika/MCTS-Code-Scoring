from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebalance review train JSONL by policy/value counts.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target_policy_count", type=int, default=-1)
    parser.add_argument("--target_value_count", type=int, default=-1)
    parser.add_argument("--target_total_count", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260424)
    return parser.parse_args()


def load_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_items(items: list[dict]) -> tuple[list[dict], list[dict]]:
    policy = [item for item in items if item.get("train_lm")]
    value_only = [item for item in items if not item.get("train_lm")]
    return policy, value_only


def sample_or_repeat(items: list[dict], target: int, rng: random.Random, tag_suffix: str) -> list[dict]:
    if target < 0:
        return [copy.deepcopy(item) for item in items]
    if not items:
        return []
    if len(items) >= target:
        sampled = rng.sample(items, target)
        return [copy.deepcopy(item) for item in sampled]

    expanded: list[dict] = []
    for index in range(target):
        base = copy.deepcopy(items[index % len(items)])
        base_tag = str(base.get("terminal_tag") or f"{tag_suffix}_{index}")
        base["terminal_tag"] = f"{base_tag}#rebalance{tag_suffix}{index}"
        base["rebalance_repeat_index"] = index
        expanded.append(base)
    return expanded


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    items = load_items(args.input)
    policy_items, value_items = split_items(items)

    out_policy = sample_or_repeat(policy_items, args.target_policy_count, rng, "policy")
    out_value = sample_or_repeat(value_items, args.target_value_count, rng, "value")
    selected = out_policy + out_value

    if args.target_total_count >= 0:
        if len(selected) > args.target_total_count:
            selected = rng.sample(selected, args.target_total_count)
        elif len(selected) < args.target_total_count and selected:
            extra = sample_or_repeat(selected, args.target_total_count, rng, "total")
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
        "output_total_count": len(selected),
        "output_policy_count": sum(1 for item in selected if item.get("train_lm")),
        "output_value_count": sum(1 for item in selected if not item.get("train_lm")),
        "output_sources": dict(sorted(Counter(str(item.get("synthetic_type") or item.get("source") or "unknown") for item in selected).items())),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
