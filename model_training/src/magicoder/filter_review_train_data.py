from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from magicoder.llm_wrapper import EncodingConfig, TokenizationContext
from magicoder.prompt_template import QWEN_REVIEW_STEP_PROMPT


def review_item_token_length(item: dict[str, Any], context: TokenizationContext) -> int:
    prompt = QWEN_REVIEW_STEP_PROMPT.format(instruction=item["instruction"], response="")
    prompt_ids = context.encode(EncodingConfig(add_bos=True, add_eos=False), [prompt])[0]
    total = len(prompt_ids) + 1
    for response in item.get("response") or []:
        response_ids = context.encode(EncodingConfig(add_bos=False, add_eos=False), [str(response).strip() + "\n"])[0]
        total += len(response_ids)
    return total


def item_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(item.get("source")),
        str(item.get("subset")),
        str(item.get("dataset_index")),
        str(item.get("terminal_tag")),
    )


def summarize_lengths(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {}
    sorted_lengths = sorted(lengths)
    return {
        "min": sorted_lengths[0],
        "p50": int(statistics.median(sorted_lengths)),
        "p90": sorted_lengths[int(0.9 * (len(sorted_lengths) - 1))],
        "p95": sorted_lengths[int(0.95 * (len(sorted_lengths) - 1))],
        "p99": sorted_lengths[int(0.99 * (len(sorted_lengths) - 1))],
        "max": sorted_lengths[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter review train JSONL to examples that fit a token budget.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model_key", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--min_policy_items", type=int, default=0, help="Fail if fewer policy items remain after filtering.")
    parser.add_argument("--max_items", type=int, default=0, help="Optional cap after filtering. 0 keeps all.")
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--allow_duplicate_keys", action="store_true")
    args = parser.parse_args()

    context = TokenizationContext.from_model_key(args.model_key, args.model_name_or_path)
    input_path = Path(args.input_file)
    items = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    lengths: list[int] = []
    kept_lengths: list[int] = []
    dropped_too_long = 0
    dropped_duplicate = 0

    for item in items:
        length = review_item_token_length(item, context)
        lengths.append(length)
        if length > args.max_tokens:
            dropped_too_long += 1
            continue
        key = item_key(item)
        if not args.allow_duplicate_keys and key in seen:
            dropped_duplicate += 1
            continue
        seen.add(key)
        item["token_length"] = length
        kept.append(item)
        kept_lengths.append(length)

    if args.max_items > 0 and len(kept) > args.max_items:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(kept)
        kept = kept[: args.max_items]

    random.Random(args.shuffle_seed).shuffle(kept)
    policy_count = sum(1 for item in kept if item.get("train_lm"))
    if policy_count < args.min_policy_items:
        raise ValueError(f"Only {policy_count} policy items remain; required at least {args.min_policy_items}.")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in kept:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "max_tokens": args.max_tokens,
        "input_items": len(items),
        "output_items": len(kept),
        "dropped_too_long": dropped_too_long,
        "dropped_duplicate": dropped_duplicate,
        "policy_items": policy_count,
        "value_only_items": sum(1 for item in kept if not item.get("train_lm")),
        "synthetic_type_counts": dict(Counter(str(item.get("synthetic_type") or "original") for item in kept)),
        "target_grade_counts": dict(Counter(str(item.get("target_axiom_grade")) for item in kept)),
        "input_length_summary": summarize_lengths(lengths),
        "output_length_summary": summarize_lengths(kept_lengths),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
