from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from transformers import AutoTokenizer

from magicoder.prompt_template import QWEN_REVIEW_STEP_PROMPT
from magicoder.review_policy_value_inference import (
    generate_response,
    load_jsonl_item,
    load_policy,
    load_value_model,
    score_response,
)


def build_prompt(instruction: str, partial_response: str) -> str:
    return QWEN_REVIEW_STEP_PROMPT.format(instruction=instruction, response=partial_response)


def candidate_sort_key(candidate: dict[str, Any], score_key: str) -> tuple[float, float]:
    value_score = candidate["value_score"]
    return float(value_score[score_key]), float(value_score["last_value"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal value-guided code-review evaluation loop.")
    parser.add_argument("--policy_model_path", required=True)
    parser.add_argument("--value_model_path")
    parser.add_argument("--datafile_path", required=True)
    parser.add_argument("--item_index", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--max_steps", type=int, default=3)
    parser.add_argument("--num_candidates", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--score_key", choices=["last_value", "response_mean_value", "response_min_value"], default="last_value")
    args = parser.parse_args()

    item = load_jsonl_item(args.datafile_path, args.item_index)
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    policy_model = load_policy(args.policy_model_path, args.device, args.dtype)
    value_model = load_value_model(args.value_model_path or args.policy_model_path, args.device, args.dtype)

    partial_response = ""
    trace: list[dict[str, Any]] = []
    for step_index in range(args.max_steps):
        is_final_step = step_index == args.max_steps - 1
        stop = "</review>" if is_final_step else "</step>"
        prompt = build_prompt(item["instruction"], partial_response)
        candidates = []
        for candidate_index in range(args.num_candidates):
            with torch.no_grad():
                continuation = generate_response(
                    policy_model=policy_model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop=stop,
                )
                value_score = score_response(value_model, tokenizer, prompt, continuation)
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "continuation": continuation,
                    "value_score": value_score,
                }
            )
        best = max(candidates, key=lambda candidate: candidate_sort_key(candidate, args.score_key))
        partial_response += best["continuation"].strip() + "\n"
        trace.append(
            {
                "step_index": step_index,
                "is_final_step": is_final_step,
                "selected_candidate_index": best["candidate_index"],
                "selected_value": best["value_score"],
                "candidates": candidates,
            }
        )

    print(
        json.dumps(
            {
                "item_index": args.item_index,
                "target_dimension": item.get("target_dimension"),
                "terminal_tag": item.get("terminal_tag"),
                "terminal_q_value": item.get("terminal_q_value"),
                "final_response": partial_response,
                "trace": trace,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
