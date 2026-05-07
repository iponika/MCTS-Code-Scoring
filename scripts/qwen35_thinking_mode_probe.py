#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from typing import Any

from transformers import AutoTokenizer


def render(tokenizer: Any, content: str, enable_thinking: bool | None) -> dict[str, Any]:
    messages = [{"role": "user", "content": content}]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        text = tokenizer.apply_chat_template(messages, **kwargs)
        return {"ok": True, "text": text, "error": None}
    except TypeError as exc:
        if enable_thinking is None:
            return {"ok": False, "text": "", "error": repr(exc)}
        kwargs.pop("enable_thinking", None)
        text = tokenizer.apply_chat_template(messages, **kwargs)
        return {"ok": False, "text": text, "error": repr(exc)}


def summarize_text(text: str) -> dict[str, Any]:
    return {
        "chars": len(text),
        "contains_think_tag": "<think>" in text or "</think>" in text,
        "contains_no_think": "/no_think" in text,
        "contains_think_switch": "/think" in text,
        "tail": text[-500:],
    }


def run_vllm(model: str, prompts: dict[str, str], max_tokens: int, gpu_memory_utilization: float) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["</think>", "</review>"],
    )
    outputs = llm.generate(list(prompts.values()), sampling_params=sampling)
    result: dict[str, Any] = {}
    for key, output in zip(prompts.keys(), outputs):
        text = output.outputs[0].text if output.outputs else ""
        result[key] = {
            "text": text,
            "contains_think_tag": "<think>" in text or "</think>" in text,
            "contains_thinking_process": "Thinking Process" in text,
            "nonempty": bool(text.strip()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Qwen3.5 chat-template think/no_think behavior.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--prompt", default="用一句话介绍你自己。")
    parser.add_argument("--run-vllm", action="store_true", help="Also run a short vLLM generation check; requires an idle GPU.")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    rendered = {
        "auto": render(tokenizer, args.prompt, None),
        "think": render(tokenizer, args.prompt, True),
        "no_think": render(tokenizer, args.prompt, False),
        "suffix_think": render(tokenizer, args.prompt.rstrip() + " /think", None),
        "suffix_no_think": render(tokenizer, args.prompt.rstrip() + " /no_think", None),
    }
    summary = {
        key: {
            "ok": value["ok"],
            "error": value["error"],
            **summarize_text(value["text"]),
        }
        for key, value in rendered.items()
    }
    summary["comparisons"] = {
        "think_vs_no_think_equal": rendered["think"]["text"] == rendered["no_think"]["text"],
        "auto_vs_think_equal": rendered["auto"]["text"] == rendered["think"]["text"],
        "auto_vs_no_think_equal": rendered["auto"]["text"] == rendered["no_think"]["text"],
    }

    if args.run_vllm:
        summary["vllm"] = run_vllm(
            args.model,
            {
                "think": rendered["think"]["text"],
                "no_think": rendered["no_think"]["text"],
                "suffix_no_think": rendered["suffix_no_think"]["text"],
            },
            args.max_tokens,
            args.gpu_memory_utilization,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
