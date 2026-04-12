from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from magicoder.llm_wrapper import AutoModelForCausalLMWithValueHead, V_HEAD_WEIGHTS_NAME
from magicoder.prompt_template import QWEN_REVIEW_STEP_PROMPT


def load_jsonl_item(path: str, index: int) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        for current_index, line in enumerate(handle):
            if current_index == index:
                return json.loads(line)
    raise IndexError(f"{path} has no item at index {index}")


def choose_dtype(dtype_name: str) -> torch.dtype | None:
    if dtype_name == "auto":
        return None
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def model_kwargs(device: str, dtype_name: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    dtype = choose_dtype(dtype_name)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if device == "auto" and torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    return kwargs


def first_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def move_to_device(model: torch.nn.Module, device: str) -> torch.nn.Module:
    if device == "cpu":
        return model.to("cpu")
    if device == "cuda":
        return model.to("cuda")
    return model


def load_policy(model_path: str, device: str, dtype_name: str):
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs(device, dtype_name))
    model = move_to_device(model, device)
    model.eval()
    return model


def load_value_model(model_path: str, device: str, dtype_name: str):
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path, **model_kwargs(device, dtype_name))
    vhead_path = Path(model_path) / V_HEAD_WEIGHTS_NAME
    if vhead_path.exists():
        model.load_state_dict(torch.load(vhead_path, map_location="cpu"), strict=False)
    model = move_to_device(model, device)
    model.eval()
    return model


def encode_text(tokenizer, text: str, device: torch.device) -> dict[str, torch.Tensor]:
    input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if tokenizer.bos_token_id is not None:
        bos = torch.tensor([[tokenizer.bos_token_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([bos, input_ids], dim=1)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": torch.ones_like(input_ids).to(device),
    }


def truncate_on_stop(text: str, stop: str) -> str:
    if not stop or stop not in text:
        return text
    return text.split(stop, 1)[0] + stop


@torch.no_grad()
def generate_response(
    policy_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop: str,
) -> str:
    encoded = encode_text(tokenizer, prompt, first_model_device(policy_model))
    generated = policy_model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    continuation_ids = generated[0, encoded["input_ids"].shape[1] :]
    continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    return truncate_on_stop(continuation, stop)


@torch.no_grad()
def score_response(value_model, tokenizer, prompt: str, response: str) -> dict[str, float | int]:
    prompt_encoded = encode_text(tokenizer, prompt, first_model_device(value_model))
    full_encoded = encode_text(tokenizer, prompt + response, first_model_device(value_model))
    _, _, values = value_model(**full_encoded, output_hidden_states=True, return_dict=True)
    values = torch.tanh(values).squeeze(0).float().detach().cpu()
    prompt_len = prompt_encoded["input_ids"].shape[1]
    response_values = values[prompt_len:] if values.shape[0] > prompt_len else values[-1:]
    return {
        "prompt_tokens": prompt_len,
        "total_tokens": int(full_encoded["input_ids"].shape[1]),
        "last_value": round(float(values[-1].item()), 6),
        "response_mean_value": round(float(response_values.mean().item()), 6),
        "response_min_value": round(float(response_values.min().item()), 6),
        "response_max_value": round(float(response_values.max().item()), 6),
    }


def item_prompt(item: dict[str, Any]) -> str:
    return QWEN_REVIEW_STEP_PROMPT.format(instruction=item["instruction"], response="")


def item_response(item: dict[str, Any]) -> str:
    return "\n".join(str(segment).strip() for segment in item.get("response", [])) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and/or score code-review reasoning with policy/value models.")
    parser.add_argument("--policy_model_path", required=True)
    parser.add_argument("--value_model_path")
    parser.add_argument("--datafile_path", required=True)
    parser.add_argument("--item_index", type=int, default=0)
    parser.add_argument("--use_gold_response", action="store_true", help="Score the stored response instead of generating.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--stop", default="</review>")
    args = parser.parse_args()

    item = load_jsonl_item(args.datafile_path, args.item_index)
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt = item_prompt(item)
    if args.use_gold_response:
        response = item_response(item)
    else:
        policy_model = load_policy(args.policy_model_path, args.device, args.dtype)
        response = generate_response(
            policy_model=policy_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop=args.stop,
        )

    value_model_path = args.value_model_path or args.policy_model_path
    value_model = load_value_model(value_model_path, args.device, args.dtype)
    score = score_response(value_model, tokenizer, prompt, response)
    print(
        json.dumps(
            {
                "item_index": args.item_index,
                "target_dimension": item.get("target_dimension"),
                "terminal_tag": item.get("terminal_tag"),
                "terminal_q_value": item.get("terminal_q_value"),
                "used_gold_response": args.use_gold_response,
                "response": response,
                "value_score": score,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
