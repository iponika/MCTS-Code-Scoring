from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from tqdm import tqdm

from mcts_math.axiom_scoring import axiom_functionally_correct, axiom_scalar_score, parse_axiom_grade
from mcts_math.config import BaseConfig
from mcts_math.llms.local_llm_engine import llm_engine
from mcts_math.prompts.prompt_sft import (
    apply_review_prompt_controls,
    build_review_prompt_from_sample,
)
from mcts_math.review_utils import (
    compute_review_reward,
    extract_reasoning_artifacts,
    load_codecriticbench_dataset,
    parse_review_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct single-call local-vLLM baseline for AXIOM code scoring.")
    parser.add_argument("--custom_cfg", default="data_collection/configs/mcts_code_review.yaml")
    parser.add_argument("--dataset", default="datasets/CodeCriticBench/data/CodeCriticBench.jsonl")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dimension", default="Correctness Verification")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    return parser.parse_args()


def load_config(path: str) -> Any:
    config = OmegaConf.structured(BaseConfig)
    custom_config = OmegaConf.load(path)
    config = OmegaConf.merge(config, custom_config)
    return OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))


def prompt_tests_text(sample: dict[str, Any], config: Any) -> str:
    if getattr(config, "show_tests_in_prompt", False):
        return sample["tests_for_prompt"]
    return "No tests are available to the reviewer."


def build_prompt(sample: dict[str, Any], dimension: str, config: Any) -> str:
    raw_prompt = build_review_prompt_from_sample(
        {
            "question": sample["question"],
            "candidate_code": sample["candidate_code"],
            "code_language": sample.get("code_language", "python"),
            "tests_for_prompt": prompt_tests_text(sample, config),
        },
        dimension=dimension,
        force_final=True,
        freeform_final_review=False,
        show_tests_in_prompt=bool(getattr(config, "show_tests_in_prompt", False)),
    )
    return apply_review_prompt_controls(
        raw_prompt,
        force_final=True,
        freeform_final_review=False,
        thinking_mode=getattr(config, "qwen_thinking_mode", ""),
        review_native_thinking_steps=bool(getattr(config, "review_native_thinking_steps", False)),
    )


def iter_batches(items: list[dict[str, Any]], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def normalize_review_text(text: str) -> str:
    artifacts = extract_reasoning_artifacts(text)
    text = artifacts["content"] or str(text or "")
    if "</review>" not in text and "<review>" in text:
        return text.rstrip() + "\n</review>"
    return text


def evaluated_candidate(index: int, text: str, sample: dict[str, Any], dimension: str) -> dict[str, Any]:
    artifacts = extract_reasoning_artifacts(text)
    normalized_text = normalize_review_text(artifacts["content"] or text)
    parsed = parse_review_payload(normalized_text)
    predicted = parse_axiom_grade(parsed or {})
    reward, reward_details = compute_review_reward(dimension, normalized_text, sample)
    return {
        "candidate_index": index,
        "raw_text": str(text or ""),
        "text": normalized_text,
        "reasoning": artifacts["reasoning"],
        "reasoning_source": artifacts["reasoning_source"],
        "parsed": parsed,
        "predicted_axiom_grade": predicted,
        "reward": reward,
        "reward_details": reward_details,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.custom_cfg)
    config.n_generate_sample = max(1, args.repeats)
    config.stop = ["</review>"]

    engine, sampling_params = llm_engine(config)
    sampling_params.n = config.n_generate_sample
    sampling_params.best_of = config.n_generate_sample
    sampling_params.stop = ["</review>"]

    samples = load_codecriticbench_dataset(args.dataset, start=args.start, limit=args.limit)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as writer:
        for sample_batch in tqdm(list(iter_batches(samples, args.batch_size)), desc="Direct Local Review"):
            prompts = [build_prompt(sample, args.dimension, config) for sample in sample_batch]
            outputs = engine.generate(prompts, sampling_params=sampling_params)
            for sample, output in zip(sample_batch, outputs):
                candidates = [
                    evaluated_candidate(index, item.text, sample, args.dimension)
                    for index, item in enumerate(output.outputs)
                ]
                best = candidates[0] if candidates else {}
                predicted_grade = best.get("predicted_axiom_grade")
                target_grade = sample.get("axiom_target_grade")
                record = {
                    "dataset_index": sample["dataset_index"],
                    "source": sample["source"],
                    "subset": sample["subset"],
                    "difficulty": sample["difficulty"],
                    "dimension": args.dimension,
                    "target_axiom_grade": target_grade,
                    "target_score": axiom_scalar_score(target_grade),
                    "predicted_axiom_grade": predicted_grade,
                    "predicted_score": axiom_scalar_score(predicted_grade) if predicted_grade is not None else None,
                    "grade_abs_error": abs(predicted_grade - target_grade) if predicted_grade is not None else None,
                    "boundary_correct": (
                        axiom_functionally_correct(predicted_grade) == axiom_functionally_correct(target_grade)
                        if predicted_grade is not None
                        else False
                    ),
                    "best_reward": best.get("reward"),
                    "best_reward_details": best.get("reward_details"),
                    "all_candidates": candidates,
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                writer.flush()


if __name__ == "__main__":
    main()
