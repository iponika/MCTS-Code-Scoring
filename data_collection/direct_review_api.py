from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from tqdm import tqdm

from mcts_math.axiom_scoring import axiom_functionally_correct, axiom_scalar_score, parse_axiom_grade
from mcts_math.config import BaseConfig
from mcts_math.llms.openai_api_llm import OpenAICompatibleGenerator, build_api_sampling_params
from mcts_math.prompts.prompt_sft import (
    REVIEW_FINAL_FORMAT_SECTION,
    AXIOM_REFINEMENT_SCALE,
    REVIEW_EVIDENCE_RULES,
    REVIEW_FINAL_CONSISTENCY_RULE,
    QWEN_REVIEW_FINAL_PROMPT,
)
from mcts_math.review_utils import (
    compute_review_reward,
    extract_reasoning_artifacts,
    load_codecriticbench_dataset,
    parse_review_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct single-call API baseline for AXIOM code scoring.")
    parser.add_argument("--custom_cfg", default="data_collection/configs/mcts_code_review_api.yaml")
    parser.add_argument("--dataset", default="datasets/CodeCriticBench/data/CodeCriticBench.jsonl")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dimension", default="Correctness Verification")
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
    return QWEN_REVIEW_FINAL_PROMPT.format(
        question=sample["question"],
        candidate_code=sample["candidate_code"],
        code_language=sample.get("code_language", "python"),
        tests=prompt_tests_text(sample, config),
        partial_solution="None",
        axiom_scale=AXIOM_REFINEMENT_SCALE,
        evidence_rules=REVIEW_EVIDENCE_RULES,
        final_consistency_rule=REVIEW_FINAL_CONSISTENCY_RULE,
        step_format_section="",
        final_format_section=REVIEW_FINAL_FORMAT_SECTION,
    )


def best_direct_prediction(outputs: list[str], sample: dict[str, Any], dimension: str) -> dict[str, Any]:
    candidates = []
    for index, text in enumerate(outputs):
        parsed = parse_review_payload(text)
        predicted_grade = parse_axiom_grade(parsed or {})
        reward, reward_details = compute_review_reward(dimension, text, sample)
        candidates.append(
            {
                "candidate_index": index,
                "text": text,
                "parsed": parsed,
                "predicted_axiom_grade": predicted_grade,
                "reward": reward,
                "reward_details": reward_details,
            }
        )
    return max(candidates, key=lambda item: item["reward"]) if candidates else {}


def main() -> None:
    args = parse_args()
    config = load_config(args.custom_cfg)
    config.llm_backend = "openai_api"
    generator = OpenAICompatibleGenerator(config)
    sampling_params = build_api_sampling_params(config)
    sampling_params.n = max(1, args.repeats)
    sampling_params.best_of = sampling_params.n
    sampling_params.stop = ["</review>"]

    samples = load_codecriticbench_dataset(args.dataset, start=args.start, limit=args.limit)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as writer:
        for sample in tqdm(samples, desc="Direct API Review"):
            prompt = build_prompt(sample, args.dimension, config)
            api_output = generator([prompt], sampling_params)[0]
            texts = []
            for output in api_output.outputs:
                artifacts = extract_reasoning_artifacts(
                    output.text,
                    structured_reasoning=getattr(output, "reasoning", None),
                )
                text = artifacts["content"] or output.text
                if "</review>" not in text and "<review>" in text:
                    text = text.rstrip() + "\n</review>"
                texts.append(
                    {
                        "text": text,
                        "raw_text": output.text,
                        "reasoning": artifacts["reasoning"],
                        "reasoning_source": artifacts["reasoning_source"],
                    }
                )
            evaluated_candidates = []
            for index, payload in enumerate(texts):
                text = payload["text"]
                parsed = parse_review_payload(text)
                predicted = parse_axiom_grade(parsed or {})
                reward, reward_details = compute_review_reward(args.dimension, text, sample)
                evaluated_candidates.append(
                    {
                        "candidate_index": index,
                        "raw_text": payload["raw_text"],
                        "text": text,
                        "reasoning": payload["reasoning"],
                        "reasoning_source": payload["reasoning_source"],
                        "parsed": parsed,
                        "predicted_axiom_grade": predicted,
                        "reward": reward,
                        "reward_details": reward_details,
                    }
                )
            best = evaluated_candidates[0] if evaluated_candidates else {}
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
                "all_candidates": evaluated_candidates,
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.flush()


if __name__ == "__main__":
    main()
