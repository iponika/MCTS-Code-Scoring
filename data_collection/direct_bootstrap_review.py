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
from mcts_math.prompts.prompt_sft import QWEN_REVIEW_PROMPT
from mcts_math.review_utils import (
    DEFAULT_DIMENSION_RUBRIC,
    compute_review_reward,
    load_codecriticbench_dataset,
    parse_review_payload,
)
from solver_review import build_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct independent bootstrap exporter for AXIOM code scoring.")
    parser.add_argument("--custom_cfg", default="data_collection/configs/mcts_code_review_qwen3_4b.yaml")
    parser.add_argument("--dataset", default="datasets/CodeCriticBench/data/CodeCriticBench.jsonl")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dimension", default="Correctness Verification")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=4)
    return parser.parse_args()


def load_config(path: str) -> Any:
    config = OmegaConf.structured(BaseConfig)
    custom_config = OmegaConf.load(path)
    config = OmegaConf.merge(config, custom_config)
    return OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))


def build_prompt(sample: dict[str, Any], dimension: str) -> str:
    rubric = sample["dimension_rubrics"].get(dimension) or DEFAULT_DIMENSION_RUBRIC.get(dimension, "")
    return QWEN_REVIEW_PROMPT.format(
        dimension=dimension,
        rubric=rubric,
        question=sample["question"],
        candidate_code=sample["candidate_code"],
        code_language=sample.get("code_language", "python"),
        tests=sample["tests_for_prompt"],
        partial_solution="None",
        mode_instruction=(
            "Output only one structured final review in the exact <review> JSON format below. "
            "Do not output <step> blocks."
        ),
    )


def iter_batches(items: list[dict[str, Any]], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def normalize_review_text(text: str) -> str:
    if "</review>" not in text and "<review>" in text:
        return text.rstrip() + "\n</review>"
    return text


def build_react(candidates: list[dict[str, Any]], dimension: str) -> dict[str, dict[str, Any]]:
    react: dict[str, dict[str, Any]] = {}
    total = max(1, len(candidates))
    for index, candidate in enumerate(candidates):
        tag = f"c{index}"
        react[tag] = {
            "text": candidate["text"],
            "q_value": candidate["reward"],
            "value": candidate["reward"],
            "prior": round(1.0 / total, 6),
            "visit_count": 1,
            "target_dimension": dimension,
            "final_answer": candidate["text"],
            "reward_details": json.dumps(candidate["reward_details"], ensure_ascii=False),
        }
    return react


def evaluated_candidate(index: int, text: str, sample: dict[str, Any], dimension: str) -> dict[str, Any]:
    text = normalize_review_text(text)
    parsed = parse_review_payload(text)
    predicted = parse_axiom_grade(parsed or {})
    reward, reward_details = compute_review_reward(dimension, text, sample)
    return {
        "candidate_index": index,
        "text": text,
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
        for sample_batch in tqdm(list(iter_batches(samples, args.batch_size)), desc="Direct Bootstrap Review"):
            prompts = [build_prompt(sample, args.dimension) for sample in sample_batch]
            outputs = engine.generate(prompts, sampling_params=sampling_params)
            for sample, output in zip(sample_batch, outputs):
                candidates = [
                    evaluated_candidate(index, item.text, sample, args.dimension)
                    for index, item in enumerate(output.outputs)
                ]
                react = build_react(candidates, args.dimension)
                record = build_record(sample, react)
                best = max(candidates, key=lambda item: item["reward"], default=None)
                record["bootstrap_mode"] = "direct_independent_rollouts"
                record["generation_repeats"] = args.repeats
                record["dimension"] = args.dimension
                record["all_candidates"] = candidates
                if best is not None:
                    predicted_grade = best.get("predicted_axiom_grade")
                    target_grade = sample.get("axiom_target_grade")
                    record["best_candidate"] = best
                    record["predicted_axiom_grade"] = predicted_grade
                    record["predicted_score"] = axiom_scalar_score(predicted_grade) if predicted_grade is not None else None
                    record["grade_abs_error"] = abs(predicted_grade - target_grade) if predicted_grade is not None else None
                    record["boundary_correct"] = (
                        axiom_functionally_correct(predicted_grade) == axiom_functionally_correct(target_grade)
                        if predicted_grade is not None
                        else False
                    )
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                writer.flush()


if __name__ == "__main__":
    main()
