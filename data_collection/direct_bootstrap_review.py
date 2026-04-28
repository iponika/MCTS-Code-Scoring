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
    REVIEW_FINAL_FORMAT_RULE,
    REVIEW_FINAL_FORMAT_SECTION,
    QWEN_REVIEW_PROMPT,
)
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
    parser.add_argument(
        "--response_mode",
        choices=["review", "stepwise"],
        default="review",
        help="Generate direct final <review> only, or non-MCTS sequential <step> reasoning followed by <review>.",
    )
    parser.add_argument(
        "--reasoning_steps",
        type=int,
        default=3,
        help="Number of sequential <step> blocks to generate when --response_mode stepwise.",
    )
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


def build_prompt(
    sample: dict[str, Any],
    dimension: str,
    config: Any,
    *,
    partial_solution: str = "None",
    force_final_review: bool = True,
) -> str:
    rubric = sample["dimension_rubrics"].get(dimension) or DEFAULT_DIMENSION_RUBRIC.get(dimension, "")
    if force_final_review:
        mode_instruction = (
            "Output only one structured final review in the exact <review> JSON format below. "
            "Do not output <step> blocks."
        )
        format_rule = REVIEW_FINAL_FORMAT_RULE
        output_format_section = REVIEW_FINAL_FORMAT_SECTION
    else:
        mode_instruction = (
            "Output exactly one concise next review reasoning step wrapped in <step>...</step>. Never output <review> yet."
        )
        from mcts_math.prompts.prompt_sft import REVIEW_STEP_FORMAT_RULE, REVIEW_STEP_FORMAT_SECTION

        format_rule = REVIEW_STEP_FORMAT_RULE
        output_format_section = REVIEW_STEP_FORMAT_SECTION
    return QWEN_REVIEW_PROMPT.format(
        dimension=dimension,
        rubric=rubric,
        question=sample["question"],
        candidate_code=sample["candidate_code"],
        code_language=sample.get("code_language", "python"),
        tests=prompt_tests_text(sample, config),
        partial_solution=partial_solution.strip() if partial_solution else "None",
        mode_instruction=mode_instruction,
        format_rule=format_rule,
        output_format_section=output_format_section,
    )


def iter_batches(items: list[dict[str, Any]], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def normalize_review_text(text: str) -> str:
    if "</review>" not in text and "<review>" in text:
        return text.rstrip() + "\n</review>"
    return text


def normalize_step_text(text: str) -> str:
    cleaned = text.strip()
    if "<review>" in cleaned:
        cleaned = cleaned.split("<review>", 1)[0].strip()
    if cleaned.startswith("<step>") and "</step>" not in cleaned:
        return cleaned.rstrip() + "\n</step>"
    if cleaned.startswith("<step>"):
        return cleaned
    if "</step>" in cleaned:
        cleaned = cleaned.split("</step>", 1)[0].strip()
    return f"<step>\n{cleaned}\n</step>"


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


def build_stepwise_react(candidates: list[dict[str, Any]], dimension: str) -> dict[str, dict[str, Any]]:
    react: dict[str, dict[str, Any]] = {}
    total = max(1, len(candidates))
    for index, candidate in enumerate(candidates):
        parent_tag = ""
        reward = candidate["reward"]
        for segment_index, segment in enumerate(candidate["segments"]):
            tag = f"c{index}" if not parent_tag else f"{parent_tag}.0"
            is_terminal = segment_index == len(candidate["segments"]) - 1
            node = {
                "text": segment,
                "q_value": reward,
                "value": reward,
                "prior": round(1.0 / total, 6) if segment_index == 0 else 1.0,
                "visit_count": 1,
                "target_dimension": dimension,
            }
            if is_terminal:
                node["final_answer"] = segment
                node["reward_details"] = json.dumps(candidate["reward_details"], ensure_ascii=False)
            react[tag] = node
            parent_tag = tag
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


def evaluated_stepwise_candidate(
    index: int,
    segments: list[str],
    sample: dict[str, Any],
    dimension: str,
) -> dict[str, Any]:
    final_review = normalize_review_text(segments[-1] if segments else "")
    parsed = parse_review_payload(final_review)
    predicted = parse_axiom_grade(parsed or {})
    reward, reward_details = compute_review_reward(dimension, final_review, sample)
    normalized_segments = [normalize_step_text(segment) for segment in segments[:-1]] + [final_review]
    return {
        "candidate_index": index,
        "text": "".join(normalized_segments),
        "segments": normalized_segments,
        "parsed": parsed,
        "predicted_axiom_grade": predicted,
        "reward": reward,
        "reward_details": reward_details,
    }


def generate_review_only(
    sample_batch: list[dict[str, Any]],
    args: argparse.Namespace,
    config: Any,
    engine: Any,
    sampling_params: Any,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    sampling_params.n = config.n_generate_sample
    sampling_params.best_of = config.n_generate_sample
    sampling_params.stop = ["</review>"]
    prompts = [
        build_prompt(sample, args.dimension, config, partial_solution="None", force_final_review=True)
        for sample in sample_batch
    ]
    outputs = engine.generate(prompts, sampling_params=sampling_params)
    result = []
    for sample, output in zip(sample_batch, outputs):
        candidates = [
            evaluated_candidate(index, item.text, sample, args.dimension)
            for index, item in enumerate(output.outputs)
        ]
        result.append((sample, candidates))
    return result


def generate_stepwise(
    sample_batch: list[dict[str, Any]],
    args: argparse.Namespace,
    config: Any,
    engine: Any,
    sampling_params: Any,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    trajectories = [
        {
            "sample": sample,
            "repeat_index": repeat_index,
            "segments": [],
        }
        for sample in sample_batch
        for repeat_index in range(max(1, args.repeats))
    ]

    sampling_params.n = 1
    sampling_params.best_of = 1
    sampling_params.stop = ["</step>"]
    for _ in range(max(0, args.reasoning_steps)):
        prompts = [
            build_prompt(
                item["sample"],
                args.dimension,
                config,
                partial_solution="".join(item["segments"]) or "None",
                force_final_review=False,
            )
            for item in trajectories
        ]
        outputs = engine.generate(prompts, sampling_params=sampling_params)
        for item, output in zip(trajectories, outputs):
            text = output.outputs[0].text if output.outputs else ""
            item["segments"].append(normalize_step_text(text))

    sampling_params.stop = ["</review>"]
    prompts = [
        build_prompt(
            item["sample"],
            args.dimension,
            config,
            partial_solution="".join(item["segments"]) or "None",
            force_final_review=True,
        )
        for item in trajectories
    ]
    outputs = engine.generate(prompts, sampling_params=sampling_params)
    by_sample_index: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(sample_batch))}
    sample_position = {id(sample): index for index, sample in enumerate(sample_batch)}
    for item, output in zip(trajectories, outputs):
        text = output.outputs[0].text if output.outputs else ""
        segments = [*item["segments"], normalize_review_text(text)]
        sample = item["sample"]
        candidate_index = item["repeat_index"]
        candidate = evaluated_stepwise_candidate(candidate_index, segments, sample, args.dimension)
        by_sample_index[sample_position[id(sample)]].append(candidate)

    return [(sample, by_sample_index[index]) for index, sample in enumerate(sample_batch)]


def main() -> None:
    args = parse_args()
    config = load_config(args.custom_cfg)
    config.n_generate_sample = max(1, args.repeats)
    config.stop = ["</review>" if args.response_mode == "review" else "</step>"]

    engine, sampling_params = llm_engine(config)

    samples = load_codecriticbench_dataset(args.dataset, start=args.start, limit=args.limit)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as writer:
        for sample_batch in tqdm(list(iter_batches(samples, args.batch_size)), desc="Direct Bootstrap Review"):
            if args.response_mode == "stepwise":
                batch_results = generate_stepwise(sample_batch, args, config, engine, sampling_params)
            else:
                batch_results = generate_review_only(sample_batch, args, config, engine, sampling_params)
            for sample, candidates in batch_results:
                react = (
                    build_stepwise_react(candidates, args.dimension)
                    if args.response_mode == "stepwise"
                    else build_react(candidates, args.dimension)
                )
                record = build_record(sample, react)
                best = max(candidates, key=lambda item: item["reward"], default=None)
                record["bootstrap_mode"] = (
                    "direct_stepwise_rollouts" if args.response_mode == "stepwise" else "direct_independent_rollouts"
                )
                record["direct_response_mode"] = args.response_mode
                record["reasoning_steps"] = args.reasoning_steps if args.response_mode == "stepwise" else 0
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
