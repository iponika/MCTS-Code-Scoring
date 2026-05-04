from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from tqdm import tqdm

from mcts_math.axiom_scoring import axiom_functionally_correct, axiom_scalar_score, parse_axiom_grade
from mcts_math.config import BaseConfig
from mcts_math.llms.local_llms import maybe_apply_chat_template
from mcts_math.llms.local_llm_engine import llm_engine
from mcts_math.prompts.prompt_sft import (
    REVIEW_FINAL_FORMAT_SECTION,
    REVIEW_STEP_FORMAT_SECTION,
    AXIOM_REFINEMENT_SCALE,
    REVIEW_EVIDENCE_RULES,
    REVIEW_FINAL_CONSISTENCY_RULE,
    QWEN_REVIEW_FINAL_PROMPT,
    QWEN_REVIEW_STEP_PROMPT,
)
from mcts_math.review_utils import (
    compact_native_think_body,
    compute_review_reward,
    extract_reasoning_artifacts,
    load_codecriticbench_dataset,
    parse_review_payload,
)
from solver_review import build_record

FINAL_REVIEW_PREFILL = '<review>\n{"axiom_grade": '


def relax_freeform_final_prompt(prompt: str) -> str:
    relaxed = prompt
    relaxed = relaxed.replace(
        "Output must be exactly one JSON object wrapped in <review> tags. Do not output natural-language text outside the tags, markdown fences, <think> blocks, <step> blocks, or code fixes. Otherwise the result cannot be parsed.\n",
        "You may reason before the final labeled answer, but finish with exactly one complete <review>...</review> block.\n",
    )
    relaxed = relaxed.replace(
        "Field rules:\n",
        "The final labeled answer begins at the last complete <review> block in your response. "
        "Inside that final block, write the JSON object required below.\n\nField rules:\n",
    )
    return relaxed


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
        help="Generate direct final <review> only, or non-MCTS sequential native reasoning followed by <review>.",
    )
    parser.add_argument(
        "--reasoning_steps",
        type=int,
        default=3,
        help="Number of sequential native reasoning notes to generate when --response_mode stepwise.",
    )
    parser.add_argument(
        "--step_context_mode",
        choices=["assistant_prefix", "instruction_context"],
        default="instruction_context",
        help="How prior reasoning notes are exposed during stepwise reasoning. instruction_context avoids assistant-prefix continuation.",
    )
    parser.add_argument(
        "--freeform_final_review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow free-form reasoning before the <review> block in review-only mode. "
             "Use --no-freeform_final_review to anchor the output with FINAL_REVIEW_PREFILL.",
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
    freeform_final_review: bool = False,
    steps_as_instruction_context: bool = False,
) -> str:
    template = QWEN_REVIEW_FINAL_PROMPT if force_final_review else QWEN_REVIEW_STEP_PROMPT
    completed_steps = partial_solution.strip() if partial_solution else ""
    partial_response = completed_steps
    if partial_response == "None":
        partial_response = ""
    if completed_steps == "None":
        completed_steps = ""
    if steps_as_instruction_context:
        partial_response = ""
    if partial_response:
        partial_response = partial_response.rstrip() + "\n"
    if force_final_review and not freeform_final_review:
        partial_response += FINAL_REVIEW_PREFILL
    prompt = template.format(
        question=sample["question"],
        candidate_code=sample["candidate_code"],
        code_language=sample.get("code_language", "python"),
        tests=prompt_tests_text(sample, config),
        partial_solution=partial_response,
        axiom_scale=AXIOM_REFINEMENT_SCALE,
        evidence_rules=REVIEW_EVIDENCE_RULES,
        final_consistency_rule=REVIEW_FINAL_CONSISTENCY_RULE,
        step_format_section=REVIEW_STEP_FORMAT_SECTION,
        final_format_section=REVIEW_FINAL_FORMAT_SECTION,
    )
    if force_final_review and steps_as_instruction_context and completed_steps:
        prompt = prompt.replace(
            "\n\nStructured final review format:\n",
            "\n\nPrevious analysis notes:\n"
            f"{completed_steps}\n\nStructured final review format:\n",
            1,
        )
    if (not force_final_review) and steps_as_instruction_context and completed_steps:
        prompt = prompt.replace(
            "\n\nIntermediate reasoning format:\n",
            "\n\nPrevious analysis notes:\n"
            f"{completed_steps}\n\nIntermediate reasoning format:\n",
            1,
        )
        prompt = prompt.replace(
            "If previous analysis notes are provided below or already present after @@ Response, use them as fixed context and add one new evidence item.\n",
            "If previous analysis notes are provided below, use them as fixed context and add one new evidence item without repeating them.\n",
        )
    if force_final_review and freeform_final_review:
        prompt = relax_freeform_final_prompt(prompt)
    thinking_mode = str(getattr(config, "qwen_thinking_mode", "") or "").strip().lower()
    if force_final_review and getattr(config, "review_native_thinking_steps", False) and not freeform_final_review:
        return prompt + "\n\n/no_think"
    if thinking_mode in {"think", "/think"}:
        return prompt + "\n\n/think"
    if thinking_mode in {"no_think", "no-think", "/no_think"}:
        return prompt + "\n\n/no_think"
    return prompt


def iter_batches(items: list[dict[str, Any]], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def normalize_review_text(text: str) -> str:
    artifacts = extract_reasoning_artifacts(text)
    text = artifacts["content"] or str(text or "")
    if "<review>" in text:
        review_suffix = text.rsplit("<review>", 1)[1]
        review_body = review_suffix.split("</review>", 1)[0].strip()
        normalized = "<review>\n" + review_body
        if "</review>" in review_suffix:
            normalized += "\n</review>"
        return normalized
    if "</review>" not in text and "<review>" in text:
        return text.rstrip() + "\n</review>"
    return text


def normalize_step_text(text: str) -> str:
    artifacts = extract_reasoning_artifacts(text)
    cleaned = (artifacts["reasoning"] or artifacts["content"] or str(text or "")).strip()
    if "<review>" in cleaned:
        cleaned = cleaned.split("<review>", 1)[0].strip()
    cleaned = cleaned.replace("<think>", "").replace("</think>", "").strip()
    if cleaned.startswith("<step>"):
        cleaned = cleaned[len("<step>"):].lstrip()
    if "</step>" in cleaned:
        cleaned = cleaned.split("</step>", 1)[0].strip()
    cleaned = compact_native_think_body(cleaned) or cleaned
    return cleaned


def direct_bootstrap_stop_tokens(response_mode: str) -> list[str]:
    if response_mode == "stepwise":
        return ["</think>"]
    return ["</review>"]


def build_react(candidates: list[dict[str, Any]], dimension: str) -> dict[str, dict[str, Any]]:
    react: dict[str, dict[str, Any]] = {}
    total = max(1, len(candidates))
    for index, candidate in enumerate(candidates):
        tag = f"c{index}"
        react[tag] = {
            "text": candidate["text"],
            "raw_text": candidate.get("raw_text", candidate["text"]),
            "reasoning": candidate.get("reasoning", ""),
            "reasoning_source": candidate.get("reasoning_source", "none"),
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
            segment_reasoning = candidate.get("segment_reasoning") or []
            segment_reasoning_source = candidate.get("segment_reasoning_source") or []
            node = {
                "text": segment,
                "reasoning": segment_reasoning[segment_index] if segment_index < len(segment_reasoning) else "",
                "reasoning_source": (
                    segment_reasoning_source[segment_index] if segment_index < len(segment_reasoning_source) else "none"
                ),
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


def evaluated_stepwise_candidate(
    index: int,
    segments: list[str],
    sample: dict[str, Any],
    dimension: str,
) -> dict[str, Any]:
    final_artifacts = extract_reasoning_artifacts(segments[-1] if segments else "")
    final_review = normalize_review_text(final_artifacts["content"] or (segments[-1] if segments else ""))
    parsed = parse_review_payload(final_review)
    predicted = parse_axiom_grade(parsed or {})
    reward, reward_details = compute_review_reward(dimension, final_review, sample)
    step_artifacts = [extract_reasoning_artifacts(segment) for segment in segments[:-1]]
    normalized_segments = [normalize_step_text(segment) for segment in segments[:-1]] + [final_review]
    return {
        "candidate_index": index,
        "text": "".join(normalized_segments),
        "segments": normalized_segments,
        "segment_reasoning": [item["reasoning"] for item in step_artifacts] + [final_artifacts["reasoning"]],
        "segment_reasoning_source": [item["reasoning_source"] for item in step_artifacts] + [final_artifacts["reasoning_source"]],
        "reasoning": final_artifacts["reasoning"],
        "reasoning_source": final_artifacts["reasoning_source"],
        "raw_final_text": str(segments[-1] if segments else ""),
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
    trajectories = [
        {
            "sample": sample,
            "repeat_index": repeat_index,
        }
        for sample in sample_batch
        for repeat_index in range(max(1, args.repeats))
    ]

    # Final-review generation now uses greedy decoding for format stability.
    # To preserve multiple direct rollouts, duplicate prompts instead of using
    # n>1 / best_of>1, which vLLM rejects for greedy sampling.
    sampling_params.n = 1
    sampling_params.best_of = 1
    sampling_params.stop = ["</review>"]
    sampling_params.temperature = 0.0
    sampling_params.top_p = 1.0
    freeform = getattr(args, "freeform_final_review", True)
    prompts = [
        build_prompt(
            item["sample"],
            args.dimension,
            config,
            partial_solution="None",
            force_final_review=True,
            freeform_final_review=freeform,
        )
        for item in trajectories
    ]
    prompts = maybe_apply_chat_template(prompts, engine, config)
    outputs = engine.generate(prompts, sampling_params=sampling_params)
    by_sample_index: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(sample_batch))}
    sample_position = {id(sample): index for index, sample in enumerate(sample_batch)}
    for item, output in zip(trajectories, outputs):
        text = output.outputs[0].text if output.outputs else ""
        if not freeform:
            text = normalize_review_text(FINAL_REVIEW_PREFILL + text)
        sample = item["sample"]
        candidate = evaluated_candidate(item["repeat_index"], text, sample, args.dimension)
        by_sample_index[sample_position[id(sample)]].append(candidate)

    return [(sample, by_sample_index[index]) for index, sample in enumerate(sample_batch)]


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
    sampling_params.stop = ["</think>"]
    for _ in range(max(0, args.reasoning_steps)):
        prompts = [
            build_prompt(
                item["sample"],
                args.dimension,
                config,
                partial_solution="".join(item["segments"]) or "None",
                force_final_review=False,
                steps_as_instruction_context=args.step_context_mode == "instruction_context",
            )
            for item in trajectories
        ]
        prompts = maybe_apply_chat_template(prompts, engine, config)
        outputs = engine.generate(prompts, sampling_params=sampling_params)
        for item, output in zip(trajectories, outputs):
            text = output.outputs[0].text if output.outputs else ""
            item["segments"].append(normalize_step_text(text))

    sampling_params.stop = ["</review>"]
    sampling_params.temperature = 0.0
    sampling_params.top_p = 1.0
    prompts = [
        build_prompt(
            item["sample"],
            args.dimension,
            config,
            partial_solution="".join(item["segments"]) or "None",
            force_final_review=True,
            steps_as_instruction_context=args.step_context_mode == "instruction_context",
        )
        for item in trajectories
    ]
    prompts = maybe_apply_chat_template(prompts, engine, config)
    outputs = engine.generate(prompts, sampling_params=sampling_params)
    by_sample_index: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(sample_batch))}
    sample_position = {id(sample): index for index, sample in enumerate(sample_batch)}
    for item, output in zip(trajectories, outputs):
        text = output.outputs[0].text if output.outputs else ""
        segments = [*item["segments"], normalize_review_text(FINAL_REVIEW_PREFILL + text)]
        sample = item["sample"]
        candidate_index = item["repeat_index"]
        candidate = evaluated_stepwise_candidate(candidate_index, segments, sample, args.dimension)
        by_sample_index[sample_position[id(sample)]].append(candidate)

    return [(sample, by_sample_index[index]) for index, sample in enumerate(sample_batch)]


def main() -> None:
    args = parse_args()
    config = load_config(args.custom_cfg)
    config.n_generate_sample = max(1, args.repeats)
    config.stop = direct_bootstrap_stop_tokens(args.response_mode)

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
