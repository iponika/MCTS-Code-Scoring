from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

from omegaconf import OmegaConf
from tqdm import tqdm

from mcts_math.agents import ReviewMCTS
from mcts_math.config import BaseConfig
from mcts_math.review_utils import load_codecriticbench_dataset
from mcts_math.solver import Solver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_cfg", type=str, default="data_collection/configs/mcts_code_review.yaml")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/CodeCriticBench/data/CodeCriticBench.jsonl",
        help="Path to CodeCriticBench jsonl file.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="If set, write one pretty JSON file per sample into this directory.",
    )
    return parser.parse_args()


def batch(items: List[dict], batch_size: int) -> Iterable[List[dict]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def build_output_path(args, config) -> str:
    if args.output:
        return args.output
    dataset_name = os.path.basename(args.dataset).replace(".jsonl", "")
    model_name = os.path.basename(config.model_dir.rstrip("/"))
    return f"{dataset_name}.review_mcts.{model_name}.{args.start}_{args.limit}.jsonl"


def best_reviews_by_dimension(states: dict) -> dict:
    terminal_reviews = []
    for tag, state in states.items():
        if not isinstance(state, dict):
            continue
        if state.get("final_answer") and state.get("target_dimension"):
            terminal_reviews.append((tag, state))

    terminal_reviews.sort(key=lambda item: item[1].get("q_value", -100), reverse=True)
    best = {}
    for tag, state in terminal_reviews:
        dimension = state["target_dimension"]
        if dimension not in best:
            best[dimension] = {
                "tag": tag,
                "q_value": state.get("q_value"),
                "value": state.get("value"),
                "final_answer": state.get("final_answer"),
                "reward_details": state.get("reward_details"),
            }
    return best


def safe_filename_part(text: object) -> str:
    value = str(text if text is not None else "unknown").strip()
    safe = []
    for char in value:
        safe.append(char if char.isalnum() or char in ("-", "_") else "_")
    return "".join(safe).strip("_") or "unknown"


def build_record(sample: dict, react: dict) -> dict:
    return {
        "dataset_index": sample["dataset_index"],
        "question": sample["question"],
        "problem": sample["problem"],
        "candidate_code": sample["candidate_code"],
        "source": sample["source"],
        "subset": sample["subset"],
        "difficulty": sample["difficulty"],
        "reference_scores": sample["reference_scores"],
        "dimension_target_scores": sample["dimension_target_scores"],
        "objective": sample["objective"],
        "overall_score": sample["overall_score"],
        "correctness_label": sample["correctness_label"],
        "tests": sample["tests"],
        "best_reviews_by_dimension": best_reviews_by_dimension(react),
        "react": react,
    }


def write_per_sample_record(output_dir: Path, record: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "{idx}_{source}_{subset}.json".format(
        idx=safe_filename_part(record["dataset_index"]),
        source=safe_filename_part(record["source"]),
        subset=safe_filename_part(record["subset"]),
    )
    output_path = output_dir / filename
    with output_path.open("w", encoding="utf-8") as writer:
        json.dump(record, writer, ensure_ascii=False, indent=2)
        writer.write("\n")


if __name__ == "__main__":
    args = parse_args()

    config = OmegaConf.structured(BaseConfig)
    custom_config = OmegaConf.load(args.custom_cfg)
    config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))

    data = load_codecriticbench_dataset(args.dataset, start=args.start, limit=args.limit)
    solver = Solver(config=config)
    output_path = build_output_path(args, config)
    output_dir = Path(args.output_dir) if args.output_dir else None
    total_batches = max(1, (len(data) + config.batch_size - 1) // config.batch_size)

    with open(output_path, "w", encoding="utf-8") as writer:
        for cur_batch in tqdm(batch(data, config.batch_size), total=total_batches, desc="Review Batch"):
            agents = [
                ReviewMCTS(
                    config=config,
                    question=sample["question"],
                    review_sample=sample,
                )
                for sample in cur_batch
            ]
            results = solver.solve(agents, True)
            for sample, agent in zip(cur_batch, agents):
                react = results[sample["question"]]
                record = build_record(sample, react)
                if output_dir is not None:
                    write_per_sample_record(output_dir, record)
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                writer.flush()
