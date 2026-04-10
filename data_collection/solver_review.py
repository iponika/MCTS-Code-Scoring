from __future__ import annotations

import argparse
import json
import os
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


if __name__ == "__main__":
    args = parse_args()

    config = OmegaConf.structured(BaseConfig)
    custom_config = OmegaConf.load(args.custom_cfg)
    config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))

    data = load_codecriticbench_dataset(args.dataset, start=args.start, limit=args.limit)
    solver = Solver(config=config)
    output_path = build_output_path(args, config)
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
            solver.solve(agents, True)
            for sample, agent in zip(cur_batch, agents):
                record = {
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
                    "react": agent.return_states(),
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                writer.flush()
