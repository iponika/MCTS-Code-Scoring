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
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume an interrupted run by appending to --output and skipping dataset_index values already written.",
    )
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
        "axiom_target_grade": sample.get("axiom_target_grade"),
        "axiom_target_score": sample.get("axiom_target_score"),
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
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as writer:
        json.dump(record, writer, ensure_ascii=False, indent=2)
        writer.write("\n")
    tmp_path.replace(output_path)


def completed_dataset_indices_from_jsonl(path: Path) -> set[str]:
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as reader:
        for line_no, line in enumerate(reader, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping incomplete/corrupt resume line in {path}:{line_no}")
                continue
            if isinstance(record, dict) and record.get("dataset_index") is not None:
                completed.add(str(record["dataset_index"]))
    return completed


def sample_records_from_dir(path: Path | None, allowed_indices: set[str] | None = None) -> dict[str, dict]:
    records = {}
    if path is None or not path.exists():
        return records
    for item in sorted(path.glob("*.json")):
        try:
            record = json.loads(item.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"Skipping incomplete/corrupt resume sample file {item}")
            continue
        if isinstance(record, dict) and record.get("dataset_index") is not None:
            index = str(record["dataset_index"])
            if allowed_indices is None or index in allowed_indices:
                records[index] = record
    return records


def sort_dataset_indices(indices: Iterable[str]) -> list[str]:
    def key(index: str) -> tuple[int, int | str]:
        try:
            return (0, int(index))
        except ValueError:
            return (1, index)

    return sorted(indices, key=key)


def ensure_trailing_newline(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as reader:
        reader.seek(-1, os.SEEK_END)
        last_byte = reader.read(1)
    if last_byte != b"\n":
        with path.open("ab") as writer:
            writer.write(b"\n")


def append_records_to_jsonl(path: Path, records_by_index: dict[str, dict]) -> None:
    if not records_by_index:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_trailing_newline(path)
    with path.open("a", encoding="utf-8") as writer:
        for index in sort_dataset_indices(records_by_index.keys()):
            writer.write(json.dumps(records_by_index[index], ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = parse_args()

    config = OmegaConf.structured(BaseConfig)
    custom_config = OmegaConf.load(args.custom_cfg)
    config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))

    data = load_codecriticbench_dataset(args.dataset, start=args.start, limit=args.limit)
    output_path = Path(build_output_path(args, config))
    output_dir = Path(args.output_dir) if args.output_dir else None
    output_parent = output_path.parent
    if output_parent != Path("."):
        output_parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        requested_indices = {str(sample["dataset_index"]) for sample in data}
        completed_indices = completed_dataset_indices_from_jsonl(output_path)
        missing_indices = requested_indices - completed_indices
        repair_records = sample_records_from_dir(output_dir, allowed_indices=missing_indices)
        if repair_records:
            append_records_to_jsonl(output_path, repair_records)
            completed_indices |= set(repair_records.keys())
            print(f"Resume repaired aggregate output with {len(repair_records)} per-sample records.")
        completed_requested_indices = completed_indices & requested_indices
        if completed_requested_indices:
            data = [sample for sample in data if str(sample["dataset_index"]) not in completed_indices]
            print(
                "Resume enabled: found "
                f"{len(completed_requested_indices)} completed samples in the requested range; {len(data)} samples remain."
            )
        ensure_trailing_newline(output_path)

    if not data:
        print("No pending samples. Nothing to run.")
        raise SystemExit(0)

    solver = Solver(config=config)
    total_batches = max(1, (len(data) + config.batch_size - 1) // config.batch_size)
    output_mode = "a" if args.resume else "w"

    with output_path.open(output_mode, encoding="utf-8") as writer:
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
