from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, set_seed

from magicoder.review_evaluator import (
    VALUE_SCORE_KEYS,
    dimensions_for_sample,
    evaluate_dimension,
    fill_sample_metadata,
    load_record,
    sample_from_record,
)
from magicoder.review_policy_value_inference import load_policy, load_value_model, resolve_model_path


def load_indices(args: argparse.Namespace) -> list[int]:
    indices: list[int] = []
    if args.record_indices:
        indices.extend(args.record_indices)
    if args.record_indices_file:
        payload = json.loads(Path(args.record_indices_file).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("indices", [])
        indices.extend(int(item) for item in payload)
    seen = set()
    deduped = []
    for index in indices:
        if index in seen:
            continue
        seen.add(index)
        deduped.append(index)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Run review_evaluator over multiple records while loading models once.")
    parser.add_argument("--policy_model_path", required=True)
    parser.add_argument("--value_model_path")
    parser.add_argument("--share_policy_value_model", action="store_true")
    parser.add_argument("--skip_value_scoring", action="store_true", help="Do not load/use a value model; useful for direct-generation baselines.")
    parser.add_argument("--input_record", required=True)
    parser.add_argument("--record_indices", nargs="*", type=int)
    parser.add_argument("--record_indices_file")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dimensions", nargs="*")
    parser.add_argument("--max_dimensions", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--max_steps", type=int, default=3)
    parser.add_argument("--num_candidates", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--final_max_new_tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--score_key", choices=VALUE_SCORE_KEYS, default="response_mean_value")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--final_only_json", action="store_true", help="Generate only one compact final <review> JSON block; no step reasoning.")
    parser.add_argument("--format_penalty", type=float, default=1.0, help="Final-candidate value penalty when no AXIOM grade can be parsed.")
    parser.add_argument("--low_grade_no_evidence_penalty", type=float, default=0.4, help="Final-candidate value penalty for grades 0-2 without concrete defect evidence.")
    parser.add_argument("--rethink_threshold", type=float, default=-0.2)
    parser.add_argument("--rethink_spread_threshold", type=float, default=0.0)
    parser.add_argument("--max_rethinks", type=int, default=1)
    parser.add_argument("--max_final_retries", type=int, default=1)
    parser.add_argument("--final_temperature", type=float, default=0.0)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    indices = load_indices(args)
    if not indices:
        raise ValueError("No record indices provided.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_model_path = resolve_model_path(args.policy_model_path)
    value_model_path = resolve_model_path(args.value_model_path or args.policy_model_path)
    tokenizer = AutoTokenizer.from_pretrained(policy_model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.skip_value_scoring and (args.share_policy_value_model or value_model_path == policy_model_path):
        value_wrapper = load_value_model(value_model_path, args.device, args.dtype)
        policy_model = value_wrapper.pretrained_model
        policy_model.eval()
        value_model = None
    elif args.skip_value_scoring:
        policy_model = load_policy(policy_model_path, args.device, args.dtype)
        value_model = None
    elif args.share_policy_value_model or value_model_path == policy_model_path:
        value_model = load_value_model(value_model_path, args.device, args.dtype)
        policy_model = value_model.pretrained_model
        policy_model.eval()
    else:
        policy_model = load_policy(policy_model_path, args.device, args.dtype)
        value_model = load_value_model(value_model_path, args.device, args.dtype)

    completed: list[str] = []
    for record_index in indices:
        output_file = output_dir / f"raw_{record_index}_correctness.json"
        if args.skip_existing and output_file.exists():
            completed.append(str(output_file))
            continue

        record = load_record(args.input_record, record_index)
        sample = fill_sample_metadata(sample_from_record(record), record_index)
        dimensions = dimensions_for_sample(sample, args.dimensions)
        if args.max_dimensions > 0:
            dimensions = dimensions[: args.max_dimensions]

        dimension_results = [
            evaluate_dimension(sample, dimension, policy_model, value_model, tokenizer, args)
            for dimension in dimensions
        ]
        result: dict[str, Any] = {
            "input_record": args.input_record,
            "record_index": record_index,
            "source": sample.get("source"),
            "subset": sample.get("subset"),
            "dataset_index": sample.get("dataset_index"),
            "problem": sample.get("problem"),
            "candidate_code": sample.get("candidate_code"),
            "dimensions": dimension_results,
        }
        output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        completed.append(str(output_file))
        print(json.dumps({"completed": str(output_file), "record_index": record_index}, ensure_ascii=False), flush=True)

    print(json.dumps({"output_dir": str(output_dir), "completed_count": len(completed)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
