from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from magicoder.axiom_scoring import axiom_grade_from_codecritic, axiom_interval_from_binary, axiom_scalar_score, clamp_axiom_grade
from magicoder.preprocess_score_datasets import codejudge_task, normalize_code_block


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    yield payload


def load_arrow(path: Path):
    from datasets import Dataset

    return Dataset.from_file(str(path))


def used_training_ids(path: Path | None) -> set[str]:
    used: set[str] = set()
    if not path or not path.exists():
        return used
    for row in iter_jsonl(path):
        source = row.get("source")
        dataset_index = row.get("dataset_index")
        if source and dataset_index is not None:
            used.add(f"{source}:{dataset_index}")
    return used


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def balanced_take(groups: dict[Any, list[dict[str, Any]]], per_group: int, rng: random.Random) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        rows = list(groups[key])
        rng.shuffle(rows)
        selected.extend(rows[:per_group])
    rng.shuffle(selected)
    return selected


def normalize_record(
    *,
    problem: str,
    candidate_code: str,
    source: str,
    subset: str,
    dataset_index: str,
    language: str = "unknown",
    grade: int | None = None,
    interval: tuple[int, int] | None = None,
    label_type: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if grade is not None:
        grade = clamp_axiom_grade(grade)
    payload: dict[str, Any] = {
        "problem": str(problem or ""),
        "candidate_code": normalize_code_block(candidate_code),
        "tests": [],
        "reference_scores": {},
        "source": source,
        "subset": subset,
        "dataset_index": dataset_index,
        "language": language or "unknown",
        "label_type": label_type,
        "axiom_target_grade": grade,
        "axiom_target_score": axiom_scalar_score(grade) if grade is not None else None,
    }
    if interval is not None:
        lower, upper = sorted((clamp_axiom_grade(interval[0]), clamp_axiom_grade(interval[1])))
        payload["axiom_target_interval"] = [lower, upper]
    elif grade is not None:
        payload["axiom_target_interval"] = [grade, grade]
    if metadata:
        payload.update(metadata)
    return payload


def build_axiom(path: Path, used: set[str], per_grade: int, rng: random.Random) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for file_path in sorted(path.glob("*.jsonl")):
        subset = file_path.stem
        for row_idx, row in enumerate(iter_jsonl(file_path)):
            dataset_index = f"{subset}:{row_idx}"
            if f"axiom:{dataset_index}" in used:
                continue
            grade = clamp_axiom_grade(row.get("score", 0))
            groups[grade].append(
                normalize_record(
                    problem=row.get("inst", ""),
                    candidate_code=row.get("code", ""),
                    source="axiom",
                    subset=subset,
                    dataset_index=dataset_index,
                    language=str(row.get("lang") or "unknown"),
                    grade=grade,
                    label_type="exact",
                )
            )
    return balanced_take(groups, per_grade, rng)


def build_codecritic(path: Path, used: set[str], per_grade: int, rng: random.Random) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row_idx, row in enumerate(iter_jsonl(path)):
        subset = str(row.get("subset") or row.get("source") or "default")
        source_name = str(row.get("source") or "codecritic")
        dataset_index = f"{source_name}:{subset}:{row_idx}"
        if f"codecritic:{dataset_index}" in used:
            continue
        grade = axiom_grade_from_codecritic(row.get("correctness"), row.get("score"))
        if grade is None:
            continue
        tests = list((row.get("public_test") or {}).get("input", []) or []) + list((row.get("private_test") or {}).get("input", []) or [])
        record = normalize_record(
            problem=row.get("question", ""),
            candidate_code=row.get("answer", ""),
            source="codecritic",
            subset=subset,
            dataset_index=dataset_index,
            language="python",
            grade=grade,
            label_type="exact_mapped",
            metadata={"tests": tests[:5], "metadata": {"raw_score": row.get("score"), "correctness": row.get("correctness")}},
        )
        groups[grade].append(record)
    return balanced_take(groups, per_grade, rng)


def build_diting(path: Path, per_label: int, rng: random.Random) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for file_path in sorted(path.rglob("*.arrow")):
        subset = file_path.name.replace("-train.arrow", "")
        dataset = load_arrow(file_path)
        for row_idx, row in enumerate(dataset):
            try:
                label = int(row.get("label"))
            except (TypeError, ValueError):
                continue
            interval = axiom_interval_from_binary(label)
            if interval is None:
                continue
            groups[label].append(
                normalize_record(
                    problem=row.get("nl", ""),
                    candidate_code=row.get("code", ""),
                    source="code_diting",
                    subset=subset,
                    dataset_index=f"{subset}:{row_idx}",
                    language="python",
                    interval=interval,
                    label_type="interval_correctness",
                    metadata={"metadata": {"label": label, "model": row.get("model")}},
                )
            )
    return balanced_take(groups, per_label, rng)


def build_codejudge(path: Path, pair_count: int, rng: random.Random) -> list[dict[str, Any]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    for file_path in sorted(path.rglob("*.arrow")):
        config = "coderepair" if "/coderepair/" in str(file_path) else "codegen" if "/codegen/" in str(file_path) else "unknown"
        subset = f"{config}:{file_path.stem}"
        dataset = load_arrow(file_path)
        for row_idx, row in enumerate(dataset):
            if not row.get("pos_response") or not row.get("neg_response"):
                continue
            pair_id = f"{subset}:{row.get('question_id', row_idx)}:{row_idx}"
            if pair_id in seen:
                continue
            seen.add(pair_id)
            task = codejudge_task(row)
            if config == "coderepair" and row.get("wrong_code"):
                task += f"\n\nBuggy code to repair:\n{row.get('wrong_code')}"
            common = {"pair_id": pair_id, "metadata": {"config": config, "platform": row.get("platform")}}
            pos = normalize_record(
                problem=task,
                candidate_code=row.get("pos_response"),
                source="codejudgebench",
                subset=subset,
                dataset_index=f"{pair_id}:pos",
                interval=(3, 5),
                label_type="pairwise_interval",
                metadata={**common, "pair_role": "pos"},
            )
            neg = normalize_record(
                problem=task,
                candidate_code=row.get("neg_response"),
                source="codejudgebench",
                subset=subset,
                dataset_index=f"{pair_id}:neg",
                interval=(0, 2),
                label_type="pairwise_interval",
                metadata={**common, "pair_role": "neg"},
            )
            pairs.append((pos, neg))
    rng.shuffle(pairs)
    selected: list[dict[str, Any]] = []
    for pos, neg in pairs[:pair_count]:
        selected.extend([pos, neg])
    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed multi-dataset review evaluation manifests.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_data", type=Path, default=Path("model_training/review_mcts_train_data/report_static_mcts_valueonly_20260420.jsonl"))
    parser.add_argument("--codecriticbench", type=Path, default=Path("datasets/CodeCriticBench/data/CodeCriticBench.jsonl"))
    parser.add_argument("--axiom_dir", type=Path, default=Path("datasets/axiom-llm-judge/axiombench"))
    parser.add_argument("--code_diting_root", type=Path, default=Path("datasets/Code-DiTing"))
    parser.add_argument("--codejudgebench_root", type=Path, default=Path("benchmarks/mattymchen___codejudgebench"))
    parser.add_argument("--per_grade", type=int, default=10)
    parser.add_argument("--per_binary_label", type=int, default=30)
    parser.add_argument("--pair_count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260421)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    used = used_training_ids(args.train_data)
    manifests = {
        "codecritic": build_codecritic(args.codecriticbench, used, args.per_grade, rng),
        "axiom": build_axiom(args.axiom_dir, used, args.per_grade, rng),
        "code_diting": build_diting(args.code_diting_root, args.per_binary_label, rng),
        "codejudgebench": build_codejudge(args.codejudgebench_root, args.pair_count, rng),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"output_dir": str(args.output_dir), "seed": args.seed, "manifests": {}}
    for name, rows in manifests.items():
        output = args.output_dir / f"{name}.jsonl"
        indices = args.output_dir / f"{name}_indices.json"
        write_jsonl(output, rows)
        indices.write_text(json.dumps({"indices": list(range(len(rows)))}, indent=2) + "\n", encoding="utf-8")
        summary["manifests"][name] = {"path": str(output), "indices": str(indices), "count": len(rows)}
    (args.output_dir / "manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
