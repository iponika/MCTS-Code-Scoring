from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from magicoder.axiom_scoring import (
    AXIOM_GRADE_DESCRIPTIONS,
    axiom_functionally_correct,
    axiom_grade_from_codecritic,
    axiom_interval_from_binary,
    axiom_scalar_score,
    axiom_value_target,
    axiom_verdict,
    clamp_axiom_grade,
)


REPAIR_EFFORT_BY_GRADE = {
    5: "none",
    4: "minor_quality",
    3: "major_quality",
    2: "minor_functional",
    1: "major_functional",
    0: "rewrite",
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    yield payload


def truncate_text(text: Any, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "\n... [truncated]"


def normalize_code_block(code: Any) -> str:
    value = str(code or "").strip()
    fenced = re.search(r"```[a-zA-Z0-9_+-]*\s*\n(?P<body>.*?)```", value, flags=re.DOTALL)
    if fenced:
        return fenced.group("body").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return value


def looks_like_code(code: Any) -> bool:
    value = normalize_code_block(code)
    if len(value) < 12:
        return False
    strong_markers = (
        "def ",
        "class ",
        "import ",
        "from ",
        "#include",
        "int main",
        "public ",
        "private ",
        "function ",
        "return ",
        "const ",
        "let ",
        "var ",
        "=>",
        "std::",
        "System.out",
    )
    marker_count = sum(1 for marker in strong_markers if marker in value)
    structural_count = value.count("{") + value.count(";") + value.count("}") + value.count("):")
    prose_prefixes = (
        "an effective method",
        "the function",
        "to solve ",
        "the correct approach",
        "here's how",
        "we need to",
        "this solution",
        "the implementation",
        "the proposed solution",
        "the solution provided",
        "writing a ",
    )
    starts_like_prose = value.lower().startswith(prose_prefixes)
    return marker_count >= 1 and not starts_like_prose


def build_instruction(task: str, code: str, *, language: str = "unknown", extra: str = "") -> str:
    extra_section = f"\n\nAdditional context:\n{extra.strip()}" if extra.strip() else ""
    code = normalize_code_block(code)
    return (
        "Score the candidate code using the AXIOM 0-5 ordinal code-quality scale.\n"
        "The scalar score is derived from the AXIOM grade: grade / 5 * 100.\n"
        "Textual review is auxiliary; the primary target is the stable scalar grade.\n\n"
        f"Language: {language}\n\n"
        f"Task description:\n{truncate_text(task, 6000)}\n\n"
        "Candidate code:\n"
        f"```{language if language != 'unknown' else ''}\n"
        f"{truncate_text(code, 10000)}\n"
        "```"
        f"{extra_section}"
    )


def review_response_for_grade(grade: int, *, source: str, label_type: str) -> str:
    grade = clamp_axiom_grade(grade)
    functional_text = (
        "The implementation is treated as functionally correct under the AXIOM rubric."
        if axiom_functionally_correct(grade)
        else "The implementation is treated as functionally defective under the AXIOM rubric."
    )
    effort_text = {
        5: "No repair effort is expected before deployment.",
        4: "Only minor quality cleanup is expected after functional acceptance.",
        3: "Functionality is accepted, but larger refactoring is expected before deployment.",
        2: "A localized functional fix is expected to make the implementation acceptable.",
        1: "Major functional repair is expected before the implementation can be accepted.",
        0: "The implementation is considered mismatched or rewrite-level incorrect.",
    }[grade]
    payload = {
        "axiom_grade": grade,
        "score": axiom_scalar_score(grade),
        "verdict": axiom_verdict(grade),
        "functional_correctness": axiom_functionally_correct(grade),
        "repair_effort": REPAIR_EFFORT_BY_GRADE[grade],
        "summary": AXIOM_GRADE_DESCRIPTIONS[grade],
        "evidence": [
            functional_text,
            effort_text,
        ],
    }
    return "<review>\n" + json.dumps(payload, ensure_ascii=False) + "\n</review>"


def make_item(
    *,
    instruction: str,
    grade: int | None = None,
    interval: tuple[int, int] | None = None,
    source: str,
    subset: str,
    dataset_index: str,
    label_type: str,
    confidence: float,
    train_lm: bool,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if grade is None and interval is None:
        raise ValueError("Either grade or interval must be provided.")
    if interval is None:
        grade = clamp_axiom_grade(grade if grade is not None else 0)
        interval = (grade, grade)
    lower, upper = sorted((clamp_axiom_grade(interval[0]), clamp_axiom_grade(interval[1])))
    midpoint_grade = clamp_axiom_grade(grade if grade is not None else (lower + upper) / 2.0)
    q_mid = axiom_value_target(midpoint_grade)
    q_min = axiom_value_target(lower)
    q_max = axiom_value_target(upper)
    item = {
        "instruction": instruction,
        "response": [review_response_for_grade(midpoint_grade, source=source, label_type=label_type)],
        "q_value": [q_mid],
        "q_min": [q_min],
        "q_max": [q_max],
        "train_lm": train_lm,
        "value_loss_weight": confidence,
        "lm_loss_weight": confidence if train_lm else 0.0,
        "source": source,
        "subset": subset,
        "dataset_index": dataset_index,
        "label_type": label_type,
        "label_confidence": confidence,
        "target_axiom_grade": midpoint_grade if lower == upper else None,
        "target_axiom_interval": [lower, upper],
        "target_score": axiom_scalar_score(midpoint_grade),
        "value_scale": "axiom_grade_0_5_mapped_to_tanh_-1_1",
    }
    if metadata:
        item["metadata"] = metadata
        for key in ("pair_id", "pair_role"):
            if key in metadata:
                item[key] = metadata[key]
    return item


def load_axiom_items(path: Path, train_lm_exact: bool, limit: int, drop_grade_zero: bool = False) -> Iterable[dict[str, Any]]:
    counts = defaultdict(int)
    files = sorted((path / "axiombench").glob("*.jsonl")) if path.is_dir() else [path]
    for file_path in files:
        subset = file_path.stem
        for row_idx, row in enumerate(iter_jsonl(file_path)):
            if limit > 0 and counts["axiom"] >= limit:
                return
            grade = clamp_axiom_grade(row.get("score", 0))
            if drop_grade_zero and grade == 0:
                continue
            instruction = build_instruction(row.get("inst", ""), row.get("code", ""), language=str(row.get("lang") or "unknown"))
            counts["axiom"] += 1
            yield make_item(
                instruction=instruction,
                grade=grade,
                source="axiom",
                subset=subset,
                dataset_index=f"{subset}:{row_idx}",
                label_type="exact",
                confidence=1.0,
                train_lm=train_lm_exact,
                metadata={"raw_score": row.get("score"), "lang": row.get("lang")},
            )


def load_codecritic_items(path: Path, train_lm_exact: bool, limit: int) -> Iterable[dict[str, Any]]:
    count = 0
    for row_idx, row in enumerate(iter_jsonl(path)):
        if limit > 0 and count >= limit:
            return
        grade = axiom_grade_from_codecritic(row.get("correctness"), row.get("score"))
        if grade is None:
            continue
        candidate_code = normalize_code_block(row.get("answer", ""))
        if not looks_like_code(candidate_code):
            continue
        tests = row.get("public_test") or {}
        test_inputs = tests.get("input") if isinstance(tests, dict) else None
        extra = ""
        if test_inputs:
            extra = "Available public tests:\n" + "\n".join(str(item) for item in test_inputs[:5])
        instruction = build_instruction(row.get("question", ""), candidate_code, language="unknown", extra=extra)
        count += 1
        yield make_item(
            instruction=instruction,
            grade=grade,
            source="codecritic",
            subset=str(row.get("subset") or row.get("source") or "default"),
            dataset_index=f"{row.get('source', 'codecritic')}:{row.get('subset', 'default')}:{row_idx}",
            label_type="exact_mapped",
            confidence=0.9,
            train_lm=train_lm_exact,
            metadata={
                "raw_score": row.get("score"),
                "correctness": row.get("correctness"),
                "checklist_scores": row.get("checklist_scores"),
            },
        )


def load_arrow(path: Path):
    from datasets import Dataset

    return Dataset.from_file(str(path))


def load_diting_items(root: Path, limit: int) -> Iterable[dict[str, Any]]:
    count = 0
    for file_path in sorted(root.rglob("*.arrow")):
        subset = file_path.name.replace("-train.arrow", "")
        dataset = load_arrow(file_path)
        for row_idx, row in enumerate(dataset):
            if limit > 0 and count >= limit:
                return
            interval = axiom_interval_from_binary(row.get("label"))
            if interval is None:
                continue
            instruction = build_instruction(row.get("nl", ""), row.get("code", ""), language="unknown")
            count += 1
            yield make_item(
                instruction=instruction,
                interval=interval,
                source="code_diting",
                subset=subset,
                dataset_index=f"{subset}:{row_idx}",
                label_type="interval_correctness",
                confidence=0.65,
                train_lm=False,
                metadata={"label": row.get("label"), "model": row.get("model")},
            )


def codejudge_task(row: dict[str, Any]) -> str:
    title = row.get("question_title") or ""
    content = row.get("question_content") or row.get("question") or ""
    task = f"{title}\n\n{content}".strip()
    starter = row.get("starter_code")
    if starter:
        task += f"\n\nStarter code:\n{starter}"
    return task


def load_codejudge_interval_items(root: Path, limit: int) -> Iterable[dict[str, Any]]:
    count = 0
    for file_path in sorted(root.rglob("*.arrow")):
        if "/codegen/" in str(file_path):
            config = "codegen"
        elif "/coderepair/" in str(file_path):
            config = "coderepair"
        else:
            config = "unknown"
        subset = f"{config}:{file_path.stem}"
        dataset = load_arrow(file_path)
        for row_idx, row in enumerate(dataset):
            if limit > 0 and count >= limit:
                return
            task = codejudge_task(row)
            if config == "coderepair" and row.get("wrong_code"):
                task += f"\n\nBuggy code to repair:\n{row.get('wrong_code')}"
            pair_id = f"{subset}:{row.get('question_id', row_idx)}:{row_idx}"
            for role, code, interval in (
                ("pos", row.get("pos_response"), (3, 5)),
                ("neg", row.get("neg_response"), (0, 2)),
            ):
                if not code:
                    continue
                if limit > 0 and count >= limit:
                    return
                instruction = build_instruction(task, code, language="unknown")
                count += 1
                yield make_item(
                    instruction=instruction,
                    interval=interval,
                    source="codejudgebench",
                    subset=subset,
                    dataset_index=f"{pair_id}:{role}",
                    label_type="pairwise_interval_proxy",
                    confidence=0.6,
                    train_lm=False,
                    metadata={"pair_id": pair_id, "pair_role": role, "config": config, "platform": row.get("platform")},
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AXIOM-style scalar scoring training data for review value calibration.")
    parser.add_argument("--axiom_dir", type=Path, default=None, help="Path to datasets/axiom-llm-judge or an AXIOM jsonl file.")
    parser.add_argument("--codecriticbench", type=Path, default=None, help="Path to CodeCriticBench.jsonl.")
    parser.add_argument("--code_diting_root", type=Path, default=None, help="Path to datasets/Code-DiTing.")
    parser.add_argument("--codejudgebench_root", type=Path, default=None, help="Path to benchmarks/mattymchen___codejudgebench.")
    parser.add_argument("--include_codejudge_as_intervals", action="store_true", help="Use CodeJudgeBench pos/neg pairs as weak interval value labels.")
    parser.add_argument("--train_lm_exact", action="store_true", help="Also train LM tokens for exact AXIOM/CodeCritic labels. Default trains value only.")
    parser.add_argument("--drop_axiom_grade_zero", action="store_true", help="Skip AXIOM grade-0 rows when they are treated as outliers for the current training objective.")
    parser.add_argument("--limit_per_source", type=int, default=0, help="Maximum rows per source family; 0 keeps all.")
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--disable_shuffle", action="store_true", help="Keep source order; useful when preserving adjacent CodeJudgeBench pairs for pairwise loss.")
    parser.add_argument("--output_file", type=Path, required=True)
    args = parser.parse_args()

    items: list[dict[str, Any]] = []
    if args.axiom_dir:
        items.extend(
            load_axiom_items(
                args.axiom_dir,
                train_lm_exact=args.train_lm_exact,
                limit=args.limit_per_source,
                drop_grade_zero=args.drop_axiom_grade_zero,
            )
        )
    if args.codecriticbench:
        items.extend(load_codecritic_items(args.codecriticbench, train_lm_exact=args.train_lm_exact, limit=args.limit_per_source))
    if args.code_diting_root:
        items.extend(load_diting_items(args.code_diting_root, limit=args.limit_per_source))
    if args.codejudgebench_root and args.include_codejudge_as_intervals:
        items.extend(load_codejudge_interval_items(args.codejudgebench_root, limit=args.limit_per_source))

    if not args.disable_shuffle:
        random.Random(args.shuffle_seed).shuffle(items)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats = defaultdict(int)
    for item in items:
        stats[f"source_{item['source']}"] += 1
        stats[f"label_type_{item['label_type']}"] += 1
        if item.get("train_lm"):
            stats["train_lm_items"] += 1
        else:
            stats["value_only_items"] += 1
    stats["output_items"] = len(items)
    print(json.dumps(dict(sorted(stats.items())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
