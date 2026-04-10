from __future__ import annotations

import json
import os
import tempfile
from subprocess import PIPE, Popen, TimeoutExpired
from typing import Any, Dict, List, Tuple

import psutil


DEFAULT_DIMENSION_RUBRIC: Dict[str, str] = {
    "Correctness Verification": "Judge whether the code satisfies the task requirements and whether there are concrete logic bugs.",
    "Time Complexity Optimization": "Judge whether the algorithmic complexity is appropriate and whether obvious performance regressions exist.",
    "Space Complexity Control": "Judge whether the memory usage is reasonable and whether unnecessary allocations are introduced.",
    "Code Readability Enhancement": "Judge naming clarity, local reasoning clarity, and whether the code is easy to inspect.",
    "Robustness Validation": "Judge input validation, boundary handling, exceptional cases, and failure modes.",
    "Algorithm Optimization": "Judge whether the algorithmic design is suitable, not just whether the current code runs.",
    "Comprehensive Testing": "Judge whether the solution is sufficiently exercised by tests and edge cases.",
    "Output Format Compliance": "Judge whether the returned value and observable behavior match the requested format.",
    "Code Style Consistency": "Judge whether naming, formatting, and conventions are consistent with Python norms.",
    "Maintainability": "Judge whether the code structure is easy to extend, debug, and modify safely.",
}


def _kill_process_tree(proc_pid: int) -> None:
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def _run_python(code: str, timeout_seconds: int = 3) -> tuple[str | None, str | None, int]:
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, dir="/tmp") as handle:
            handle.write(code.encode("utf-8"))
            temp_path = handle.name

        temp_dir = os.path.dirname(temp_path)
        proc = Popen(f"cd {temp_dir} ; python {temp_path}", shell=True, stdout=PIPE, stderr=PIPE)
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
            return stdout.decode("utf-8"), stderr.decode("utf-8"), proc.returncode
        except TimeoutExpired:
            _kill_process_tree(proc.pid)
            return None, None, -1
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    except Exception:
        return None, None, -1


def _normalize_candidate_code(code: str) -> str:
    cleaned = code.strip()
    if "```" in cleaned:
        if "```python" in cleaned:
            return cleaned.split("```python", 1)[1].split("```", 1)[0].strip()
        return cleaned.split("```", 1)[1].split("```", 1)[0].strip()
    return cleaned


def evaluate_assertion(code: str, assertion: str) -> bool:
    candidate_code = _normalize_candidate_code(code)
    combined = f"{candidate_code}\n\n{assertion}\n"
    _, stderr, returncode = _run_python(combined)
    return returncode == 0 and (stderr or "").find("AssertionError") == -1


def compute_pass_rate(code: str, assertions: List[str]) -> float:
    if not assertions:
        return 0.0
    passed = sum(1 for assertion in assertions if evaluate_assertion(code, assertion))
    return passed / len(assertions)


def score_to_verdict(score: float) -> str:
    if score >= 8:
        return "accept"
    if score >= 5:
        return "minor_issue"
    return "major_issue"


def _verdict_distance(expected: str, predicted: str) -> float:
    ordering = {"major_issue": 0, "minor_issue": 1, "accept": 2}
    if expected not in ordering or predicted not in ordering:
        return 0.0
    distance = abs(ordering[expected] - ordering[predicted])
    if distance == 0:
        return 1.0
    if distance == 1:
        return 0.5
    return 0.0


def parse_review_payload(text: str) -> Dict[str, Any] | None:
    payload = text.strip()
    if payload.endswith("</review>"):
        payload = payload[: -len("</review>")].rstrip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def build_dimension_target_scores(sample: Dict[str, Any]) -> Dict[str, float]:
    targets: Dict[str, float] = {}
    pass_rate = sample["objective"]["full_test_pass_rate"]
    hard_correctness_score = 10.0 * pass_rate
    for dimension, reference_score in sample["reference_scores"].items():
        if dimension == "Correctness Verification":
            targets[dimension] = round(0.6 * reference_score + 0.4 * hard_correctness_score, 2)
        else:
            targets[dimension] = float(reference_score)
    return targets


def compute_review_reward(target_dimension: str, final_answer: str, sample: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    parsed = parse_review_payload(final_answer)
    if parsed is None:
        return -1.0, {"error": "invalid_review_json"}

    predicted_dimension = str(parsed.get("dimension", "")).strip()
    predicted_score_raw = parsed.get("score")
    try:
        predicted_score = float(predicted_score_raw)
    except (TypeError, ValueError):
        return -1.0, {"error": "missing_or_invalid_score", "parsed": parsed}

    predicted_verdict = str(parsed.get("verdict", "")).strip()
    evidence = parsed.get("evidence")
    evidence_ok = isinstance(evidence, list) and len(evidence) > 0

    target_score = sample["dimension_target_scores"][target_dimension]
    expected_verdict = score_to_verdict(target_score)
    score_alignment = max(0.0, 1.0 - abs(predicted_score - target_score) / 9.0)
    dimension_alignment = 1.0 if predicted_dimension == target_dimension else 0.0
    verdict_alignment = _verdict_distance(expected_verdict, predicted_verdict)
    evidence_alignment = 1.0 if evidence_ok else 0.0

    hard_alignment = None
    if target_dimension == "Correctness Verification":
        pass_rate = sample["objective"]["full_test_pass_rate"]
        hard_expected = "accept" if pass_rate >= 0.999 else "major_issue"
        hard_alignment = _verdict_distance(hard_expected, predicted_verdict)
        reward_01 = (
            0.45 * score_alignment
            + 0.20 * dimension_alignment
            + 0.20 * verdict_alignment
            + 0.10 * hard_alignment
            + 0.05 * evidence_alignment
        )
    else:
        reward_01 = (
            0.60 * score_alignment
            + 0.20 * dimension_alignment
            + 0.15 * verdict_alignment
            + 0.05 * evidence_alignment
        )

    reward = round(reward_01 * 2.0 - 1.0, 4)
    details = {
        "parsed": parsed,
        "target_dimension": target_dimension,
        "target_score": target_score,
        "expected_verdict": expected_verdict,
        "score_alignment": round(score_alignment, 4),
        "dimension_alignment": round(dimension_alignment, 4),
        "verdict_alignment": round(verdict_alignment, 4),
        "evidence_alignment": round(evidence_alignment, 4),
        "reward_01": round(reward_01, 4),
    }
    if hard_alignment is not None:
        details["hard_alignment"] = round(hard_alignment, 4)
        details["full_test_pass_rate"] = sample["objective"]["full_test_pass_rate"]
    return reward, details


def format_public_tests(sample: Dict[str, Any], max_tests: int = 5) -> str:
    assertions = sample.get("tests", [])
    if not assertions:
        return "No tests are available."
    snippet = assertions[:max_tests]
    body = "\n".join(snippet)
    if len(assertions) > max_tests:
        body += f"\n... ({len(assertions) - max_tests} more assertions omitted)"
    return body


def build_review_question(sample: Dict[str, Any]) -> str:
    return (
        "Review the candidate code against the original task. "
        "Keep the reasoning grounded in the task requirements and the observed code behavior.\n\n"
        f"Problem:\n{sample['problem']}"
    )


def prepare_codecriticbench_sample(raw_sample: Dict[str, Any]) -> Dict[str, Any]:
    public_assertions = list(raw_sample.get("public_test", {}).get("input", []) or [])
    private_assertions = list(raw_sample.get("private_test", {}).get("input", []) or [])
    all_assertions = public_assertions + private_assertions
    candidate_code = raw_sample["answer"]

    reference_scores = {
        dimension: score
        for dimension, score in zip(raw_sample["checklist_dimensions"], raw_sample["checklist_scores"])
    }
    dimension_rubrics = {}
    for dimension, checklist in zip(raw_sample["checklist_dimensions"], raw_sample["checklists"]):
        default_rubric = DEFAULT_DIMENSION_RUBRIC.get(dimension, "")
        dimension_rubrics[dimension] = f"{default_rubric}\nReference checklist item: {checklist}".strip()

    sample = {
        "problem": raw_sample["question"],
        "candidate_code": candidate_code,
        "tests": all_assertions,
        "tests_for_prompt": format_public_tests({"tests": public_assertions}),
        "difficulty": raw_sample.get("difficulty"),
        "source": raw_sample.get("source"),
        "subset": raw_sample.get("subset"),
        "reference_scores": reference_scores,
        "dimension_rubrics": dimension_rubrics,
        "overall_score": raw_sample.get("score"),
        "correctness_label": raw_sample.get("correctness"),
    }
    sample["objective"] = {
        "public_test_pass_rate": round(compute_pass_rate(candidate_code, public_assertions), 4) if public_assertions else 0.0,
        "private_test_pass_rate": round(compute_pass_rate(candidate_code, private_assertions), 4) if private_assertions else 0.0,
        "full_test_pass_rate": round(compute_pass_rate(candidate_code, all_assertions), 4) if all_assertions else 0.0,
    }
    sample["dimension_target_scores"] = build_dimension_target_scores(sample)
    sample["question"] = build_review_question(sample)
    return sample


def load_codecriticbench_dataset(path: str, start: int = 0, limit: int | None = None) -> List[Dict[str, Any]]:
    loaded: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start:
                continue
            raw_sample = json.loads(line)
            has_tests = bool(raw_sample.get("public_test", {}).get("input") or raw_sample.get("private_test", {}).get("input"))
            if not raw_sample.get("checklist_dimensions") or "answer" not in raw_sample or not has_tests:
                continue
            loaded.append(prepare_codecriticbench_sample(raw_sample))
            if limit is not None and len(loaded) >= limit:
                break
    return loaded
