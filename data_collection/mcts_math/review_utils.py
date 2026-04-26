from __future__ import annotations

import ast
import json
import os
import re
import tempfile
from subprocess import PIPE, Popen, TimeoutExpired
from typing import Any, Dict, List, Tuple

import psutil

from mcts_math.axiom_scoring import (
    AXIOM_SCALE_TEXT,
    axiom_functionally_correct,
    axiom_grade_from_codecritic,
    axiom_scalar_score,
    axiom_verdict,
    grade_alignment,
    parse_axiom_grade,
)


DEFAULT_DIMENSION_RUBRIC: Dict[str, str] = {
    "Correctness Verification": "Judge whether the code satisfies the task requirements and whether there are concrete logic bugs.",
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


def evaluate_assertion(code: str, assertion: str, timeout_seconds: float = 3) -> bool:
    candidate_code = _normalize_candidate_code(code)
    combined = f"{candidate_code}\n\n{assertion}\n"
    _, stderr, returncode = _run_python(combined, timeout_seconds=timeout_seconds)
    return returncode == 0 and (stderr or "").find("AssertionError") == -1


def compute_pass_rate(
    code: str,
    assertions: List[str],
    max_assertions: int | None = None,
    timeout_seconds: float = 3,
) -> float:
    if not assertions:
        return 0.0
    selected_assertions = assertions
    if max_assertions is not None and max_assertions > 0:
        selected_assertions = assertions[:max_assertions]
    passed = sum(1 for assertion in selected_assertions if evaluate_assertion(code, assertion, timeout_seconds=timeout_seconds))
    return passed / len(selected_assertions)


def score_to_verdict(score: float) -> str:
    if score > 5:
        score = score / 20.0
    if score >= 4:
        return "accept"
    if score >= 3:
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


def _candidate_function_name(code: str) -> str | None:
    try:
        tree = ast.parse(_normalize_candidate_code(code))
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None


def _safe_literal_node(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _safe_literal_node(node.operand)
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_safe_literal_node(element) for element in node.elts)
    return False


def _safe_candidate_call(call_text: str, function_name: str) -> str | None:
    try:
        parsed = ast.parse(call_text, mode="eval")
    except SyntaxError:
        return None
    call = parsed.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None
    if call.func.id != function_name or call.keywords:
        return None
    if not all(_safe_literal_node(argument) for argument in call.args):
        return None
    return ast.unparse(call)


def _run_candidate_call(code: str, call_text: str) -> Dict[str, Any]:
    candidate_code = _normalize_candidate_code(code)
    probe = (
        f"{candidate_code}\n\n"
        "try:\n"
        f"    __mcts_review_result = {call_text}\n"
        "    print('RESULT:' + repr(__mcts_review_result))\n"
        "except Exception as exc:\n"
        "    print('EXCEPTION:' + type(exc).__name__ + ':' + str(exc))\n"
    )
    stdout, stderr, returncode = _run_python(probe)
    stdout = stdout or ""
    if returncode != 0 and not stdout.startswith(("RESULT:", "EXCEPTION:")):
        return {"status": "error", "stderr": stderr}
    if stdout.startswith("RESULT:"):
        return {"status": "result", "value": stdout[len("RESULT:"):].strip()}
    if stdout.startswith("EXCEPTION:"):
        payload = stdout[len("EXCEPTION:"):].strip()
        exc_type, _, message = payload.partition(":")
        return {"status": "exception", "exception": exc_type, "message": message}
    return {"status": "unknown", "stdout": stdout, "stderr": stderr}


def _literal_equal(actual_repr: str, expected_text: str) -> bool:
    try:
        actual = ast.literal_eval(actual_repr)
    except (SyntaxError, ValueError):
        actual = actual_repr
    try:
        expected = ast.literal_eval(expected_text)
    except (SyntaxError, ValueError):
        expected = expected_text.strip().strip("`")
    return actual == expected


def _test_oracle_outputs(sample: Dict[str, Any], function_name: str) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    for assertion in sample.get("tests", []):
        try:
            tree = ast.parse(assertion)
        except SyntaxError:
            continue
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
            continue
        test = tree.body[0].test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        if len(test.comparators) != 1:
            continue
        call = test.left
        expected = test.comparators[0]
        if not isinstance(call, ast.Call):
            continue
        call_text = _safe_candidate_call(ast.unparse(call), function_name)
        if call_text is None or not _safe_literal_node(expected):
            continue
        outputs[call_text] = ast.unparse(expected)
    return outputs


def _validate_modulo_claims(text: str) -> List[Dict[str, Any]]:
    checks = []
    pattern = re.compile(r"(?P<expr>-?\d+\s*%\s*-?\d+)\s*(?:=|==|returns?)\s*(?P<expected>-?\d+(?:\.\d+)?)")
    for match in pattern.finditer(text):
        expr = match.group("expr")
        expected = match.group("expected")
        try:
            actual = eval(expr, {"__builtins__": {}}, {})
        except Exception as exc:
            checks.append({"claim": match.group(0), "supported": False, "actual": type(exc).__name__})
            continue
        supported = _literal_equal(repr(actual), expected)
        checks.append({"claim": match.group(0), "supported": supported, "actual": repr(actual)})
    return checks


def _validate_call_instead_of_claims(text: str, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    function_name = _candidate_function_name(sample["candidate_code"])
    if not function_name:
        return []
    checks = []
    oracle_outputs = _test_oracle_outputs(sample, function_name)
    value_pattern = r"-?\d+(?:\.\d+)?|True|False|None|'[^']*'|\"[^\"]*\""
    pattern = re.compile(
        rf"(?P<call>\b{re.escape(function_name)}\([^)]*\))\s+"
        rf"(?:returns|return|yields|yield|=|==)\s*(?P<actual>{value_pattern})\s+"
        rf"(?:instead of|rather than)\s*(?P<expected>{value_pattern})"
    )
    for match in pattern.finditer(text):
        call_text = _safe_candidate_call(match.group("call"), function_name)
        if call_text is None:
            continue
        actual = _run_candidate_call(sample["candidate_code"], call_text)
        actual_supported = actual.get("status") == "result" and _literal_equal(actual.get("value", ""), match.group("actual"))
        oracle_expected = oracle_outputs.get(call_text)
        if oracle_expected is None:
            if actual_supported:
                continue
            supported = False
        else:
            supported = actual_supported and _literal_equal(oracle_expected, match.group("expected"))
        checks.append(
            {
                "claim": match.group(0),
                "supported": supported,
                "actual": actual,
                "oracle_expected": oracle_expected,
            }
        )
    return checks


def _validate_call_exception_claims(text: str, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    function_name = _candidate_function_name(sample["candidate_code"])
    if not function_name:
        return []
    checks = []
    pattern = re.compile(
        rf"(?P<call>\b{re.escape(function_name)}\([^)]*\))\s+"
        r"(?:raises|raise|will raise|throws|throw|will throw)\s+"
        r"(?:an?\s+)?(?P<expected>\w*Error)"
    )
    for match in pattern.finditer(text):
        call_text = _safe_candidate_call(match.group("call"), function_name)
        if call_text is None:
            continue
        expected = match.group("expected")
        actual = _run_candidate_call(sample["candidate_code"], call_text)
        supported = actual.get("status") == "exception" and actual.get("exception") == expected
        checks.append({"claim": match.group(0), "supported": supported, "actual": actual})
    return checks


def _validate_call_return_claims(text: str, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    function_name = _candidate_function_name(sample["candidate_code"])
    if not function_name:
        return []
    checks = []
    value_pattern = r"-?\d+(?:\.\d+)?|True|False|None|'[^']*'|\"[^\"]*\""
    pattern = re.compile(
        rf"(?P<call>\b{re.escape(function_name)}\([^)]*\))\s+"
        rf"(?:returns|return|=|==)\s*(?P<expected>{value_pattern})"
    )
    for match in pattern.finditer(text):
        call_text = _safe_candidate_call(match.group("call"), function_name)
        if call_text is None:
            continue
        expected = match.group("expected")
        actual = _run_candidate_call(sample["candidate_code"], call_text)
        supported = actual.get("status") == "result" and _literal_equal(actual.get("value", ""), expected)
        checks.append({"claim": match.group(0), "supported": supported, "actual": actual})
    return checks


def _identifier_usage_counts(code: str, name: str) -> Dict[str, int] | None:
    try:
        tree = ast.parse(_normalize_candidate_code(code))
    except SyntaxError:
        return None
    counts = {"load": 0, "store": 0, "param": 0, "delete": 0}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            if isinstance(node.ctx, ast.Load):
                counts["load"] += 1
            elif isinstance(node.ctx, ast.Store):
                counts["store"] += 1
            elif isinstance(node.ctx, ast.Del):
                counts["delete"] += 1
        elif isinstance(node, ast.arg) and node.arg == name:
            counts["param"] += 1
    return counts


def _validate_unused_identifier_claims(text: str, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks = []
    pattern = re.compile(
        r"(?:variable|parameter|argument|identifier|name|input|target(?:\s+\w+)?)?\s*"
        r"[`'\"](?P<name>[A-Za-z_]\w*)[`'\"]\s+"
        r"(?:is\s+|was\s+|being\s+)?"
        r"(?P<claim>unused|not\s+used|never\s+used|ignored|not\s+referenced|never\s+referenced)",
        flags=re.IGNORECASE,
    )
    seen: set[tuple[str, str]] = set()
    for match in pattern.finditer(text):
        name = match.group("name")
        key = (name, match.group(0))
        if key in seen:
            continue
        seen.add(key)
        counts = _identifier_usage_counts(sample["candidate_code"], name)
        if counts is None:
            continue
        defined = counts["store"] + counts["param"] > 0
        supported = defined and counts["load"] == 0
        checks.append(
            {
                "kind": "unused_identifier",
                "claim": match.group(0).strip(),
                "supported": supported,
                "actual": {
                    "identifier": name,
                    "usage": counts,
                    "reason": "identifier_unused" if supported else "identifier_used_or_not_defined",
                },
            }
        )
    return checks


def _validate_provided_test_failure_claim(parsed: Dict[str, Any], sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence_type = str(parsed.get("evidence_type", "")).strip()
    if evidence_type != "provided_test_failure":
        return []

    assertions = list(sample.get("tests") or [])
    if not assertions:
        return [
            {
                "kind": "provided_test_failure",
                "claim": "evidence_type=provided_test_failure",
                "supported": False,
                "actual": {"reason": "no_listed_executable_tests"},
            }
        ]

    objective = sample.get("objective") or {}
    pass_rate = objective.get("full_test_pass_rate")
    if isinstance(pass_rate, (int, float)):
        if pass_rate >= 0.999:
            return [
                {
                    "kind": "provided_test_failure",
                    "claim": "evidence_type=provided_test_failure",
                    "supported": False,
                    "actual": {"full_test_pass_rate": pass_rate, "reason": "all_listed_tests_pass"},
                }
            ]
        return [
            {
                "kind": "provided_test_failure",
                "claim": "evidence_type=provided_test_failure",
                "supported": True,
                "actual": {"full_test_pass_rate": pass_rate},
            }
        ]

    checked = []
    for assertion in assertions[:8]:
        checked.append({"assertion": assertion, "passed": evaluate_assertion(sample["candidate_code"], assertion)})
    failing = [item for item in checked if not item["passed"]]
    return [
        {
            "kind": "provided_test_failure",
            "claim": "evidence_type=provided_test_failure",
            "supported": bool(failing),
            "actual": {"checked": checked, "failing_count": len(failing), "truncated": len(assertions) > len(checked)},
        }
    ]


def validate_review_evidence(parsed: Dict[str, Any], sample: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    evidence = parsed.get("evidence")
    evidence_items = evidence if isinstance(evidence, list) else []
    evidence_text = "\n".join(str(item) for item in evidence_items)
    summary_text = str(parsed.get("summary", ""))
    claim_text = f"{summary_text}\n{evidence_text}"

    if not evidence_items:
        return 0.0, {"evidence_count": 0, "fact_checks": [], "false_claim_count": 0, "true_claim_count": 0}

    fact_checks = []
    fact_checks.extend(_validate_modulo_claims(claim_text))
    fact_checks.extend(_validate_call_instead_of_claims(claim_text, sample))
    fact_checks.extend(_validate_call_exception_claims(claim_text, sample))
    fact_checks.extend(_validate_call_return_claims(claim_text, sample))
    fact_checks.extend(_validate_unused_identifier_claims(claim_text, sample))
    fact_checks.extend(_validate_provided_test_failure_claim(parsed, sample))

    true_claim_count = sum(1 for check in fact_checks if check["supported"])
    false_claim_count = sum(1 for check in fact_checks if not check["supported"])
    if false_claim_count:
        evidence_alignment = true_claim_count / (true_claim_count + false_claim_count)
    else:
        evidence_alignment = 1.0

    return evidence_alignment, {
        "evidence_count": len(evidence_items),
        "fact_checks": fact_checks,
        "false_claim_count": false_claim_count,
        "true_claim_count": true_claim_count,
    }


def parse_review_payload(text: str) -> Dict[str, Any] | None:
    payload = text.strip()
    if payload.startswith("<review>"):
        payload = payload[len("<review>"):].lstrip()
    if payload.endswith("</review>"):
        payload = payload[: -len("</review>")].rstrip()
    if payload.startswith("```"):
        payload = payload.split("\n", 1)[1] if "\n" in payload else payload.strip("`")
        if payload.endswith("```"):
            payload = payload.rsplit("```", 1)[0].rstrip()
    if "{" in payload and "}" in payload:
        payload = payload[payload.find("{") : payload.rfind("}") + 1]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def review_semantic_signature(final_answer: str) -> Tuple[Any, ...] | None:
    parsed = parse_review_payload(final_answer)
    if parsed is None:
        return None
    grade = parse_axiom_grade(parsed)
    if grade is None:
        return None
    return (
        grade,
        str(parsed.get("verdict", "")).strip().lower(),
        bool(parsed.get("functional_correctness")),
        str(parsed.get("repair_effort", "")).strip().lower(),
        str(parsed.get("evidence_type", "")).strip().lower(),
    )


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


def build_axiom_target_grade(sample: Dict[str, Any]) -> int:
    mapped = axiom_grade_from_codecritic(sample.get("correctness_label"), sample.get("overall_score"))
    has_executable_tests = bool(sample.get("tests"))
    pass_rate = sample.get("objective", {}).get("full_test_pass_rate", 0.0)
    if mapped is not None:
        if has_executable_tests and pass_rate >= 0.999 and mapped < 3:
            return 3
        if has_executable_tests and pass_rate <= 0.0 and mapped >= 3:
            return 2
        return mapped
    if pass_rate >= 0.999:
        return 4
    if pass_rate > 0.0:
        return 2
    return 1


def compute_review_reward(target_dimension: str, final_answer: str, sample: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    parsed = parse_review_payload(final_answer)
    if parsed is None:
        return -1.0, {"error": "invalid_review_json"}

    predicted_dimension = str(parsed.get("dimension", "")).strip()
    predicted_grade = parse_axiom_grade(parsed)
    if predicted_grade is None:
        return -1.0, {"error": "missing_or_invalid_axiom_grade", "parsed": parsed}
    predicted_score = axiom_scalar_score(predicted_grade)

    predicted_verdict = str(parsed.get("verdict", "")).strip()
    evidence_type = str(parsed.get("evidence_type", "")).strip()
    evidence_alignment, evidence_details = validate_review_evidence(parsed, sample)

    target_grade = int(sample.get("axiom_target_grade", build_axiom_target_grade(sample)))
    target_score = axiom_scalar_score(target_grade)
    expected_verdict = axiom_verdict(target_grade)
    grade_distance = abs(predicted_grade - target_grade)
    score_alignment = grade_alignment(predicted_grade, target_grade)
    dimension_alignment = 1.0 if not predicted_dimension or predicted_dimension == target_dimension else 0.0
    verdict_alignment = _verdict_distance(expected_verdict, predicted_verdict)
    functionality_alignment = 1.0 if axiom_functionally_correct(predicted_grade) == axiom_functionally_correct(target_grade) else 0.0

    hard_alignment = functionality_alignment
    if target_dimension == "Correctness Verification":
        reward_01 = (
            0.45 * score_alignment
            + 0.25 * functionality_alignment
            + 0.10 * dimension_alignment
            + 0.10 * verdict_alignment
            + 0.10 * evidence_alignment
        )
    else:
        reward_01 = (
            0.50 * score_alignment
            + 0.20 * functionality_alignment
            + 0.10 * dimension_alignment
            + 0.10 * verdict_alignment
            + 0.10 * evidence_alignment
        )

    reward_caps: List[Tuple[str, float]] = []
    unsupported_provided_test_failure = any(
        check.get("kind") == "provided_test_failure" and not check.get("supported")
        for check in evidence_details.get("fact_checks", [])
    )
    unsupported_unused_identifier = any(
        check.get("kind") == "unused_identifier" and not check.get("supported")
        for check in evidence_details.get("fact_checks", [])
    )
    if evidence_details["false_claim_count"] > 0:
        false_claim_cap = 0.70 if evidence_details["true_claim_count"] > 0 else 0.55
        reward_caps.append(("false_executable_evidence_claim", false_claim_cap))
    if unsupported_provided_test_failure:
        if predicted_grade < 3 or axiom_functionally_correct(predicted_grade) != axiom_functionally_correct(target_grade):
            reward_caps.append(("unsupported_provided_test_failure_evidence", 0.15))
        else:
            reward_caps.append(("unsupported_provided_test_failure_evidence_correct_boundary", 0.55))
    if unsupported_unused_identifier:
        reward_caps.append(("unsupported_unused_identifier_evidence", 0.25))

    concrete_evidence_types = {"provided_test_failure", "deduced_counterexample", "static_logic_contradiction"}
    has_listed_tests = bool(sample.get("tests"))
    if predicted_grade < 3 and evidence_type not in concrete_evidence_types:
        reward_caps.append(("low_grade_without_concrete_evidence_type", 0.55))
    if predicted_grade < 3 and evidence_type == "provided_test_failure" and not has_listed_tests:
        reward_caps.append(("test_failure_claim_without_available_tests", 0.35))
    if predicted_grade < 3 and evidence_type == "uncertain":
        reward_caps.append(("uncertain_evidence_cannot_support_low_grade", 0.45))

    correctness_label = str(sample.get("correctness_label") or "").lower()
    if (
        target_dimension == "Correctness Verification"
        and correctness_label == "correct"
        and sample["objective"]["full_test_pass_rate"] >= 0.999
        and predicted_grade < 3
    ):
        reward_caps.append(("functional_correctness_boundary_conflict", 0.55))

    if axiom_functionally_correct(predicted_grade) != axiom_functionally_correct(target_grade):
        reward_caps.append(("axiom_functionality_boundary_mismatch", 0.35))

    if grade_distance >= 2:
        ordinal_cap = max(0.10, 0.45 - 0.15 * (grade_distance - 2))
        reward_caps.append((f"axiom_ordinal_grade_distance_{grade_distance}", ordinal_cap))

    if reward_caps:
        reward_01 = min(reward_01, *(cap for _, cap in reward_caps))

    reward = round(reward_01 * 2.0 - 1.0, 4)
    details = {
        "parsed": parsed,
        "target_dimension": target_dimension,
        "target_axiom_grade": target_grade,
        "predicted_axiom_grade": predicted_grade,
        "target_score": target_score,
        "predicted_score": predicted_score,
        "grade_distance": grade_distance,
        "score_scale": "axiom_0_5_scalar_0_100",
        "score_semantics": AXIOM_SCALE_TEXT,
        "expected_verdict": expected_verdict,
        "score_alignment": round(score_alignment, 4),
        "dimension_alignment": round(dimension_alignment, 4),
        "functionality_alignment": round(functionality_alignment, 4),
        "verdict_alignment": round(verdict_alignment, 4),
        "evidence_type": evidence_type,
        "evidence_alignment": round(evidence_alignment, 4),
        "evidence_details": evidence_details,
        "reward_01": round(reward_01, 4),
    }
    if reward_caps:
        details["reward_caps"] = [{"reason": reason, "cap": cap} for reason, cap in reward_caps]
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


def prepare_codecriticbench_sample(
    raw_sample: Dict[str, Any],
    dataset_index: int | None = None,
    max_objective_assertions_per_split: int | None = None,
    assertion_timeout_seconds: float = 3,
) -> Dict[str, Any]:
    public_assertions = list(raw_sample.get("public_test", {}).get("input", []) or [])
    private_assertions = list(raw_sample.get("private_test", {}).get("input", []) or [])
    all_assertions = public_assertions + private_assertions
    candidate_code = raw_sample["answer"]

    raw_reference_scores = {
        dimension: score
        for dimension, score in zip(raw_sample["checklist_dimensions"], raw_sample["checklist_scores"])
    }
    correctness_score = float(raw_reference_scores.get("Correctness Verification", raw_sample.get("score") or 0))
    reference_scores = {"Correctness Verification": correctness_score}
    dimension_rubrics = {}
    for dimension, checklist in zip(raw_sample["checklist_dimensions"], raw_sample["checklists"]):
        if dimension != "Correctness Verification":
            continue
        default_rubric = DEFAULT_DIMENSION_RUBRIC.get(dimension, "")
        dimension_rubrics[dimension] = f"{default_rubric}\nReference checklist item: {checklist}".strip()
    if "Correctness Verification" not in dimension_rubrics:
        dimension_rubrics["Correctness Verification"] = DEFAULT_DIMENSION_RUBRIC["Correctness Verification"]

    sample = {
        "dataset_index": dataset_index,
        "problem": raw_sample["question"],
        "candidate_code": candidate_code,
        "code_language": raw_sample.get("lang") or raw_sample.get("language") or "python",
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
        "public_test_pass_rate": round(
            compute_pass_rate(
                candidate_code,
                public_assertions,
                max_assertions=max_objective_assertions_per_split,
                timeout_seconds=assertion_timeout_seconds,
            ),
            4,
        )
        if public_assertions
        else 0.0,
        "private_test_pass_rate": round(
            compute_pass_rate(
                candidate_code,
                private_assertions,
                max_assertions=max_objective_assertions_per_split,
                timeout_seconds=assertion_timeout_seconds,
            ),
            4,
        )
        if private_assertions
        else 0.0,
        "full_test_pass_rate": round(
            compute_pass_rate(
                candidate_code,
                all_assertions,
                max_assertions=(max_objective_assertions_per_split * 2)
                if max_objective_assertions_per_split is not None and max_objective_assertions_per_split > 0
                else None,
                timeout_seconds=assertion_timeout_seconds,
            ),
            4,
        )
        if all_assertions
        else 0.0,
    }
    if max_objective_assertions_per_split is not None and max_objective_assertions_per_split > 0:
        sample["objective"]["test_eval_cap_per_split"] = max_objective_assertions_per_split
        sample["objective"]["assertion_timeout_seconds"] = assertion_timeout_seconds
    sample["dimension_target_scores"] = build_dimension_target_scores(sample)
    sample["axiom_target_grade"] = build_axiom_target_grade(sample)
    sample["axiom_target_score"] = axiom_scalar_score(sample["axiom_target_grade"])
    sample["question"] = build_review_question(sample)
    return sample


def prepare_prebuilt_review_sample(raw_sample: Dict[str, Any], dataset_index: int | None = None) -> Dict[str, Any]:
    """Load a normalized review-scoring sample produced by preprocessing scripts."""
    sample = dict(raw_sample)
    sample["dataset_index"] = dataset_index if dataset_index is not None else sample.get("dataset_index")
    sample.setdefault("code_language", "python")
    sample.setdefault("difficulty", None)
    sample.setdefault("source", "prepared")
    sample.setdefault("subset", "prepared")
    sample.setdefault("tests", [])
    sample.setdefault("tests_for_prompt", format_public_tests({"tests": sample["tests"]}))
    existing_reference_scores = sample.get("reference_scores") if isinstance(sample.get("reference_scores"), dict) else {}
    correctness_score = float(existing_reference_scores.get("Correctness Verification", sample.get("overall_score", 0) or 0))
    sample["reference_scores"] = {"Correctness Verification": correctness_score}
    existing_rubrics = sample.get("dimension_rubrics") if isinstance(sample.get("dimension_rubrics"), dict) else {}
    sample["dimension_rubrics"] = {
        "Correctness Verification": existing_rubrics.get(
            "Correctness Verification",
            DEFAULT_DIMENSION_RUBRIC["Correctness Verification"],
        )
    }
    sample["dimension_target_scores"] = dict(sample["reference_scores"])
    sample.setdefault("objective", {"public_test_pass_rate": 0.0, "private_test_pass_rate": 0.0, "full_test_pass_rate": 0.0})
    sample.setdefault("overall_score", axiom_scalar_score(sample.get("axiom_target_grade", 0)) / 10.0)
    sample.setdefault("correctness_label", "Correct" if int(sample.get("axiom_target_grade", 0)) >= 3 else "Error")
    sample.setdefault("axiom_target_grade", build_axiom_target_grade(sample))
    sample.setdefault("axiom_target_score", axiom_scalar_score(sample["axiom_target_grade"]))
    sample.setdefault("question", build_review_question(sample))
    return sample


def load_codecriticbench_dataset(path: str, start: int = 0, limit: int | None = None) -> List[Dict[str, Any]]:
    loaded: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start:
                continue
            raw_sample = json.loads(line)
            if raw_sample.get("prepared_review_sample") or (
                "candidate_code" in raw_sample and "reference_scores" in raw_sample
            ):
                loaded.append(prepare_prebuilt_review_sample(raw_sample, dataset_index=index))
                if limit is not None and len(loaded) >= limit:
                    break
                continue
            has_tests = bool(raw_sample.get("public_test", {}).get("input") or raw_sample.get("private_test", {}).get("input"))
            if not raw_sample.get("checklist_dimensions") or "answer" not in raw_sample or not has_tests:
                continue
            loaded.append(prepare_codecriticbench_sample(raw_sample, dataset_index=index))
            if limit is not None and len(loaded) >= limit:
                break
    return loaded
