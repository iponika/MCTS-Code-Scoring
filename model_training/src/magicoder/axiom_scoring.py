from __future__ import annotations

from typing import Any


AXIOM_GRADE_DESCRIPTIONS: dict[int, str] = {
    5: "Production-ready; no refinement effort is needed.",
    4: "Functionally correct; only minor code-quality tweaking is needed.",
    3: "Functionally correct; major refactoring is needed to improve code quality.",
    2: "Functionally defective; minor tweaking can repair functionality.",
    1: "Functionally defective; major refactoring is needed to repair functionality.",
    0: "Fundamentally flawed or mismatched; rewriting is more efficient than repairing.",
}

AXIOM_SCALE_TEXT = (
    "AXIOM score semantics: 5=production-ready; 4=functionally correct with minor quality tweaks; "
    "3=functionally correct but major quality refactor needed; 2=functionally defective but minor fix; "
    "1=functionally defective and major repair; 0=fundamentally flawed or mismatched."
)


def clamp_axiom_grade(value: float | int) -> int:
    return max(0, min(5, int(round(float(value)))))


def axiom_scalar_score(grade: float | int) -> float:
    return round(clamp_axiom_grade(grade) * 20.0, 4)


def axiom_value_target(grade: float | int) -> float:
    return round(clamp_axiom_grade(grade) / 2.5 - 1.0, 4)


def axiom_grade_from_scalar(score: Any, max_score: float = 100.0) -> int | None:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if max_score <= 0:
        return None
    return clamp_axiom_grade(value / max_score * 5.0)


def axiom_grade_from_codecritic(correctness: Any, score: Any) -> int | None:
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        return None
    correctness_text = str(correctness or "").strip().lower()
    is_correct = "correct" in correctness_text and "incorrect" not in correctness_text and "error" not in correctness_text
    if is_correct:
        if numeric_score <= 6:
            return 3
        if numeric_score <= 8:
            return 4
        return 5
    if numeric_score <= 1:
        return 0
    if numeric_score <= 3:
        return 1
    return 2


def axiom_interval_from_binary(label: Any) -> tuple[int, int] | None:
    try:
        value = int(label)
    except (TypeError, ValueError):
        return None
    if value == 1:
        return (3, 5)
    if value == 0:
        return (0, 2)
    return None


def axiom_verdict(grade: float | int) -> str:
    grade = clamp_axiom_grade(grade)
    if grade >= 4:
        return "accept"
    if grade == 3:
        return "minor_issue"
    return "major_issue"


def axiom_functionally_correct(grade: float | int) -> bool:
    return clamp_axiom_grade(grade) >= 3


def parse_axiom_grade(payload: dict[str, Any]) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in ("axiom_grade", "grade"):
        if key in payload:
            try:
                return clamp_axiom_grade(float(payload[key]))
            except (TypeError, ValueError):
                pass
    score = payload.get("score")
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        return None
    if 0 <= numeric_score <= 5:
        return clamp_axiom_grade(numeric_score)
    if 0 <= numeric_score <= 100:
        return axiom_grade_from_scalar(numeric_score, max_score=100.0)
    return None


def grade_alignment(predicted_grade: int, target_grade: int) -> float:
    return max(0.0, 1.0 - abs(predicted_grade - target_grade) / 5.0)

