from __future__ import annotations

from typing import Iterable


def normalize_step_counts(values: Iterable[int]) -> list[int]:
    normalized = sorted({int(value) for value in values if int(value) > 0})
    return normalized


def stepwise_variants(step_counts: Iterable[int]) -> list[dict[str, int | str]]:
    return [
        {"tag": f"direct_stepwise_{step_count}step", "reasoning_steps": step_count}
        for step_count in normalize_step_counts(step_counts)
    ]


def direct_review_alignment_targets(
    static_counts: dict[str, int],
    direct_counts: dict[str, int],
) -> dict[str, int]:
    static_policy = int(static_counts.get("policy", 0) or 0)
    direct_policy = int(direct_counts.get("policy", 0) or 0)
    target_policy = min(static_policy, direct_policy)
    return {
        "target_policy_count": target_policy,
        "target_value_count": -1,
        "target_total_count": -1,
        "static_policy_count": static_policy,
        "direct_policy_count": direct_policy,
    }
