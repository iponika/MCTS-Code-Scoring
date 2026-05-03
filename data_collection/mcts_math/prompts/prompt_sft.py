import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from shared.prompt_contract import (  # noqa: F401, E402
    AXIOM_REFINEMENT_SCALE,
    REVIEW_EVIDENCE_RULES,
    REVIEW_FINAL_CONSISTENCY_RULE,
    REVIEW_FINAL_FORMAT_SECTION,
    REVIEW_STEP_FORMAT_SECTION,
    FINAL_REVIEW_PREFILL,
    REVIEW_STEP_PROMPT,
    REVIEW_FINAL_PROMPT,
    QWEN_USER_PREAMBLE,
    build_review_instruction,
    build_review_instruction_from_sample,
    build_review_user_content,
    build_review_prompt,
    prompt_to_chat_messages,
    render_eval_prompt,
    truncate_for_review,
)


# ---------------------------------------------------------------------------
# Data-generation templates with raw field placeholders
# ({question}, {candidate_code}, etc.).  These are used by callers
# (agents/utils.py, direct_review_local.py, direct_review_api.py,
# direct_bootstrap_review.py) that format the template with per-field values
# instead of a pre-built {instruction} block.  New code should prefer
# build_review_prompt() from shared.prompt_contract.
# ---------------------------------------------------------------------------

QWEN_REVIEW_STEP_PROMPT = """You are a code scoring model for functional correctness.
@@ Instruction
This is an intermediate turn of a multi-step code review. Earlier turns may have already analyzed the candidate code. If previous analysis notes are provided below or already present after @@ Response, use them as fixed context and add one new evidence item.

You are not assigning the final score in this turn. Generate one concise native reasoning note. Do not output XML tags, JSON, markdown fences, code fixes, or the final <review> block in this intermediate turn.

You must gather evidence for the AXIOM 0-5 refinement-effort scale:

{axiom_scale}

Reasoning rules:
- Focus on one new evidence point: requirement trace, visible-test trace, counterexample, static logic check, or challenge to an unsupported prior claim.
- Each new reasoning note must add new evidence and must not restate the whole task, code, or earlier analysis.
- If previous analysis notes conflict, challenge only the claim best contradicted by the task, code, or visible tests.
- Do not decide the final AXIOM grade yet.

Task description:
{question}

Candidate code:
```{code_language}
{candidate_code}
```

Available tests:
{tests}

End of available tests.

{evidence_rules}

{step_format_section}

@@ Response
{partial_solution}
"""


QWEN_REVIEW_FINAL_PROMPT = """You are a code scoring model for functional correctness.
@@ Instruction
You evaluate code based on functional correctness.

This is the final turn of a multi-step code review. Earlier turns may have already analyzed the candidate code. If previous analysis notes are provided below, synthesize them, resolve conflicts using the task, code, and visible tests, and assign one AXIOM grade.

You must score according to the AXIOM 0-5 refinement-effort scale:

{axiom_scale}

Output must be exactly one JSON object wrapped in <review> tags. Do not output natural-language text outside the tags, markdown fences, <think> blocks, <step> blocks, or code fixes. Otherwise the result cannot be parsed.

Field rules:
- axiom_grade is the AXIOM grade.
- functional_correctness must be true for grades 3-5 and false for grades 0-2.
- repair_effort must match the selected AXIOM grade: 5 -> none, 4 -> minor_quality, 3 -> major_quality, 2 -> minor_functional, 1 -> major_functional, 0 -> rewrite.
- evidence_type must be one of provided_test_failure, deduced_counterexample, static_logic_contradiction, uncertain.
- summary should be one short sentence explaining the final judgment.
- evidence should contain 1-2 short evidence strings grounded in the task, candidate code, visible tests, or previous analysis notes.
- If previous analysis notes conflict, follow the claim best supported by the task, code, and visible tests.

Task description:
{question}

Candidate code:
```{code_language}
{candidate_code}
```

Available tests:
{tests}

End of available tests.

{evidence_rules}

{final_consistency_rule}

{final_format_section}

@@ Response
{partial_solution}
"""
