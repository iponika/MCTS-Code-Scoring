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

SRC_INSTRUCT_INSTRUCTION_PROMPT = """{problem}"""

SRC_INSTRUCT_ILLUSTRATION_PROMPT = """[Problem]
{problem}

[Solution]
{solution}"""


# ---------------------------------------------------------------------------
# Legacy eval templates with {instruction}/{response}/{axiom_scale} placeholders.
# Used by review_value_guided_evaluator.py, review_policy_value_inference.py,
# review_evaluator.py (raw-text fallback path), and review_prompt_for_response.
# New code should prefer build_chat_eval_prompt in review_evaluator.py.
# ---------------------------------------------------------------------------

QWEN_REVIEW_STEP_PROMPT = """You are a code scoring model for functional correctness.
@@ Instruction
This training response may contain intermediate native reasoning followed by a final code score. Treat every reasoning note as evidence for the final <review>, not as an independent final answer.

You must score according to the AXIOM 0-5 refinement-effort scale:

{axiom_scale}

Reasoning rules:
- Each reasoning note should focus on one evidence point: requirement trace, visible-test trace, counterexample, static logic check, or challenge to an unsupported prior claim.
- Do not restate the whole task, code, or earlier analysis; each new note must add new evidence.
- Do not output XML step tags or JSON step objects for intermediate reasoning.

Review rules:
- Finish with exactly one <review> JSON block.
- Do not claim tests pass or fail unless tests are visible and traced exactly.
- A grade below 3 requires a concrete functional defect, such as a requirement contradiction, runtime/syntax issue, missing required behavior, or specific counterexample.
- If no functional defect is verifiable, keep functional_correctness=true and choose grade 3-5.
- When finishing with <review>, reconcile supported reasoning evidence with the final judgment. If a prior reasoning note gives a supported counterexample or trace, the final review cannot silently contradict it.

{instruction}

{step_format_section}

Final review format:
<review>
{{"axiom_grade": <0-5 integer>, "functional_correctness": true|false, "repair_effort": "none|minor_quality|major_quality|minor_functional|major_functional|rewrite", "evidence_type": "provided_test_failure|deduced_counterexample|static_logic_contradiction|uncertain", "summary": "...", "evidence": ["...", "..."]}}
</review>

@@ Response
{response}"""


QWEN_REVIEW_STEP_ONLY_PROMPT = """You are assisting a code scoring model by gathering functional evidence.
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
- Do not claim tests pass or fail unless tests are visible and traced exactly.
- Do not output code fixes.

{instruction}

{step_format_section}

@@ Response
{response}"""


QWEN_REVIEW_FINAL_ONLY_PROMPT = """You are a code scoring model for functional correctness.
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
- If visible tests are not provided, do not claim that tests pass or fail.
- A grade below 3 requires a concrete functional defect, such as a requirement contradiction, runtime/syntax issue, missing required behavior, or specific counterexample.
- If no functional defect is verifiable, keep functional_correctness=true and choose grade 3-5.
- If previous analysis notes conflict, follow the claim best supported by the task, code, and visible tests.

{final_consistency_rule}

{instruction}

Required output format:
<review>
{{"axiom_grade": <0-5 integer>, "functional_correctness": true|false, "repair_effort": "none|minor_quality|major_quality|minor_functional|major_functional|rewrite", "evidence_type": "provided_test_failure|deduced_counterexample|static_logic_contradiction|uncertain", "summary": "...", "evidence": ["...", "..."]}}
</review>

@@ Response
{response}"""


def review_prompt_for_response(instruction: str, response: str = "") -> str:
    stripped = str(response or "").lstrip()
    if stripped.startswith("<review>"):
        return QWEN_REVIEW_FINAL_ONLY_PROMPT.format(
            instruction=instruction,
            response="",
            axiom_scale=AXIOM_REFINEMENT_SCALE,
            final_consistency_rule=REVIEW_FINAL_CONSISTENCY_RULE,
        )
    return QWEN_REVIEW_STEP_PROMPT.format(
        instruction=instruction,
        response="",
        axiom_scale=AXIOM_REFINEMENT_SCALE,
        step_format_section=REVIEW_STEP_FORMAT_SECTION,
    )
