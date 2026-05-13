import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from shared.prompt_contract import (  # noqa: F401, E402
    AXIOM_REFINEMENT_SCALE,
    CODECRITIC_CORRECTNESS_SCALE,
    CODECRITIC_FINAL_REVIEW_PREFILL,
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
    build_review_prompt_from_sample,
    build_review_user_content,
    build_review_prompt,
    apply_review_prompt_controls,
    normalize_partial_solution,
    relax_freeform_final_prompt,
    prompt_to_chat_messages,
    review_prompt_for_response,
    render_eval_prompt,
    TRAINING_REVIEW_FINAL_ONLY_PROMPT,
    TRAINING_REVIEW_STEP_ONLY_PROMPT,
    TRAINING_REVIEW_STEP_PROMPT,
    truncate_for_review,
)

SRC_INSTRUCT_INSTRUCTION_PROMPT = """{problem}"""

SRC_INSTRUCT_ILLUSTRATION_PROMPT = """[Problem]
{problem}

[Solution]
{solution}"""


# ---------------------------------------------------------------------------
# Training-side compatibility aliases
# ---------------------------------------------------------------------------

QWEN_REVIEW_STEP_PROMPT = TRAINING_REVIEW_STEP_PROMPT
QWEN_REVIEW_STEP_ONLY_PROMPT = TRAINING_REVIEW_STEP_ONLY_PROMPT
QWEN_REVIEW_FINAL_ONLY_PROMPT = TRAINING_REVIEW_FINAL_ONLY_PROMPT
