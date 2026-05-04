"""Canonical prompt contract shared by data generation, training, and evaluation.

Every stage of the pipeline (data_collection, model_training preprocessing,
and evaluation) MUST import prompt constants and builder functions from this
module to guarantee format alignment. The remaining compatibility wrapper
``model_training/src/magicoder/prompt_template.py`` re-exports symbols from
here so that legacy training-side import paths continue to work.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# AXIOM scale & evidence constants
# ---------------------------------------------------------------------------

AXIOM_REFINEMENT_SCALE = """AXIOM refinement-effort scale:
- First decide functional status. Grades 3-5 require perfect or not-disproven functionality; grades 1-2 require a concrete functional defect; grade 0 means the code is fundamentally mismatched to the task.
- Then decide repair scope. "minor tweaking" means a small localized change; "major refactoring" means a structural change, for example, rewriting an entire code block, algorithm, state flow, or multiple coordinated sites.
- 5/5: Production-ready; no code change is needed for the stated requirement.
- 4/5: Functionally correct, but minor code-quality tweaking is needed, for example, clearer naming, clarifying ambiguous operator precedence, replacing a magic number, removing an unused variable or dead code, or splitting an overlong statement.
- 3/5: Functionally correct, but major code-quality refactoring is needed, for example, reducing deep nesting, decomposing a long method, removing duplicated/scattered logic, reducing tight coupling, removing speculative generality, or replacing mutable global state.
- 2/5: Functionally defective, but minor localized functionality repair is enough, for example, adding a boundary check, changing one comparison/logical/arithmetic operator, correcting one initializer/index/argument order/constant, assigning an immutable-method return value, or returning the intended expression. A typical localized defect is, for example, an off-by-one error.
- 1/5: Functionally defective and requires major functional refactoring, for example, replacing the required algorithm, restoring a missing non-trivial processing step, changing an unsuitable data structure, fixing cross-iteration state corruption, redesigning recursion/base cases, repairing lifecycle/state-machine logic, restoring boundary validation, or correcting a serialization-format interpretation.
- 0/5: Fundamentally flawed; rewriting is more efficient than repairing, for example, code for an unrelated task, a severe language/API mismatch, empty/non-runnable code that prevents meaningful repair, or behavior that contradicts the core requirement.
"""


REVIEW_EVIDENCE_RULES = """Evidence rules:
1. Use only the task, candidate code, visible tests, and previous analysis notes.
2. If tests are not visible, never claim that tests pass or fail.
3. A grade below 3 requires a concrete functional defect, such as a requirement contradiction, runtime/syntax issue, missing required behavior, or specific counterexample.
4. If no functional defect is verifiable, keep functional_correctness=true and choose grade 3-5.
5. evidence should contain 1-2 short strings. Do not output code fixes."""


REVIEW_FINAL_CONSISTENCY_RULE = """Final consistency rule:
Before choosing axiom_grade, reconcile supported previous reasoning evidence with the final judgment. If an earlier reasoning note contains a concrete counterexample or trace supported by the task, code, or visible tests, the final review cannot silently contradict it; either reflect the defect in functional_correctness/axiom_grade or explain why that note is unsupported."""


REVIEW_STEP_FORMAT_SECTION = """Intermediate reasoning format:
Write one concise functional-correctness evidence note in the model's native reasoning channel. Do not output XML tags, JSON, markdown fences, code fixes, or the final <review> block in this intermediate turn."""


REVIEW_FINAL_FORMAT_SECTION = """Structured final review format:
<review>
{{"axiom_grade": <0-5 integer>, "functional_correctness": true|false, "repair_effort": "none|minor_quality|major_quality|minor_functional|major_functional|rewrite", "evidence_type": "provided_test_failure|deduced_counterexample|static_logic_contradiction|uncertain", "summary": "...", "evidence": ["...", "..."]}}
</review>"""

# Greedy prefill for anchoring final review format during generation.
FINAL_REVIEW_PREFILL = '<review>\n{"axiom_grade": '


# ---------------------------------------------------------------------------
# Unified prompt templates
# ---------------------------------------------------------------------------

REVIEW_STEP_PROMPT = """You are a code analysis model for functional correctness.
@@ Instruction
This is an intermediate turn of a multi-step code review. You are not assigning the final score in this turn. Your job is to analyze code correctness succeeding previous analysis.

Your analysis must gather evidence for the AXIOM 0-5 refinement-effort scale:

{axiom_scale}

Remember this is a intermediate turn. Earlier turns may have already analyzed the candidate code and marked serial number for each point, I insert them into your "previous output" as if it were you who thought them out. You should also start with a serial number succeeding previous serial number and followed by your analysis and everytime you output a serial number you should check through your previous output.

If you don't see previous output, you're the first round, and you first output token must be "1. " (and must not be natural language such as "Okay")followed by your initial analysis, and if you have more analisys, use "2. ", "3. " etc.

Don't repeat - each new reasoning note must not restate the whole task, code, or earlier analysis, don't ignore or abort previous analysis. Your analysis object is as follows:

{instruction}

@@ Response
{partial_solution}"""


REVIEW_FINAL_PROMPT = """You are a code scoring model for functional correctness.
@@ Instruction
You evaluate code based on functional correctness.

This is the final turn of a multi-step code review. Synthesize all available earlier analysis with the task, code, and visible tests, then assign one AXIOM grade.

You must score according to the AXIOM 0-5 refinement-effort scale:

{axiom_scale}

Output must be exactly one JSON object wrapped in <review> tags. Do not output natural-language text outside the tags, markdown fences, <think> blocks, <step> blocks, or code fixes. Otherwise the result cannot be parsed.

{instruction}

{final_consistency_rule}

Attention, your output MUST be in the following format:

{final_format_section}

Field rules:
- axiom_grade is the AXIOM grade.
- functional_correctness must be true for grades 3-5 and false for grades 0-2.
- repair_effort must match the selected AXIOM grade: 5 -> none, 4 -> minor_quality, 3 -> major_quality, 2 -> minor_functional, 1 -> major_functional, 0 -> rewrite.
- evidence_type must be one of provided_test_failure, deduced_counterexample, static_logic_contradiction, uncertain.
- summary should be one short sentence explaining the final judgment.
- evidence should contain 1-2 short evidence strings grounded in the task, candidate code, visible tests, or previous analysis notes.
- If previous analysis notes conflict, follow the claim best supported by the task, code, and visible tests.

@@ Response
{partial_solution}"""


# Legacy aliases – used by older code that references the Qwen-prefixed names.
QWEN_REVIEW_STEP_PROMPT = REVIEW_STEP_PROMPT
QWEN_REVIEW_FINAL_PROMPT = REVIEW_FINAL_PROMPT


# ---------------------------------------------------------------------------
# Training-side compatibility templates. 新代码只能用上面的 Unified，compatibility 仅供旧训练入口使用
# ---------------------------------------------------------------------------

TRAINING_REVIEW_STEP_PROMPT = """You are a code scoring model for functional correctness.
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


TRAINING_REVIEW_STEP_ONLY_PROMPT = """You are assisting a code scoring model by gathering functional evidence.
@@ Instruction
This is an intermediate turn of a multi-step code review. Earlier turns may have already analyzed the candidate code. If previous analysis notes are provided below, or if earlier assistant messages already contain analysis notes, use them as fixed context and add one new evidence item.

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


TRAINING_REVIEW_FINAL_ONLY_PROMPT = """You are a code scoring model for functional correctness.
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
        return TRAINING_REVIEW_FINAL_ONLY_PROMPT.format(
            instruction=instruction,
            response="",
            axiom_scale=AXIOM_REFINEMENT_SCALE,
            final_consistency_rule=REVIEW_FINAL_CONSISTENCY_RULE,
        )
    return TRAINING_REVIEW_STEP_PROMPT.format(
        instruction=instruction,
        response="",
        axiom_scale=AXIOM_REFINEMENT_SCALE,
        step_format_section=REVIEW_STEP_FORMAT_SECTION,
    )


# ---------------------------------------------------------------------------
# Instruction builder (single source of truth)
# ---------------------------------------------------------------------------

def truncate_for_review(
    value: object,
    max_chars: int,
    marker: str = "\n... [truncated]",
) -> tuple[str, bool]:
    text = str(value or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars].rstrip() + marker, True


def build_review_instruction(
    problem: str,
    candidate_code: str,
    code_language: str = "python",
    tests_text: str = "No tests are available to the reviewer.",
    *,
    max_problem_chars: int = 3500,
    max_code_chars: int = 3500,
    mark_code_truncation_inside_block: bool = True,
) -> str:
    """Build the instruction body that describes the scoring task.

    This function is the single source of truth for the instruction content
    used by data generation, training preprocessing, and evaluation.
    """
    language = str(code_language or "python").strip() or "python"
    problem_text, problem_truncated = truncate_for_review(problem, max_problem_chars)
    code_marker = "\n... [truncated]" if mark_code_truncation_inside_block else ""
    code_text, code_truncated = truncate_for_review(candidate_code, max_code_chars, marker=code_marker)
    truncation_notice = ""
    if code_truncated and not mark_code_truncation_inside_block:
        truncation_notice = (
            "\n\nPrompt-budget note: the candidate code was shortened for evaluation input length. "
            "Do not treat the shortening itself as evidence of a syntax error, missing implementation, "
            "or truncated user code; score only the visible code and task evidence."
        )
    if problem_truncated:
        truncation_notice += (
            "\nPrompt-budget note: the task description was shortened; do not treat omitted text as a code defect."
        )

    return (
        "Scoring target: assess candidate code correctness and assign the overall AXIOM code grade.\n\n"
        f"Task description:\n{problem_text}\n\n"
        "Candidate code:\n"
        f"```{language}\n"
        f"{code_text}\n"
        "```\n\n"
        f"Available tests:\n{tests_text}\n\n"
        "Assess functional correctness using concrete evidence from the task, code, and any reviewer-visible tests."
        f"{truncation_notice}"
    )


def build_review_instruction_from_sample(
    sample: dict[str, Any],
    dimension: str = "Correctness Verification",
    *,
    max_problem_chars: int = 3500,
    max_code_chars: int = 3500,
    mark_code_truncation_inside_block: bool = True,
    show_tests_in_prompt: bool = False,
) -> str:
    """Convenience wrapper that extracts fields from a sample dict."""
    problem = str(sample.get("problem") or sample.get("question") or "")
    candidate_code = str(sample.get("candidate_code") or sample.get("answer") or "")
    code_language = str(sample.get("language") or sample.get("lang") or sample.get("code_language") or "python")

    tests = sample.get("tests") or []
    tests_for_prompt = sample.get("tests_for_prompt")
    if not show_tests_in_prompt:
        tests_text = "No tests are available to the reviewer."
    elif tests_for_prompt is not None:
        tests_text = str(tests_for_prompt)
    elif tests:
        tests_text = "\n".join(str(test) for test in tests[:5])
        if len(tests) > 5:
            tests_text += f"\n... ({len(tests) - 5} more assertions omitted)"
    else:
        tests_text = "No tests are available."

    return build_review_instruction(
        problem=problem,
        candidate_code=candidate_code,
        code_language=code_language,
        tests_text=tests_text,
        max_problem_chars=max_problem_chars,
        max_code_chars=max_code_chars,
        mark_code_truncation_inside_block=mark_code_truncation_inside_block,
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_review_prompt(
    instruction: str,
    partial_solution: str = "",
    *,
    force_final: bool = False,
) -> str:
    """Format a complete prompt string from an instruction and partial solution.

    The returned string contains an ``@@ Response`` separator.  Use
    :func:`prompt_to_chat_messages` to split it into chat messages.

    Parameters
    ----------
    instruction : str
        Output of :func:`build_review_instruction`.
    partial_solution : str
        Previously generated steps (may be empty).
    force_final : bool
        If *True*, use the final-review template; otherwise use the step
        template.
    """
    partial = str(partial_solution or "").strip()
    if partial == "None":
        partial = ""
    if partial:
        partial = partial.rstrip() + "\n"
    if force_final:
        partial += FINAL_REVIEW_PREFILL
        return REVIEW_FINAL_PROMPT.format(
            instruction=instruction,
            partial_solution=partial,
            axiom_scale=AXIOM_REFINEMENT_SCALE,
            evidence_rules=REVIEW_EVIDENCE_RULES,
            final_consistency_rule=REVIEW_FINAL_CONSISTENCY_RULE,
            final_format_section=REVIEW_FINAL_FORMAT_SECTION,
        )
    return REVIEW_STEP_PROMPT.format(
        instruction=instruction,
        partial_solution=partial,
        axiom_scale=AXIOM_REFINEMENT_SCALE,
        evidence_rules=REVIEW_EVIDENCE_RULES,
        step_format_section=REVIEW_STEP_FORMAT_SECTION,
    )


def normalize_partial_solution(partial_solution: str = "") -> tuple[str, str]:
    """Return both the completed-note text and the response-prefix variant."""
    completed_steps = str(partial_solution or "").strip()
    if completed_steps == "None":
        completed_steps = ""
    response_prefix = completed_steps.rstrip() + "\n" if completed_steps else ""
    return completed_steps, response_prefix


def relax_freeform_final_prompt(prompt: str) -> str:
    """Relax anchored-final wording so the model may reason before the last <review> block."""
    relaxed = str(prompt or "")
    relaxed = relaxed.replace(
        "Output must be exactly one JSON object wrapped in <review> tags. Do not output natural-language text outside the tags, markdown fences, <think> blocks, <step> blocks, or code fixes. Otherwise the result cannot be parsed.\n",
        "You may reason before the final labeled answer, but finish with exactly one complete <review>...</review> block.\n",
    )
    relaxed = relaxed.replace(
        "Field rules:\n",
        "The final labeled answer begins at the last complete <review> block in your response. "
        "Inside that final block, write the JSON object required below.\n\nField rules:\n",
    )
    return relaxed


def build_review_prompt_from_sample(
    sample: dict[str, Any],
    *,
    dimension: str = "Correctness Verification",
    partial_solution: str = "",
    force_final: bool = False,
    final_only: bool = False,
    step_context_mode: str = "assistant_prefix",
    parse_error: dict[str, Any] | None = None,
    freeform_final_review: bool = False,
    max_problem_chars: int = 3500,
    max_code_chars: int = 3500,
    mark_code_truncation_inside_block: bool = True,
    show_tests_in_prompt: bool = False,
) -> str:
    """Build the canonical raw-text review prompt from one sample.

    This is the single shared prompt-construction entrypoint used by data
    generation and evaluation. Callers should add model-specific suffixes
    (for example ``/think`` or ``/no_think``) separately via
    :func:`apply_review_prompt_controls`.
    """
    instruction = build_review_instruction_from_sample(
        sample,
        dimension,
        max_problem_chars=max_problem_chars,
        max_code_chars=max_code_chars,
        mark_code_truncation_inside_block=mark_code_truncation_inside_block,
        show_tests_in_prompt=show_tests_in_prompt,
    )
    completed_steps, response_prefix = normalize_partial_solution(partial_solution)

    if final_only or force_final:
        instruction += (
            "\n\nThis is the final scoring turn. Output exactly one <review> JSON block. "
            "Do not add more intermediate reasoning notes."
        )
        if completed_steps:
            instruction += (
                "\n\nYou don't have to analyze code by yourself."
                f"\n\nEarlier analysis:\n{completed_steps}"
            )
        if parse_error:
            instruction += (
                "\n\nPrevious final review parse error: "
                f"{parse_error.get('error')}: {parse_error.get('message', '')}. "
                "Correct the JSON syntax in the next review block."
            )
        prompt = REVIEW_FINAL_PROMPT.format(
            instruction=instruction,
            partial_solution="" if freeform_final_review else FINAL_REVIEW_PREFILL,
            axiom_scale=AXIOM_REFINEMENT_SCALE,
            evidence_rules=REVIEW_EVIDENCE_RULES,
            final_consistency_rule=REVIEW_FINAL_CONSISTENCY_RULE,
            final_format_section=REVIEW_FINAL_FORMAT_SECTION,
        )
        return relax_freeform_final_prompt(prompt) if freeform_final_review else prompt

    if step_context_mode == "instruction_context":
        instruction += (
            "\n\nThis is an intermediate scoring turn. "
            "Use previous analysis notes as fixed context and do not repeat them. "
            "Generate one concise native reasoning note. Do not output XML tags, JSON, or <review> yet. "
            "Each new reasoning note must add new evidence instead of continuing the wording of a previous note."
        )
        if completed_steps:
            instruction += f"\n\nPrevious analysis notes:\n{completed_steps}"
        prompt = REVIEW_STEP_PROMPT.format(
            instruction=instruction,
            partial_solution="",
            axiom_scale=AXIOM_REFINEMENT_SCALE,
            evidence_rules=REVIEW_EVIDENCE_RULES,
            step_format_section=REVIEW_STEP_FORMAT_SECTION,
        )
        return prompt.replace(
            "If previous analysis notes are provided below, or if earlier assistant messages already contain analysis notes, use them as fixed context and add one new evidence item.\n",
            "If previous analysis notes are provided below, use them as fixed context and add one new evidence item without repeating them.\n",
        )

    instruction += (
        "\n\nPrevious analysis notes may already appear in the assistant history for this conversation. "
        "Use them as fixed context and continue the analysis. "
        "The previous rounds' analysis text may be inserted directly as your prior thinking history. "
        "If you can see prior thoughts, first summarize the code's correctness status you previously reached in one sentence, "
        "then continue with a further correctness judgment that builds on that prior analysis. "
        "Generate one concise native reasoning note. Do not output XML tags, JSON, or <review> yet. "
        "Do not repeat, paraphrase, or restart previous steps."
    )
    return REVIEW_STEP_PROMPT.format(
        instruction=instruction,
        partial_solution=response_prefix,
        axiom_scale=AXIOM_REFINEMENT_SCALE,
        evidence_rules=REVIEW_EVIDENCE_RULES,
        step_format_section=REVIEW_STEP_FORMAT_SECTION,
    )


def apply_review_prompt_controls(
    prompt: str,
    *,
    force_final: bool,
    freeform_final_review: bool = False,
    thinking_mode: str = "",
    review_native_thinking_steps: bool = False,
) -> str:
    """Apply model-control suffixes such as /think or /no_think."""
    normalized_mode = str(thinking_mode or "").strip().lower()
    if force_final and review_native_thinking_steps and not freeform_final_review:
        return str(prompt) + "\n\n/no_think"
    if normalized_mode in {"think", "/think"}:
        return str(prompt) + "\n\n/think"
    if normalized_mode in {"no_think", "no-think", "/no_think"}:
        return str(prompt) + "\n\n/no_think"
    return str(prompt)


# ---------------------------------------------------------------------------
# Chat-message helpers
# ---------------------------------------------------------------------------

QWEN_USER_PREAMBLE = (
    "You are a code scoring model for functional correctness.\n"
    "Use the task, candidate code, and AXIOM refinement-effort scale to assign one stable score.\n"
    "Put intermediate evidence inside <think> as concise reasoning notes, then finish with exactly one "
    "<review> JSON block.\n"
    "Do not put q_value, reward, or training metadata in the answer."
)


def build_review_user_content(instruction: str) -> str:
    """Build the user message content that matches the training format.

    This is the single source of truth for the user-side chat message.
    Training preprocessing, evaluation, and data generation should all
    produce the same user content for a given instruction.
    """
    return f"{QWEN_USER_PREAMBLE}\n\n{instruction.strip()}"


def prompt_to_chat_messages(prompt_text: str) -> list[dict[str, str]]:
    """Split a prompt at ``@@ Response`` into chat messages.

    Mirrors ``chat_messages_for_prompt`` in
    ``data_collection/mcts_math/llms/local_llms.py``.
    """
    cleaned = str(prompt_text or "").rstrip()
    # Strip /think or /no_think suffixes
    lower = cleaned.lower()
    for suffix in ("/no_think", "/think"):
        if lower.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].rstrip()
            break

    if "@@ Response" not in cleaned:
        return [{"role": "user", "content": cleaned}]

    instruction, response = cleaned.split("@@ Response", 1)
    user_content = instruction.rstrip() + "\n\n@@ Response"
    response_prefix = response.lstrip("\n").rstrip()
    if not response_prefix:
        return [{"role": "user", "content": user_content}]
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response_prefix},
    ]


def render_eval_prompt(
    tokenizer: Any,
    user_content: str,
    assistant_prefix: str = "",
    *,
    enable_thinking: bool | None = None,
) -> str:
    """Render a chat-template-formatted prompt string for model generation.

    Parameters
    ----------
    tokenizer
        A HuggingFace tokenizer with ``apply_chat_template``.
    user_content : str
        The user message content (output of :func:`build_review_user_content`).
    assistant_prefix : str
        Partial assistant content to continue from (e.g. previous ``<think>``
        blocks).  If empty, the model starts a fresh assistant turn.
    enable_thinking : bool or None
        Controls native thinking mode in the chat template.  *None* uses
        the tokenizer default.
    """
    messages = [{"role": "user", "content": user_content}]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    try:
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        prompt = tokenizer.apply_chat_template(messages, **kwargs)

    if assistant_prefix:
        prompt += assistant_prefix
    return prompt
