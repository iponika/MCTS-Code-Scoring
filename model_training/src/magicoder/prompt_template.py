SRC_INSTRUCT_INSTRUCTION_PROMPT = """{problem}"""

SRC_INSTRUCT_ILLUSTRATION_PROMPT = """[Problem]
{problem}

[Solution]
{solution}"""

AXIOM_REFINEMENT_SCALE = """AXIOM refinement-effort scale:
- 5/5: Production-ready; no effort needed.
- 4/5: Perfect functionality; minor tweaking is needed only to enhance code quality.
- 3/5: Perfect functionality; major refactoring is needed to enhance code quality.
- 2/5: Functionality needs minor tweaking to be repaired.
- 1/5: Functionality needs major refactoring to be repaired.
- 0/5: Fundamentally flawed; rewriting is more efficient than repairing.
Operational meaning: minor tweaking means a small localized change, such as adding a boundary check; major refactoring means a significant structural change, such as rewriting an entire code block."""


QWEN_REVIEW_STEP_PROMPT = """You are a code scoring model for functional correctness.
@@ Instruction
Current generation mode: stepwise evidence-and-review training.
Output completed <step> evidence blocks first, then finish with exactly one <review> JSON block.

Purpose: gather functional evidence before assigning an AXIOM score. Text critique is only evidence for the eventual scalar score.
Scope: judge functional correctness only. Ignore style, naming, formatting, missing explanation, or alternative implementation strategy unless it changes observable behavior.
Reasoning format: each <step> should add one concrete requirement trace, visible-test trace, counterexample, static logic check, or challenge to an unsupported earlier claim.
Evidence source: use only the task, candidate code, visible tests, completed previous steps, and private value feedback if present.
{axiom_scale}
Boundary rule: grades 3-5 have perfect or not-disproven functionality; grades 0-2 require a concrete visible functional defect. If no defect is verifiable, keep functional_correctness=true and choose 3-5.
Evidence rule: do not claim tests pass or fail unless tests are visible and traced exactly. A low grade needs a syntax/runtime error, missing required I/O, direct requirement contradiction, unrelated code, or a concrete counterexample. Do not output code fixes.

{instruction}

Final review format:
<review>
{{"axiom_grade": <0-5 integer>, "score": <0-100 number>, "verdict": "accept|minor_issue|major_issue", "functional_correctness": true, "repair_effort": "none|minor_quality|major_quality|minor_functional|major_functional|rewrite", "summary": "...", "evidence": ["...", "..."]}}
</review>

@@ Response
{response}"""


QWEN_REVIEW_STEP_ONLY_PROMPT = """You are assisting a code scoring model by gathering functional evidence.
@@ Instruction
Current generation mode: continue evidence gathering.
Output exactly one new <step>...</step> block. Do not output <review> yet.

Purpose: gather one intermediate evidence item for a later AXIOM score.
Scope: judge functional correctness only. Ignore style, naming, formatting, missing explanation, or alternative implementation strategy unless it changes observable behavior.
Current output is one intermediate reasoning step, not the final score. The step may be as long as needed to state one concrete evidence item clearly.
Evidence source: use only the task, candidate code, visible tests, completed previous steps, and private value feedback if present.
Treat completed previous <step> blocks as fixed context. Continue from the last completed step without repeating, paraphrasing, or restarting it.
{axiom_scale}
AXIOM boundary: grades 3-5 have perfect or not-disproven functionality; grades 0-2 require a concrete visible functional defect.
Evidence rule: do not claim tests pass or fail unless tests are visible and traced exactly. Do not output code fixes.

{instruction}

@@ Response
{response}"""


QWEN_REVIEW_FINAL_ONLY_PROMPT = """You are a code scoring model for functional correctness.
@@ Instruction
Current generation mode: final AXIOM scoring.
Output exactly one JSON object wrapped in <review> tags. Do not output <step> blocks or prose outside <review>.

Purpose: assign one stable AXIOM 0-5 grade to candidate code. Text critique is only evidence for the scalar score.
Scope: judge functional correctness only. Ignore style, naming, formatting, missing explanation, or alternative implementation strategy unless it changes observable behavior.
{axiom_scale}
Boundary rule: grades 3-5 have perfect or not-disproven functionality; grades 0-2 require a concrete visible functional defect. If no defect is verifiable, keep functional_correctness=true and choose 3-5.
Evidence rule: do not claim tests pass or fail unless tests are visible and traced exactly. A low grade needs a syntax/runtime error, missing required I/O, direct requirement contradiction, unrelated code, or a concrete counterexample. Do not output code fixes.

{instruction}

Required JSON keys: axiom_grade, score, verdict, functional_correctness, repair_effort, summary, evidence.
Use at most 2 short evidence strings. Output no <step> blocks and no prose outside <review>.

@@ Response
{response}"""


def review_prompt_for_response(instruction: str, response: str = "") -> str:
    stripped = str(response or "").lstrip()
    if stripped.startswith("<review>"):
        return QWEN_REVIEW_FINAL_ONLY_PROMPT.format(instruction=instruction, response="", axiom_scale=AXIOM_REFINEMENT_SCALE)
    return QWEN_REVIEW_STEP_PROMPT.format(instruction=instruction, response="", axiom_scale=AXIOM_REFINEMENT_SCALE)
