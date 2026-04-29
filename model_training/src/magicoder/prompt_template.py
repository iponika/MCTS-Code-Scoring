SRC_INSTRUCT_INSTRUCTION_PROMPT = """{problem}"""

SRC_INSTRUCT_ILLUSTRATION_PROMPT = """[Problem]
{problem}

[Solution]
{solution}"""

AXIOM_REFINEMENT_SCALE = """AXIOM refinement-effort scale:
- First decide functional status. Grades 3-5 require perfect or not-disproven functionality; grades 1-2 require a concrete functional defect; grade 0 means the code is fundamentally mismatched to the task.
- Then decide repair scope. "minor tweaking" means a small localized change; "major refactoring" means a structural change, for example, rewriting an entire code block, algorithm, state flow, or multiple coordinated sites.
- 5/5: Production-ready; no code change is needed for the stated requirement.
- 4/5: Functionally correct, but minor code-quality tweaking is needed, for example, clearer naming, clarifying ambiguous operator precedence, replacing a magic number, removing an unused variable or dead code, or splitting an overlong statement.
- 3/5: Functionally correct, but major code-quality refactoring is needed, for example, reducing deep nesting, decomposing a long method, removing duplicated/scattered logic, reducing tight coupling, removing speculative generality, or replacing mutable global state.
- 2/5: Functionally defective, but minor localized functionality repair is enough, for example, adding a boundary check, changing one comparison/logical/arithmetic operator, correcting one initializer/index/argument order/constant, assigning an immutable-method return value, or returning the intended expression. A typical localized defect is, for example, an off-by-one error.
- 1/5: Functionally defective and requires major functional refactoring, for example, replacing the required algorithm, restoring a missing non-trivial processing step, changing an unsuitable data structure, fixing cross-iteration state corruption, redesigning recursion/base cases, repairing lifecycle/state-machine logic, restoring boundary validation, or correcting a serialization-format interpretation.
- 0/5: Fundamentally flawed; rewriting is more efficient than repairing, for example, code for an unrelated task, a severe language/API mismatch, empty/non-runnable code that prevents meaningful repair, or behavior that contradicts the core requirement.
Examples are illustrative, not exhaustive criteria; score by the closest AXIOM repair-effort level supported by concrete evidence."""


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
