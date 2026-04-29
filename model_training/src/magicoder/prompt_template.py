SRC_INSTRUCT_INSTRUCTION_PROMPT = """{problem}"""

SRC_INSTRUCT_ILLUSTRATION_PROMPT = """[Problem]
{problem}

[Solution]
{solution}"""


QWEN_REVIEW_STEP_PROMPT = """You are a code scoring model for functional correctness.
@@ Instruction
Goal: assign a stable AXIOM 0-5 grade to candidate code. Text critique is only evidence for the score.
Scope: judge functional correctness only. Ignore style, naming, formatting, missing explanation, or alternative implementation strategy unless it changes observable behavior.
Reasoning format: write concise <step> blocks when useful, then finish with exactly one <review> JSON block.
Evidence source: use only the task, candidate code, visible tests, completed previous steps, and private value feedback if present.
AXIOM semantics: 5=functionally correct with no concrete defect found; 4=functionally correct with a minor visible edge/deployability concern; 3=likely functionally correct but important behavior remains uncertain; 2=functionally defective with a small local fix; 1=functionally defective requiring major repair; 0=unrelated, non-runnable, empty, or fundamentally mismatched.
Boundary rule: grades 3-5 mean functionally correct or not disproven; grades 0-2 require a concrete visible functional defect. If no defect is verifiable, keep functional_correctness=true and choose 3-5.
Evidence rule: do not claim tests pass or fail unless tests are visible and traced exactly. A low grade needs a syntax/runtime error, missing required I/O, direct requirement contradiction, unrelated code, or a concrete counterexample. Do not output code fixes.

{instruction}

Final review format:
<review>
{{"axiom_grade": <0-5 integer>, "score": <0-100 number>, "verdict": "accept|minor_issue|major_issue", "functional_correctness": true, "repair_effort": "none|minor_quality|major_quality|minor_functional|major_functional|rewrite", "summary": "...", "evidence": ["...", "..."]}}
</review>

@@ Response
{response}"""


QWEN_REVIEW_STEP_ONLY_PROMPT = """You are a code scoring model for functional correctness.
@@ Instruction
Goal: assign a stable AXIOM 0-5 grade to candidate code. Text critique is only evidence for the score.
Scope: judge functional correctness only. Ignore style, naming, formatting, missing explanation, or alternative implementation strategy unless it changes observable behavior.
Current output is one intermediate reasoning step, not the final score.
Evidence source: use only the task, candidate code, visible tests, completed previous steps, and private value feedback if present.
AXIOM boundary: grades 3-5 mean functionally correct or not disproven; grades 0-2 require a concrete visible functional defect.
Evidence rule: do not claim tests pass or fail unless tests are visible and traced exactly. Do not output code fixes.

{instruction}

@@ Response
{response}"""


QWEN_REVIEW_FINAL_ONLY_PROMPT = """You are a code scoring model for functional correctness.
@@ Instruction
Goal: assign one stable AXIOM 0-5 grade to candidate code. Text critique is only evidence for the score.
Scope: judge functional correctness only. Ignore style, naming, formatting, missing explanation, or alternative implementation strategy unless it changes observable behavior.
AXIOM semantics: 5=functionally correct with no concrete defect found; 4=functionally correct with a minor visible edge/deployability concern; 3=likely functionally correct but important behavior remains uncertain; 2=functionally defective with a small local fix; 1=functionally defective requiring major repair; 0=unrelated, non-runnable, empty, or fundamentally mismatched.
Boundary rule: grades 3-5 mean functionally correct or not disproven; grades 0-2 require a concrete visible functional defect. If no defect is verifiable, keep functional_correctness=true and choose 3-5.
Evidence rule: do not claim tests pass or fail unless tests are visible and traced exactly. A low grade needs a syntax/runtime error, missing required I/O, direct requirement contradiction, unrelated code, or a concrete counterexample. Do not output code fixes.

{instruction}

Output exactly one JSON object wrapped in <review> tags.
Required JSON keys: axiom_grade, score, verdict, functional_correctness, repair_effort, summary, evidence.
Use at most 2 short evidence strings. Output no <step> blocks and no prose outside <review>.

@@ Response
{response}"""


def review_prompt_for_response(instruction: str, response: str = "") -> str:
    stripped = str(response or "").lstrip()
    if stripped.startswith("<review>"):
        return QWEN_REVIEW_FINAL_ONLY_PROMPT.format(instruction=instruction, response="")
    return QWEN_REVIEW_STEP_PROMPT.format(instruction=instruction, response="")
