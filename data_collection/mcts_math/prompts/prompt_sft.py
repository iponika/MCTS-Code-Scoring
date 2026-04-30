REVIEW_FINAL_FORMAT_SECTION = """Structured final review format:
<review>
{{"axiom_grade": <0-5 integer>, "score": <0-100 number>, "verdict": "accept|minor_issue|major_issue", "functional_correctness": true, "repair_effort": "none|minor_quality|major_quality|minor_functional|major_functional|rewrite", "evidence_type": "provided_test_failure|deduced_counterexample|static_logic_contradiction|uncertain", "summary": "...", "evidence": ["...", "..."]}}
</review>"""

REVIEW_STEP_FORMAT_SECTION = """Next-step format:
<step>
trace_requirement | trace_visible_test | derive_counterexample | static_logic_check | challenge_previous_claim: concise new reasoning
</step>"""

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


REVIEW_EVIDENCE_RULES = """Evidence rules:
1. Use only the task, candidate code, visible tests, and completed previous steps.
2. If tests are not visible, never claim that tests pass or fail.
3. A low grade needs concrete visible evidence: syntax/runtime error, missing required I/O, direct requirement contradiction, unrelated code, or a concrete counterexample.
4. If no functional defect is verifiable, keep functional_correctness=true and choose grade 3-5.
5. For provided_test_failure, trace the exact visible test and quote its expected assertion.
6. Treat unusual visible-test expectations as authoritative. Do not replace them with intuition.
7. Equivalent variable names, decomposition, formulas, DP states, or helper classes can still be correct.
8. Do not output code fixes."""


REVIEW_FINAL_CONSISTENCY_RULE = """Final consistency rule:
Before choosing axiom_grade, reconcile supported previous-step evidence with the final verdict. If a completed step contains a concrete counterexample or trace supported by the task, code, or visible tests, the final review cannot silently contradict it; either reflect the defect in functional_correctness/axiom_grade or explain why that step is unsupported."""


QWEN_REVIEW_STEP_PROMPT = """You are a code scoring model for functional correctness.
@@ Instruction
Current generation mode: continue evidence gathering.
Output exactly one new <step>...</step> block. Do not output <review> yet.
Treat completed previous steps as fixed context. Continue from the last completed step without repeating it.
Do not restate the whole task, code, or earlier analysis. Add one concise evidence increment: a requirement trace, visible-test trace, counterexample, static logic check, or challenge to an unsupported prior claim.

Purpose: gather evidence for a later AXIOM score. Text critique is only evidence for the eventual scalar score.
Scope: judge whether the candidate satisfies the task's functional requirements. Ignore style, naming, formatting, missing explanation, or alternative implementation strategy unless it changes observable behavior.

Task description:
{question}

Candidate code:
```{code_language}
{candidate_code}
```

Available tests:
{tests}

{axiom_scale}

{evidence_rules}

{step_format_section}

@@ Response
{partial_solution}
"""


QWEN_REVIEW_FINAL_PROMPT = """You are a code scoring model for functional correctness.
@@ Instruction
Current generation mode: final AXIOM scoring.
Output exactly one compact JSON object wrapped in <review> tags. Do not output <step> blocks or prose outside <review>.
Treat completed previous steps as fixed evidence context. Use them if supported by the task/code/tests; ignore unsupported or repeated claims.

Purpose: assign a stable AXIOM 0-5 grade to the candidate code. Text critique is only evidence for the scalar score.
Scope: judge whether the candidate satisfies the task's functional requirements. Ignore style, naming, formatting, missing explanation, or alternative implementation strategy unless it changes observable behavior.

Task description:
{question}

Candidate code:
```{code_language}
{candidate_code}
```

Available tests:
{tests}

{axiom_scale}

{evidence_rules}

{final_consistency_rule}

Use at most 2 short evidence strings in the final JSON.

{final_format_section}

@@ Response
{partial_solution}
"""
