REVIEW_FINAL_FORMAT_SECTION = """Structured final review format:
<review>
{{"axiom_grade": <0-5 integer>, "score": <0-100 number>, "verdict": "accept|minor_issue|major_issue", "functional_correctness": true, "repair_effort": "none|minor_quality|major_quality|minor_functional|major_functional|rewrite", "evidence_type": "provided_test_failure|deduced_counterexample|static_logic_contradiction|uncertain", "summary": "...", "evidence": ["...", "..."]}}
</review>"""

REVIEW_STEP_FORMAT_SECTION = """Next-step format:
<step>
trace_requirement | trace_visible_test | derive_counterexample | static_logic_check | challenge_previous_claim: concise new reasoning
</step>"""

AXIOM_REFINEMENT_SCALE = """AXIOM refinement-effort scale:
- 5/5: Production-ready; no effort needed.
- 4/5: Perfect functionality; minor tweaking is needed only to enhance code quality.
- 3/5: Perfect functionality; major refactoring is needed to enhance code quality.
- 2/5: Functionality needs minor tweaking to be repaired.
- 1/5: Functionality needs major refactoring to be repaired.
- 0/5: Fundamentally flawed; rewriting is more efficient than repairing.
Operational meaning: minor tweaking means a small localized change, such as adding a boundary check; major refactoring means a significant structural change, such as rewriting an entire code block."""


REVIEW_EVIDENCE_RULES = """Evidence rules:
1. Use only the task, candidate code, visible tests, and completed previous steps.
2. If tests are not visible, never claim that tests pass or fail.
3. A low grade needs concrete visible evidence: syntax/runtime error, missing required I/O, direct requirement contradiction, unrelated code, or a concrete counterexample.
4. If no functional defect is verifiable, keep functional_correctness=true and choose grade 3-5.
5. For provided_test_failure, trace the exact visible test and quote its expected assertion.
6. Treat unusual visible-test expectations as authoritative. Do not replace them with intuition.
7. Equivalent variable names, decomposition, formulas, DP states, or helper classes can still be correct.
8. Do not output code fixes."""


QWEN_REVIEW_STEP_PROMPT = """You are a code scoring model for functional correctness.
@@ Instruction
Current generation mode: continue evidence gathering.
Output exactly one new <step>...</step> block. Do not output <review> yet.
Treat completed previous steps as fixed context; continue from the last completed step without repeating it.
The step may be as long as needed to state one concrete functional-evidence item, trace, counterexample, or challenge to an unsupported prior claim.

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

Completed previous review steps:
{partial_solution}

{axiom_scale}

{evidence_rules}

{step_format_section}

@@ Response
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

Completed previous review steps:
{partial_solution}

{axiom_scale}

{evidence_rules}

Use at most 2 short evidence strings in the final JSON.

{final_format_section}

@@ Response
"""
