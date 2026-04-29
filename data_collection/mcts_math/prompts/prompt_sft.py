REVIEW_FINAL_FORMAT_RULE = (
    "Final answer only. Output one compact JSON object wrapped in <review> tags. "
    "Use at most 2 short evidence strings. Output no prose outside <review>."
)

REVIEW_FINAL_FORMAT_SECTION = """Structured final review format:
<review>
{{"axiom_grade": <0-5 integer>, "score": <0-100 number>, "verdict": "accept|minor_issue|major_issue", "functional_correctness": true, "repair_effort": "none|minor_quality|major_quality|minor_functional|major_functional|rewrite", "evidence_type": "provided_test_failure|deduced_counterexample|static_logic_contradiction|uncertain", "summary": "...", "evidence": ["...", "..."]}}
</review>"""

REVIEW_STEP_FORMAT_RULE = (
    "Step answer only. Output one <step> block under 40 words. Continue from the completed previous steps. "
    "Add exactly one new functional-evidence item or challenge one unsupported prior claim. "
    "Do not restate, paraphrase, restart, output JSON, or propose code fixes."
)

REVIEW_STEP_FORMAT_SECTION = """Next-step format:
<step>
trace_requirement | trace_visible_test | derive_counterexample | static_logic_check | challenge_previous_claim: concise new reasoning
</step>"""


QWEN_REVIEW_PROMPT = """You are a code scoring model for functional correctness.
@@ Instruction
Goal: assign a stable AXIOM 0-5 grade to the candidate code. Text critique is only evidence for the score.
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

AXIOM grade semantics for this project:
- 5: functionally correct for the visible task, with no concrete defect found.
- 4: functionally correct, with a minor visible edge-case or deployability concern.
- 3: likely functionally correct, but important behavior remains uncertain from visible evidence.
- 2: functionally defective, but a small local fix appears sufficient.
- 1: functionally defective and requires major repair.
- 0: unrelated, non-runnable, empty, or fundamentally mismatched.
Boundary rule: grades 3-5 mean functionally correct or not disproven; grades 0-2 require a concrete visible functional defect.

Rules:
1. Keep reasoning focused on functional behavior and AXIOM repair effort.
2. {mode_instruction}
3. Do not output code fixes.
4. Use only the task, candidate code, visible tests, and completed previous steps. If tests are not visible, never claim that tests pass or fail.
5. Low grades require concrete evidence: a certain syntax/runtime error, missing required I/O, direct requirement contradiction, unrelated code, or a concrete counterexample.
6. If you cannot state a verifiable functional defect, set functional_correctness=true and choose grade 3-5.
7. Evidence discipline:
   - provided_test_failure: only use this when a listed Available test directly fails.
   - deduced_counterexample: give a concrete input and expected/actual behavior that follows from the code.
   - static_logic_contradiction: cite the exact violated requirement and the exact code logic that contradicts it.
   - uncertain: use this when the concern is speculative, opaque, stylistic, or not fully verified.
8. A listed test is evidence only if you trace the candidate code against that exact test. Quote the exact expected assertion when citing failure.
9. Treat unusual listed-test expectations as authoritative. Do not replace them with intuition.
10. Equivalent variable names, decomposition, formulas, DP states, or helper classes can still be correct.
11. {format_rule}

{output_format_section}

@@ Response
"""
