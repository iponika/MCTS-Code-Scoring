#react_sft_prompt = "<question> {question} </question>\n{partial_solution}"
DEEPSEEK_PROMPT   = "You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n@@ Instruction\nQuestion:{question}\n\n@@ Response\n{partial_solution}"

DEEPSEEK_LCB_PROMPT   = "You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n@@ Instruction\n{question}\n\n@@ Response\n{partial_solution}"


QWEN_DIRECT_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. 
@@ Instruction
Please solve the following programming problem:
{question}

@@ Response
{partial_solution}"""



QWEN_STEP_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. 
@@ Instruction
You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. 
Please solve this question by first writing a high-level problem-solving plan and then write the code.
Please use <step> and </step> as delim between steps and use <code> and </code> to wrap the final answer.
Here is an example of desired output format:
<step>
1. To solve this problem, we need pi and sqrt from math module to calculate cone's lateral surface area (π * r * √(r² + h²)).
</step>
<step>
2. Calculate the generatrix (slant height).
- Import math functions pi and sqrt
- Calculate sqrt(radius² + height²)
</step>
<step>
3. Calculate and return the lateral surface area.
- Multiply pi, radius, and generatrix
- Return the final result
</step>
<step>
<code>
```python
def lateral_surface_area_cone(radius, height):
    from math import pi, sqrt
    generatrix = sqrt(radius**2 + height**2)
    return pi * radius * generatrix
```
</code>
</step>

Here is your question: {question}

@@ Response
{partial_solution}"""


QWEN_REVIEW_PROMPT = """You are an exceptionally intelligent code reviewer.
@@ Instruction
You are scoring candidate code. Textual critique is only supporting evidence for the scalar score.
Use the target review dimension to focus the evidence, but the final score must be the overall AXIOM code-quality grade.

Dimension: {dimension}

Review rubric:
{rubric}

Task description:
{question}

Candidate code:
```{code_language}
{candidate_code}
```

Available tests:
{tests}

Previous review steps:
{partial_solution}

AXIOM grade semantics: 5=production-ready; 4=functionally correct with minor quality tweaks; 3=functionally correct but major quality refactor needed; 2=functionally defective but minor fix; 1=functionally defective and major repair; 0=fundamentally flawed or mismatched. Functionality is the primary boundary: grades 3-5 are functionally correct, grades 0-2 are not.

Rules:
1. Keep all reasoning focused on the assigned dimension only.
2. {mode_instruction}
3. Do not output code fixes.
4. Calibrate against the AXIOM grade semantics. Grades 0-2 require a concrete functional defect. If you cannot state a verifiable defect, keep functional_correctness=true and choose 3-5 based on repair effort.
5. Evidence discipline is mandatory:
   - provided_test_failure: only use this when a listed Available test directly fails.
   - deduced_counterexample: give a concrete input and expected/actual behavior that follows from the code.
   - static_logic_contradiction: cite the exact violated requirement and the exact code logic that contradicts it.
   - uncertain: use this when the concern is speculative, opaque, stylistic, or not fully verified.
6. Do not claim that tests pass or fail unless those tests are explicitly listed in Available tests. If Available tests says no tests are available, never cite test results.
7. Do not lower a functionally correct solution below 3 for style, readability, performance, maintainability, opacity, or missing explanation alone.
8. Keep the final JSON compact: summary under 35 words; evidence has at most 2 items, each under 25 words. Output no prose outside <review>.

Structured final review format:
<review>
{{"dimension": "{dimension}", "axiom_grade": <0-5 integer>, "score": <0-100 number>, "verdict": "accept|minor_issue|major_issue", "functional_correctness": true, "repair_effort": "none|minor_quality|major_quality|minor_functional|major_functional|rewrite", "evidence_type": "provided_test_failure|deduced_counterexample|static_logic_contradiction|uncertain", "summary": "...", "evidence": ["...", "..."]}}
</review>

@@ Response
"""
