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
You are reviewing candidate code for exactly one review dimension.

Dimension: {dimension}

Review rubric:
{rubric}

Task description:
{question}

Candidate code:
```python
{candidate_code}
```

Available tests:
{tests}

Previous review steps:
{partial_solution}

Rules:
1. Keep all reasoning focused on the assigned dimension only.
2. {mode_instruction}
3. Do not output code fixes.

Structured final review format:
<review>
{{"dimension": "{dimension}", "score": <1-10 integer>, "verdict": "accept|minor_issue|major_issue", "summary": "...", "evidence": ["...", "..."]}}
</review>

@@ Response
"""
