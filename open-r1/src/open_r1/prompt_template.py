
think_template = {
    'system': ("You are a helpful assistant. The assistant first thinks about the reasoning process in the mind "
"and then provides the user with the answer. The reasoning process and answer are enclosed "
"within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning "
"process here </think><answer> answer here </answer>. Now the user asks you to solve a "
"programming problem. After thinking, when you finally reach a conclusion, clearly state "
"the solution code within <answer> </answer> tags. i.e., <answer> ```python ... ``` </answer>."),
    'user': '{question}'
}






step_template = {
    'system':'''A conversation between User and Assistant. 
The user asks a question (problem specification), and the Assistant solves it by first writing a high-level step-by-step problem-solving plan and then write a correct Python program that matches the specification and passes all tests. 
The reasoning and code are enclosed within <step> </step> and <code> </code> tags.
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
</step>''',
    'user': "Now! It's your turn. Please use <step> and </step> as delim between each step and use <code> and </code> to wrap the final answer.\n\nQuestion: {question}"
}













QWEN_R1_PROMPT = """system
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. The answer should only contain the solution code and do not have anything else.
<|im_end|>
<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
{response}"""

QWEN_R1S_PROMPT = """system
A conversation between User and Assistant. 
The user asks a question (problem specification), and the Assistant solves it by first writing a high-level step-by-step problem-solving plan and then write a correct Python program that matches the specification and passes all tests. 
The reasoning and code are enclosed within <step> </step> and <code> </code> tags.
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
<|im_end|>
<|im_start|>user
Now! It's your turn. Please use <step> and </step> as delim between each step and use <code> and </code> to wrap the final answer.\n\nQuestion: 
{instruction}
<|im_end|>
<|im_start|>assistant
{response}"""


QWEN_DIRECT_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. 
@@ Instruction
You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. 
{instruction}

@@ Response
{response}"""




# with no more than six steps
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

Here is your question: {instruction}

@@ Response
{response}"""



DSC_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. 
@@ Instruction
{instruction}

@@ Response
{response}"""



MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. 
@@ Instruction
{instruction}

@@ Response
{response}"""

