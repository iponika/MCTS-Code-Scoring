
from typing import Optional, Any, Dict, List, Callable, Type, Tuple
import os
import sys
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.outputs import CompletionOutput, RequestOutput
import jsonlines


GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')

REFINE_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. 
@@ Instruction
You will be given a programming problem, an incorrect reasoning path and the groud truth solution for this problem. 
You task is to identify from which step the path is not incorrect based on the groud truth solution and rewrite the incorrect part.
Please note that the reasoning path might also be empty, in that case you should generate the whole path based on the groud truth solution.

Desired output format:
From step <num>, the path is not correct. I will rewrite the reasoning path from step <num>.
<answer>

Here are three examples:
Example 1:
Input:
Question:
Implement a Python function `get_avg_prices(prices)` that takes a list of prices and returns the average price, rounded to the nearest integer. The function should handle empty lists and lists containing `None` values correctly. Use the `numpy` or `pandas` library functions to calculate the average.\nYour code should pass the following test case:\n```python\nimport numpy as np\nprices = [-100, -200, -300]\nassert get_avg_prices(prices) == -200\n```
Incorrect reasoning path:
<step>\n1. Import numpy library to use its mean function.\n</step>\n<step>\n2. Filter the prices list to remove `None` values.\n- Use list comprehension to create a new list with only the prices.\n</step>\n<step>\n3. Calculate the average price.\n- Use numpy.mean() to calculate the average of the prices.\n- Use round() to round the average to the nearest integer.<step>\n<code>\n```python\nimport numpy as np\n\ndef get_avg_prices(prices):\n    # Remove None values\n    prices = [p for p in prices if p is not None]\n    # Calculate average and round\n    avg_price = round(np.mean(prices))\n    return avg_price\n```\n</code>\n</step>\n
Solution:
import numpy as np\n\ndef get_avg_prices(prices):\n    prices = [price for price in prices if price is not None]\n    if not prices:  # handle empty list\n        return None\n    avg_price = np.mean(prices)\n    return round(avg_price)
Output:
From step <code>, the path is not correct. I will rewrite the reasoning path from step <code>.
<step>\n<code>\n```python\nimport numpy as np\n\ndef get_avg_prices(prices):\n    prices = [price for price in prices if price is not None]\n    if not prices:  # handle empty list\n        return None\n    avg_price = np.mean(prices)\n    return round(avg_price)\n```\n</code>\n</step>\n

Example 2:
Input:
Question:
I'd like to build a function that introspects a function definition and outputs its docstring in a single line. For example, if I have the following function:\n\n```python\ndef add(a, b):\n    \"\"\"Add two numbers together.\"\"\"\n    return a + b\n```\n\nI want the docstring to read:\n\n```python\ndef add(a, b):\n    \"\"\"Return the sum of two numbers.\"\"\"\n```\n\nYour solution should handle functions that call other functions and functions that use f-strings, as well as recursive functions.\nYour code should pass the following test case:\n```python\nimport inspect\ndef recursive_factorial(n):\n    if n == 0:\n        return 1\n    return n * recursive_factorial(n - 1)\nimport os\nos.makedirs('./tmp', exist_ok=True)\nwith open('./tmp/test_introspect_docstring.py', 'w') as f:\n```
Incorrect reasoning path:

Solution:
def introspect_docstring(func):\n    docstring = inspect.getdoc(func)\n    if docstring:\n        return docstring.replace(\"add two numbers\", \"return the sum of two numbers\")\n    return docstring\n
Output:
From step 1, the path is not correct. I will rewrite the reasoning path from step 1.
<step>\n1. Use the inspect module to introspect the function definition.\n</step>\n<step>\n2. Retrieve the docstring of the function.\n</step>\n<step>\n3. Replace \"add two numbers\" with \"return the sum of two numbers\" in the docstring.\n</step>\n<step>\n<code>\n```python\ndef introspect_docstring(func):\n    docstring = inspect.getdoc(func)\n    if docstring:\n        return docstring.replace(\"add two numbers\", \"return the sum of two numbers\")\n    return docstring\n```\n</code>\n</step>\n

Example 3:
Input:
Question:
Write a Python function to randomly generate a list of size `n` with values from 0 to `n - 1`. Next, split the list into two parts, with the first part having the same length as the second part.\n\nFor example, when `n = 6`, the output could be:\n\n```python\n[2, 0, 1]\n[4, 3, 5]\n```\nYour code should pass the following test case:\n```python\nimport random\nfrom typing import List\ndef generate_and_split_list(n: int) -> List[List[int]]:\n    nums = list(range(n))\n    random.shuffle(nums)\n    first_part = nums[: n // 2]\n    second_part = nums[n // 2:]\n    return [first_part, second_part]\nn = 1\nresult = generate_and_split_list(n)\nassert result == [[], [0]]\n```
Incorrect reasoning path:
<step>
1. Import necessary modules (random and typing).
</step>
<step>
2. Define a function generate_and_split_list(n: int) -> List[List[int]].
- Generate a list nums from 0 to n - 1 using list(range(n)).
- Shuffle nums to ensure randomness.
- Split nums into two parts first_part and second_part.
- Return a list containing first_part and second_part.
</step>
<step>
<code>
```python
from typing import List
import random

def generate_and_split_list(n: int) -> List[List[int]]:
    nums = list(range(n))
    random.shuffle(nums)
    first_part = nums[: n // 2]
    second_part = nums[n // 2:]
    return [first_part, second_part]

n = 6
result = generate_and_split_list(n)
assert result == [[2, 0, 1], [4, 3, 5]]
```
</code>
</step>
Solution:
from typing import List
import random
def generate_and_split_list(n: int) -> List[List[int]]:
    nums = list(range(n))
    random.shuffle(nums)
    first_part = nums[: n // 2]
    second_part = nums[n // 2:]
    return [first_part, second_part]
Output:
From step 2, the path is not correct. I will rewrite the reasoning path from step 2.
<step>
2. Define the function generate_and_split_list(n: int) -> List[List[int]].
- Generate a list nums from 0 to n - 1 using list(range(n)).
- Shuffle nums to ensure randomness.
- Split nums into two parts first_part and second_part.
- Return a list containing first_part and second_part.
</step>
<step>
<code>
```python
from typing import List
import random
def generate_and_split_list(n: int) -> List[List[int]]:
    nums = list(range(n))
    random.shuffle(nums)
    first_part = nums[: n // 2]
    second_part = nums[n // 2:]
    return [first_part, second_part]
```
</code>
</step>



Now its your turn:
Input:
Question:
{question}
Incorrect reasoning path:
{instruction}
Solution:
{answer}
Output:
"""
def correction(data, path):
    prompts = []
    for i in data:
        prompts.append(REFINE_PROMPT.format(question=i["problem"], instruction=i["incorrect_path"], answer=i["answer"]))
    
    llm = LLM(
        model='deepseek-ai/deepseek-coder-6.7b-instruct', 
        tensor_parallel_size=len(GPUS), 
        trust_remote_code=True,
        seed=42,
        swap_space=24,
        max_model_len=7000,
        gpu_memory_utilization=0.9
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        use_beam_search=False,
        max_tokens=512, 
        n=1,
        best_of=1,
        stop=['</code>\n</step>']
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params) 
    
    for i,o in zip(data,outputs):
        i['res'] = o.outputs[0].text
    
    with jsonlines.open(os.path.join(path, 'refined_fail.jsonl'),'w') as f:
        f.write_all(data)

path = sys.argv[1]
data = []
with jsonlines.open(os.path.join(path, 'all_fail.jsonl')) as f:
    for i in f:
        data.append(i)
correction(data, path)

