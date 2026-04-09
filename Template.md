# Prompt Template

## Prompt Template for path refinement

```
You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. 
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
assert result == [[2, 0, 1], [4, 3, 5]]```
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
    return [first_part, second_part]```
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
```

## In-context examples used in self-planning (HumanEval)
```
Question: Write a solution to the following problem:
```python
def encrypt(s):
    """
    Create a function encrypt that takes a string as an argument and returns a string encrypted with the alphabet being
    rotated. The alphabet should be rotated in a manner such that the letters shift down by two multiplied to two places.
    
    For example:
    encrypt('m') returns 'lm'
    encrypt('selfhglpj') returns 'ewhjgklnop' 
    encrypt('hi') returns 'kj'
    encrypt('gl') returns 'kj'
    encrypt('at') returns 'ix'
    """\n```
Let's think step by step:
1. Create a alphabet, bias two places multiplied by two.
2. Loop the input, find the latter bias letter in alphabet.
3. Return result.
```python
def encrypt(s):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    result = ''
    
    for char in s:
        idx = alphabet.index(char)
        # Shift 4 places (2*2) and get corresponding letter
        new_idx = (idx - 4) % 26
        result += alphabet[new_idx]
    
    return result\n```

Question: Write a solution to the following problem:
```python
def check_if_last_char_is_a_letter(txt):
    """
    Create a function that returns True if the last character of a given string is an alphabetical character and is not a
    part of a word, and False otherwise. Note: "word" is a group of characters separated by space.
    
    Examples:
    check_if_last_char_is_a_letter("apple pie") → False
    check_if_last_char_is_a_letter("apple pi c") → True
    check_if_last_char_is_a_letter("apple pi c ") → False
    check_if_last_char_is_a_letter("") → False
    """\n```
Let's think step by step:
1. If the string is empty, return False.
2. If the string does not end with a alphabetical character, return False.
3. Split the given string into a list of words.
4. Check if the length of the last word is equal to 1.
```python
def check_if_last_char_is_a_letter(txt):
    if not txt or txt[-1].isspace():
        return False
    
    if not txt[-1].isalpha():
        return False
        
    words = txt.split()
    return len(words[-1]) == 1\n```

Question: Write a solution to the following problem:
```python
def file_name_check(file_name):
    """
    Create a function which takes a string representing a file's name, and returns 'Yes' if the file's name is valid,
    and returns 'No' otherwise. A file's name is considered to be valid if and only if all the following conditions are met:
    - There should not be more than three dots ('0'-'9') in the file's name. - The file's name contains exactly one dot.
    - The substring before the dot should not be empty, and it starts with a letter from the latin alphabet ('a'-'z' and
    'A'-'Z'). - The substring after the dot should be one of these: ['txt', 'exe', 'dll']
    
    Examples:
    file_name_check("example.txt") => 'Yes'
    file_name_check("1example.dll") => 'No' (the name should start with a latin alphabet letter)
    """\n```
Let's think step by step:
1. Check if the file name is valid according to the conditions.
2. Return "Yes" if valid, otherwise return "NO".
```python
def file_name_check(file_name):
    if file_name.count('.') != 1:
        return 'No'
        
    name, ext = file_name.split('.')
    
    if not name or not name[0].isalpha():
        return 'No'
        
    if ext not in ['txt', 'exe', 'dll']:
        return 'No'
        
    return 'Yes'\n```

Question: Write a solution to the following problem:
```python
def fruit_distribution(s, n):
    """
    In this task, you will be given a string that represents a number of apples and oranges that are distributed in a
    basket of fruit this basket contains apples, oranges, and mango fruits. Given the string that represent the total number
    of the oranges and apples and an integer that represent the total number of the fruits in the basket return the number
    of the mango fruits in the basket.
    
    for example:
    fruit_distribution("5 apples and 6 oranges", 19) → 19 - 5 - 6 = 8
    fruit_distribution("0 apples and 1 oranges",3) → 3 - 0 - 1 = 2
    fruit_distribution("2 apples and 3 oranges", 100) → 100 - 2 - 3 = 95
    fruit_distribution("100 apples and 1 oranges",120) → 120 - 100 - 1 = 19
    """\n```
Let's think step by step:
1. Extract the numbers of oranges and apples from given string.
2. Calculate the sum of oranges and apples.
3. Deduct the sum from the total number of fruits.
4. Return the number of mangoes.
```python
def fruit_distribution(s, n):
    # Extract numbers using string operations
    words = s.split()
    apples = int(words[0])
    oranges = int(words[3])
    
    # Calculate mangoes
    mangoes = n - apples - oranges
    return mangoes\n```

Question: Write a solution to the following problem:
```python
def prime_fib(n):
    """
    prime_fib returns n-th number that is a Fibonacci number and it's also prime.
    
    Examples:
    >>> prime_fib(1) 2
    >>> prime_fib(2) 3
    >>> prime_fib(3) 5
    >>> prime_fib(4) 13
    >>> prime_fib(5) 89
    """\n```
Let's think step by step:
1. Create a function to check if a number is prime.
2. Generate a Fibonacci sequence number.
3. Check if each number in the Fibonacci sequence is prime, decrement the counter.
4. If the counter is 0, return the Fibonacci number.
```python
def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

def fib():
    a, b = 0, 1
    while True:
        yield b
        a, b = b, a + b
    
def prime_fib(n):
    count = 0
    for num in fib():
        if is_prime(num):
            count += 1
            if count == n:
                return num\n```

Question: Write a solution to the following problem:
```python
def compare_one(a, b):
    """
    Create a function that takes integers, floats, or strings representing real numbers, and returns the larger variable
    in its given variable type. Return None if the values are equal. Note: If a real number is represented as a string, the
    floating point might be '.' or ','.
    
    Examples:
    compare_one(1, 2.5) → 2.5
    compare_one(1, "2.3") → "2.3"
    compare_one("5.1", "6") → "6"
    compare_one("1", 1) → None
    """\n```
Let's think step by step:
1. Store the original inputs.
2. Check if inputs are strings and convert to floats.
3. Compare the two inputs and return the larger one in its original data type.
```python
def compare_one(a, b):
    # Store original inputs
    orig_a, orig_b = a, b
    
    # Convert strings to floats if needed
    if isinstance(a, str):
        a = float(a.replace(',', '.'))
    if isinstance(b, str):
        b = float(b.replace(',', '.'))
    
    # Convert to float for comparison
    a, b = float(a), float(b)
    
    if a == b:
        return None
    return orig_a if a > b else orig_b\n```

Question: Write a solution to the following problem:
```python
def sort_even(l: list):
    """
    This function takes a list l and returns a list l' such that l' is identical to l in the odd indices, while its values at
    the even indices are equal to the values of the even indices of l but sorted.
    
    Examples:
    >>> sort_even([1, 2, 3])
    [1, 2, 3]
    >>> sort_even([5, 6, 3, 4])
    [3, 6, 5, 4]
    """\n```
Let's think step by step:
1. Create a list of all the even indices of the given list.
2. Sort the list of even indices.
3. Return a new list that is identical to the original list in the odd indices, and equal to the sorted even indices
in the even indices.
```python
def sort_even(l: list):
    # Get even-indexed elements
    even_indices = [l[i] for i in range(0, len(l), 2)]
    even_indices.sort()
    
    # Create result list
    result = []
    even_idx = 0
    
    for i in range(len(l)):
        if i % 2 == 0:
            result.append(even_indices[even_idx])
            even_idx += 1
        else:
            result.append(l[i])
            
    return result\n```

Question: Write a solution to the following problem:
```python
def search(lst):
    """
    You are given a non-empty list of positive integers. Return the greatest integer that is greater than zero, and has
    a frequency greater than or equal to the value of the integer itself. The frequency of an integer is the number of times
    it appears in the list. If no such value exist, return -1.
    
    Examples:
    search([4, 1, 2, 2, 3, 1]) == 2
    search([1, 2, 2, 3, 3, 3, 4, 4, 4]) == 3
    search([5, 5, 4, 4, 4]) == -1
    """\n```
Let's think step by step:
1. Create a frequency dict.
2. Sort the input list.
3. Loop the input list, if frequency no lesser than the integer, set result.
4. Return the result.
```python
def search(lst):
    # Create frequency dictionary
    freq = {{}}
    for num in lst:
        freq[num] = freq.get(num, 0) + 1
    
    result = -1
    # Check each number
    for num in sorted(lst, reverse=True):
        if freq[num] >= num:
            result = num
            break
            
    return result\n```
```




## In-context examples used in self-planning (MBPP)
```
Write a function to sum the length of the names of a given list of names after removing the names that start with a
lowercase letter.
Let's think step by step.
Step 1. Loop the input list.
Step 2. If the name not start with lowercase letter, add the length of the name to result.
Step 3. Return the result.
Step ```python
def sum_uppercase_names(names):
    total_length = 0
    for name in names:
        if name[0].isupper():
            total_length += len(name)
    return total_length\n```


Write a function to increment the numeric values in the given strings by k.
Let's think step by step.
Step 1. Loop the input list.
Step 2. If a string is a number, increment it.
Step 3. Return modified list.
Step ```python
def increment_numbers(strings, k):
    result = []
    for s in strings:
        if s.isdigit():
            new_value = str(int(s) + k)
            result.append(new_value)
        else:
            result.append(s)
    return result\n```


Write a function to find the lateral surface area of a cone.
Let's think step by step.
Step 1. Calculate the generatrix of the cone.
Step 2. Return the result.
Step 3. Please import inside the function.
Step ```python
def lateral_surface_area_cone(radius, height):
    from math import pi, sqrt
    generatrix = sqrt(radius**2 + height**2)
    return pi * radius * generatrix\n```


Write a function to remove all tuples with all none values in the given tuple list.
Let's think step by step.
Step 1. Loop the given tuple list.
Step 2. Check if all elements in the tuple are None.
Step 3. If not, append the tuple to the result list.
Step 4. Return the result.
Step ```python
def remove_none_tuples(tuple_list):
    result = []
    for t in tuple_list:
        if not all(x is None for x in t):
            result.append(t)
    return result\n```


Write a python function to find the last two digits in factorial of a given number.
Let's think step by step.
Step 1. Calculate the factorial of the input number.
Step 2. Return the last two digits of it.
Step ```python
def last_two_digits_factorial(n):
    import math
    factorial = math.factorial(n)
    return f"{{factorial % 100:02d}}"\n```


Write a python function to replace multiple occurence of character by single.
Let's think step by step.
Step 1. Create a pattern that the input character repeats mulitiple times.
Step 2. Replace the pattern in input string with input character.
Step 3. Please import inside the function.
Step ```python
def remove_multiple_chars(input_string, char):
    import re
    pattern = re.compile(char + '+')
    result = re.sub(pattern, char, input_string)
    return result\n```


Write a python function to move all zeroes to the end of the given list.
Let's think step by step.
Step 1. Count the number of zeros.
Step 2. Remove the zeros from the list.
Step 3. Append the zeros to the end of the list.
Step 4. Return the list
Step ```python
def move_zeros_to_end(lst):
    zero_count = lst.count(0)
    result = [x for x in lst if x != 0]
    result.extend([0] * zero_count)
    return result\n```

```
