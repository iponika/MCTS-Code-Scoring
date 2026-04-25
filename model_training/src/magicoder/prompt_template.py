# Used to format the `{instruction}` part above
# SRC_INSTRUCT_INSTRUCTION_PROMPT = """Write a solution to the following coding problem:
# {problem}"""
SRC_INSTRUCT_INSTRUCTION_PROMPT = """{problem}"""

# Used to format src-instruct data points
SRC_INSTRUCT_ILLUSTRATION_PROMPT = """[Problem]
{problem}

[Solution]
{solution}"""




SPHE_PROMPT =  '''"You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. 
@@ Instruction
Please solve the programming question step-by-step.

Here are some examples:

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
    """
```
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
    
    return result
```

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
    """
```
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
    return len(words[-1]) == 1
```

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
    """
```
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
        
    return 'Yes'
```

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
    """
```
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
    return mangoes
```

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
    """
```
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
                return num
```

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
    """
```
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
    return orig_a if a > b else orig_b
```

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
    """
```
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
            
    return result
```

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
    """
```
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
            
    return result
```

Now, it's your turn:
{instruction}

@@ Response
{response}
'''



SPMP_PROMPT =  '''You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. 
@@ Instruction
Please solve the programming question step-by-step.

Here are some examples:

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
    return total_length
```


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
    return result
```


Write a function to find the lateral surface area of a cone.
Let's think step by step.
Step 1. Calculate the generatrix of the cone.
Step 2. Return the result.
Step 3. Please import inside the function.
Step ```python
def lateral_surface_area_cone(radius, height):
    from math import pi, sqrt
    generatrix = sqrt(radius**2 + height**2)
    return pi * radius * generatrix
```


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
    return result
```


Write a python function to find the last two digits in factorial of a given number.
Let's think step by step.
Step 1. Calculate the factorial of the input number.
Step 2. Return the last two digits of it.
Step ```python
def last_two_digits_factorial(n):
    import math
    factorial = math.factorial(n)
    return f"{{factorial % 100:02d}}"
```


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
    return result
```


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
    return result
```

Now, it's your turn:
{instruction}

@@ Response
{response}
'''



COTHE_PROMPT =  '''You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. 
@@ Instruction
Please solve the programming question step-by-step.

Here is an examples:

Question: Write a solution to the following problem:
```python
def encrypt(s):
    """
    Create a function encrypt that takes a string as an argument and returns a string encrypted with the alphabet being
    rotated. The alphabet should be rotated in a manner such that the letters shift down by two multiplied to two places.
    For example:
    encrypt(’hi’) returns ’lm’
    encrypt(’asdfghjkl’) returns ’ewhjklnop’
    encrypt(’gf’) returns ’kj’
    encrypt(’et’) returns ’ix’
    """
```
Let’s think step by step.
1. Create a string ”alphabet” with all letters of the alphabet.
2. Assign the number of places to shift the letters to a variable ”bias”.
3. Initialize a string ”result” with an empty string.
4. Iterate over the characters of the string ”s”.
5. Find the index of the character in the string ”alphabet”.
6. Add the number of places to shift the letters to the index.
7. If the index is larger than 25, subtract 26 from the index.
8. Add the character at the index to the string ”result”.
9. Return the string ”result”.
```python
def encrypt(s):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    result = ''
    
    for char in s:
        idx = alphabet.index(char)
        # Shift 4 places (2*2) and get corresponding letter
        new_idx = (idx - 4) % 26
        result += alphabet[new_idx]
    
    return result
```

Now, it's your turn:
{instruction}

@@ Response
{response}
'''


COTMP_PROMPT =  '''"You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. 
@@ Instruction
Please solve the programming question step-by-step.

Here is an examples:

Create a function encrypt that takes a string as an argument and returns a string encrypted with the alphabet being
rotated. The alphabet should be rotated in a manner such that the letters shift down by two multiplied to two places.
Let’s think step by step.
1. Create a string ”alphabet” with all letters of the alphabet.
2. Assign the number of places to shift the letters to a variable ”bias”.
3. Initialize a string ”result” with an empty string.
4. Iterate over the characters of the string ”s”.
5. Find the index of the character in the string ”alphabet”.
6. Add the number of places to shift the letters to the index.
7. If the index is larger than 25, subtract 26 from the index.
8. Add the character at the index to the string ”result”.
9. Return the string ”result”.
```python
def encrypt(s):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    result = ''
    
    for char in s:
        idx = alphabet.index(char)
        # Shift 4 places (2*2) and get corresponding letter
        new_idx = (idx - 4) % 26
        result += alphabet[new_idx]
    
    return result
```

Now, it's your turn:
{instruction}

@@ Response
{response}
'''



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


QWEN_DIRECT_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. 
@@ Instruction
You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. 
{instruction}

@@ Response
{response}"""


QWEN_REVIEW_STEP_PROMPT = """You are an exceptionally intelligent code reviewer.
@@ Instruction
You will be given one code scoring task.
Think through functional correctness and AXIOM repair effort in concise <step>...</step> blocks, then finish with exactly one <review> JSON block.
Keep the reasoning grounded in the task, candidate code, and available tests. Do not output code fixes.
The project goal is scalar code scoring; textual critique is only supporting evidence.
AXIOM grade semantics: 5=production-ready; 4=functionally correct with minor quality tweaks; 3=functionally correct but major quality refactor needed; 2=functionally defective but minor fix; 1=functionally defective and major repair; 0=fundamentally flawed or mismatched. Functionality is the primary boundary: grades 3-5 are functionally correct, grades 0-2 are not.
Calibration rule: do not assign grades 0-2 merely because an issue is suspected or because no tests are available. Low grades require concrete visible evidence such as a syntax/runtime error, missing required I/O, unrelated or empty code, a direct contradiction of the task, or a simple counterexample grounded in the prompt/tests. If the implementation is complete and plausibly functional but you cannot prove a functional defect, keep the grade in 3-5 and use repair_effort to express quality/refactoring concerns.

{instruction}

Final review format:
<review>
{{"axiom_grade": <0-5 integer>, "score": <0-100 number>, "verdict": "accept|minor_issue|major_issue", "functional_correctness": true, "repair_effort": "none|minor_quality|major_quality|minor_functional|major_functional|rewrite", "summary": "...", "evidence": ["...", "..."]}}
</review>

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
