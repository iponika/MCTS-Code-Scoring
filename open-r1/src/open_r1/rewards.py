"""Reward functions for GRPO training."""

import math
import os
import re
from typing import Dict
from subprocess import TimeoutExpired,Popen,PIPE
import psutil
import tempfile

from latex2sympy2_extended import NormalizationConfig
#from math_verify import LatexExtractionConfig, parse, verify


    
def test_pass(func_str, test_case):    
    try:
        func_name = func_str.split('def')[1].split('(')[0].strip()
        refined_test = []
        for line in test_case.splitlines():
            if line.startswith('from tmp import') or line.startswith('from your_module import'):
                continue
            if 'import' in line and func_name in line:
                continue
            refined_test.append(line)
        extracted_test = '\n'.join(refined_test)
    
        code = func_str + "\n" + extracted_test
        
        def kill(proc_pid:int):
            process= psutil.Process(proc_pid)
            
            for proc in process.children(recursive=True):
                proc.kill()
            process.kill()
        
        def pexec(strcmd:str,n_timeout:int):
            try:
                p=Popen(strcmd,shell=True, stdout=PIPE, stderr=PIPE)
                stdout, stderr = p.communicate(timeout=n_timeout)
                return stdout.decode('utf-8'), stderr.decode('utf-8'), p.returncode
            except TimeoutExpired:
                kill(p.pid)
                return None, None,-1
            except Exception as e:
                return None, None,-1
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, dir='/tmp') as f:
            f.write(code.encode('utf-8'))
            temp_path = f.name
    
        temp_dir = os.path.dirname(temp_path)
        cmd = f'cd {temp_dir} ; python {temp_path}'
        stdout, stderr, returncode = pexec(cmd, 3)
    
        os.unlink(temp_path)
        test_pass = returncode == 0 and 'AssertionError' not in stderr
        return test_pass
    except:
        return False


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        if '<answer>' in content and '</answer>' in content:
            content = content.split('<answer>')[1].split('</answer>')[0]
        if '<code>' in content and '</code>' in content:
            content = content.split('<code>')[1].split('</code>')[0]

        if '```python' in content:
            code = content.split('```python')[1].split('```')[0]
        elif content.count('```')==2:
            code = content.split('```')[1]
        else:
            code = content

        reward=1
        for test_case in sol.split('[test_case]'):
            if not test_pass(code, test_case):
                reward=0
                break
        rewards.append(reward)

    return rewards


def validate_step_format(text):
    steps = text.strip().split('</step>')
    steps = [s.strip() for s in steps if s.strip()]
    
    if not steps:
        return False
        
    prev_num = None
    
    for i, step in enumerate(steps):
        if not step.startswith('<step>'):
            return False
            
        content = step[6:].strip()
        
        if '```' in content:
            if '<code>' not in content:
                return False
            if i!=len(steps)-1:
                return False
        else:
            first_line = content.split('\n')[0].strip()
            if not first_line or not first_line[0].isdigit():
                return False
            current_num = int(first_line[0])
            if prev_num is not None and current_num != prev_num + 1:
                return False
            prev_num = current_num
    return True



def format_step_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # reward = []
    # for completion in completions:
    #     if validate_step_format(completion[0]['content']):
    #         reward.append(1)
    #     else:
    #         reward.append(-1)
    # # print(completions)
    # # print(reward)
    # return reward
    pass


def format_think_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0 for match in matches]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solutions: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solutions: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solutions):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward
