"""
author: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import re
import random
import json
from time import sleep
from openai import OpenAI
from typing import List, Dict, Any, Optional, Type, Tuple, Union


from mcts_math.prompts.prompt_react import PROMPT_REACT
from mcts_math.tools.python_tool import PythonInterpreter
from mcts_math.review_utils import compact_native_think_body
try:
    from shared.prompt_contract import (
        apply_review_prompt_controls,
        build_review_prompt_from_sample,
    )
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from shared.prompt_contract import (
        apply_review_prompt_controls,
        build_review_prompt_from_sample,
    )

from mcts_math.constants import *
from subprocess import TimeoutExpired,Popen,PIPE
import psutil
import tempfile
import os


python_tool_string = f"{PythonInterpreter().name}: {PythonInterpreter().description}"
python_tool_name = PythonInterpreter().name
    

# standard react for Round 1
def react_prompt_wrap(
    question: str, 
    partial_solution: str,
    config,
) -> str:
    prompt_react = PROMPT_REACT(config)
    if '[pass_expand]' in question:
        partial_solution = question.split('[pass_expand]')[1] + partial_solution
        question = question.split('[pass_expand]')[0]
    if partial_solution:
        inputs = f"{question}\nAnswer:\n<step>\n{partial_solution}"  
    else:
        inputs = f"{question}\nAnswer:\n<step>\n"  
    
    react_examples = prompt_react.random_examples()
    react_examples = [f'Example {idx+1}:\n{p}' for idx, p in enumerate(react_examples)]
    assert len(react_examples) > 0, "at least one example should be provided."

    if len(react_examples) > 1:
        example_prefix = "The following are %d demonstration examples." % len(react_examples)
    elif len(react_examples) == 1:
        example_prefix = "The following is a demonstration example."

    format_instructions = prompt_react.react_format_instructions

    prompt = "\n\n".join([format_instructions, example_prefix, *react_examples, prompt_react.react_suffix.format(input=inputs)])
    return prompt


def react_obs_wrap(observation: str) -> str:
    return f"{OBSERVATION}{observation}"


def react_step_result_unwrap(
    text: str,
    final_answer_action: str = FINAL_ANSWER_ACTION,
    action: str = ACTION,
    action_input: str = ACTION_INPUT,
) -> Tuple[str, Dict[str, str]]:
    includes_answer = '<code>' in text
    parser_result = {
        "action": "",
        "action_input": "",
        "final_answer": "",
    }
    if includes_answer:
        parser_result["final_answer"] = text.split('<code>')[-1].split('</code>')[0] 
        return text+'\n</step>\n', parser_result
    else:
        parser_result["action"] = text
        parser_result["action_input"] = ''
        return text, parser_result


# SFT react for Round >1
def react_sft_prompt_wrap(
    question: str, 
    partial_solution: str,
    config, 
    is_value_only=False
) -> str:
    raise RuntimeError("react_sft_prompt_wrap is a retired code-generation prompt path; use review_prompt_wrap.")


def react_sft_obs_wrap(observation: str) -> str:
    return f"{OBSERVATION_LTAG}\n{observation}\n{OBSERVATION_RTAG}\n{STEP_RTAG}"


def react_sft_step_result_unwrap(
    text: str,
    final_answer_action: str = FINAL_ANSWER_ACTION,
) -> Tuple[str, Dict[str, str]]:
    includes_answer = '<code>' in text
    parser_result = { 
        "action": "",
        "action_input": "",
        "final_answer": "",
    }
    
    if includes_answer:
        parser_result["final_answer"] = text.split('<code>')[1].split('</code>')[0] 
        return text+'\n</step>\n', parser_result

    else:
        parser_result["action"] = text
        parser_result["action_input"] = ''
        return text, parser_result


def review_prompt_wrap(
    question: str,
    partial_solution: str,
    config,
    review_context: Dict[str, Any],
    is_value_only: bool = False,
) -> str:
    del is_value_only
    force_final = review_context.get("force_final_review", False)
    sample = {
        "question": question,
        "candidate_code": review_context["candidate_code"],
        "code_language": review_context.get("code_language", "python"),
        "tests_for_prompt": review_context.get("tests_for_prompt", "No tests are available."),
    }
    prompt = build_review_prompt_from_sample(
        sample,
        partial_solution=partial_solution,
        force_final=force_final,
        step_context_mode="assistant_prefix",
        show_tests_in_prompt=bool(review_context.get("show_tests_in_prompt", False)),
        freeform_final_review=False,
    )
    return apply_review_prompt_controls(
        prompt,
        force_final=force_final,
        freeform_final_review=False,
        thinking_mode=getattr(config, "qwen_thinking_mode", ""),
        review_native_thinking_steps=bool(getattr(config, "review_native_thinking_steps", False)),
    )


def _strip_outer_step_block(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<step>"):
        cleaned = cleaned[len("<step>"):].lstrip()
    if cleaned.endswith("</step>"):
        cleaned = cleaned[:-len("</step>")].rstrip()
    return cleaned.strip()


def review_step_result_unwrap(
    text: str,
) -> Tuple[str, Dict[str, str]]:
    cleaned = _strip_outer_step_block(text)
    parser_result = {
        "action": "",
        "action_input": "",
        "final_answer": "",
    }
    if "<review>" in cleaned:
        parser_result["final_answer"] = cleaned.split("<review>", 1)[1].strip()
        return cleaned, parser_result
    if "<think>" in cleaned:
        think_body = cleaned.split("<think>", 1)[1]
        if "</think>" in think_body:
            think_body = think_body.split("</think>", 1)[0]
        think_body = compact_native_think_body(think_body)
        if think_body:
            parser_result["action"] = think_body
            return think_body, parser_result

    parser_result["action"] = cleaned
    return cleaned, parser_result


def extract_code_blocks(text):
    import re
    pattern = r'```(\w+)\n([\s\S]*?)```'
    matches = re.finditer(pattern, text)
    results = []
    response  = ''
    
    for match in matches:
        code = match.group(2)
        response+=code.strip()+'\n'    
    return response


    
def test_pass(test_case, func_str):
    func_str = extract_code_blocks(func_str)
    
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



def math_is_equiv(grt: Union[str, list[str]], prd: str, question: str):
    # return True
    if isinstance(grt, list):
        for g in grt:
            if not test_pass(g, prd):
                return False
        return True
    else:
        return test_pass(grt, prd)
