import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from tqdm.auto import tqdm
from transformers import HfArgumentParser
import jsonlines
import json

from experiments.utils import wget
from magicoder.llm_wrapper import GenerationConfig, get_model_context
from magicoder.prompt_template import QWEN_STEP_PROMPT, QWEN_DIRECT_PROMPT, DSC_PROMPT, SPHE_PROMPT, SPMP_PROMPT, COTHE_PROMPT, COTMP_PROMPT
from magicoder.utils import chunked, read_jsonl


class Text2CodeProblem(TypedDict):
    id: str
    instruction: str
    response_prefix: str

prompt_method = ''
model_key = ''


def get_mbpp_raw_problems() -> list[dict]:
    problems = get_mbpp_plus()
    return list(problems.values())


def get_humaneval_raw_problems() -> list[dict]:
    problems = get_human_eval_plus()
    return list(problems.values())

def get_lcb_raw_problems() -> list[dict]:
    problems = []
    with jsonlines.open('./data/full_problems.jsonl') as f:
        for i in f:
            problems.append({"task_id":i["task_id"], 'prompt':i['question'], 'starter_code':i['starter_code']})

    return problems


def map_mbpp_problem(p: dict) -> Text2CodeProblem:
    id = p["task_id"]
    prompt = p["prompt"]
    start_index = prompt.index('"""')
    end_index = prompt.rindex('"""')
    prompt = prompt[start_index + 3 : end_index]
    assert_index = prompt.index("assert")
    instruction = prompt[:assert_index].strip()
    if not instruction.endswith("."):
        instruction += "."
    assertion = prompt[assert_index:].strip()
    instruction = f"""{instruction} Your code should satisfy the following assertion:
```python
{assertion}
```"""
    if prompt_method == 'direct' and ('deepseek' in model_key or 'dsc' in model_key):
        response_prefix = f"""<step>\n<code>\n```python\n"""
    elif prompt_method == 'direct':
        response_prefix = f"""```python"""
    else:
        response_prefix = f"""<step>\n"""
    return Text2CodeProblem(
        id=str(id), instruction=instruction, response_prefix=response_prefix
    )


def map_humaneval_problem(p: dict) -> Text2CodeProblem:
    id = p["task_id"]
    prompt = p["prompt"]
    prompt = prompt.strip()
    instruction = f"""Write a solution to the following problem:\n
```python
{prompt}
```"""
    if prompt_method == 'direct' and ('deepseek' in model_key or 'dsc' in model_key):
        response_prefix = f"""<step>\n<code>\n```python\n"""
    elif prompt_method == 'direct':
        response_prefix = f"""```python"""
    else:
        response_prefix = f"""<step>\n"""
    return Text2CodeProblem(
        id=id, instruction=instruction, response_prefix=response_prefix
    )


def map_lcb_problem(p: dict) -> Text2CodeProblem:
    id = p["task_id"]
    prompt = p["prompt"].strip()
    starter_code = p["starter_code"]
    instruction = f"""{prompt}\n""" 
    
    if len(starter_code.strip())>0:
        instruction += f"Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
        instruction += f"```python\n{starter_code}\n```\n"
    else:
       instruction += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters."
    
    if prompt_method == 'direct' and ('deepseek' in model_key or 'dsc' in model_key):
        response_prefix = f"""<step>\n<code>\n```python\n"""
    elif prompt_method == 'direct':
        response_prefix = f"""```python"""
    else:
        response_prefix = f"""<step>"""

    return Text2CodeProblem(
        id=id, instruction=instruction, response_prefix=response_prefix
    )



@dataclass(frozen=True)
class Args:
    model_key: str
    dataset: Literal["humaneval", "mbpp", "lcb"]
    save_path: str

    n_batches: int
    n_problems_per_batch: int
    n_samples_per_problem: int
    # prompted: bool

    model_name_or_path: str | None = None
    prompt: str | None = None


def extract(code, lang):
    delim = f"""```{lang}"""
    if '<code>' in code and '</code>' in code:
        code = code.split('<code>')[1].split('</code>')[0]
    if delim in code:
        code = code.split(delim)[-1].split('```')[0]
    else:
        code = code.split('```')[0]
    return code


def main():
    parser = HfArgumentParser((Args, GenerationConfig))
    args, generation_config = cast(
        tuple[Args, GenerationConfig],
        parser.parse_args_into_dataclasses(),
    )

    global prompt_method
    global model_key
    prompt_method = args.prompt
    model_key =  args.model_key
    if 'deepseek' in args.model_key:
        prompt_template = DSC_PROMPT
    else:
        if args.prompt == 'step':
            prompt_template = QWEN_STEP_PROMPT
        elif args.prompt == 'direct':
            prompt_template = QWEN_DIRECT_PROMPT


    if args.dataset == "humaneval":
        raw_problem_fn, map_problem_fn = (get_humaneval_raw_problems, map_humaneval_problem)
    elif args.dataset == "mbpp":
        raw_problem_fn, map_problem_fn = (get_mbpp_raw_problems, map_mbpp_problem)
    elif args.dataset == "lcb":
        raw_problem_fn, map_problem_fn = (get_lcb_raw_problems, map_lcb_problem)


    raw_problems = raw_problem_fn()
    problems = list(map(map_problem_fn, raw_problems))
    problem_dict = {}
    if args.dataset == "humaneval":
        for i in problems:
            problem_dict[i['id']] = i['instruction'].split('def ')[0].split('```python')[1]

    state = get_model_context(args.model_key, args.model_name_or_path)

    problems_chunked = list(chunked(list(problems), args.n_problems_per_batch))
    iter = itertools.product(problems_chunked, range(args.n_batches))
    n_total = len(problems_chunked) * args.n_batches

    Path(args.save_path).write_text("")
    for problems, batch_idx in tqdm(iter, total=n_total):
        task_ids = [problem["id"] for problem in problems]
        prompts = [
            # TODO: make it generic for all models
            prompt_template.format(
                instruction=problem["instruction"], response=problem["response_prefix"]
            )
            for problem in problems
        ]
        all_prompts = prompts * args.n_samples_per_problem
        all_task_ids = task_ids * args.n_samples_per_problem
        response = state.complete(generation_config, all_prompts)
        completions = response.decoded_outputs
        assert len(problems) <= args.n_problems_per_batch
        assert len(completions) == len(problems) * args.n_samples_per_problem
        if args.dataset == "humaneval":
            samples = [
                dict(
                    task_id=task_id,
                    solution=problem_dict[task_id]+'\n'+extract(completion, 'python'),
                    original_completion=completion
                )
                for task_id, completion in zip(all_task_ids, completions)
            ]
        else:
            samples = [
                dict(
                    task_id=task_id,
                    solution=extract(completion, 'python'),
                    original_completion=completion
                )
                for task_id, completion in zip(all_task_ids, completions)
            ]
        write_jsonl(args.save_path, samples, append=True)


if __name__ == "__main__":
    main()
