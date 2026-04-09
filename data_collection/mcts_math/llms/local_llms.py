import os
import time
import math
import copy

from typing import Optional, Any, Dict, List, Callable, Type, Tuple

from vllm import LLM, SamplingParams
from vllm.outputs import CompletionOutput, RequestOutput
from mcts_math.prompts.prompt_sft import QWEN_DIRECT_PROMPT


def local_vllm(
    prompt: str,
    llm: LLM,
    sampling_params: SamplingParams,
    n: int,
    temperature: float,
    with_value: bool = False,
) -> List[str]:  
    """
    This one is not for batch inference.
    """
    # update args
    sampling_params.n = n
    sampling_params.temperature = temperature
    # n samples for each prompt
    prompts = [prompt]
    outputs = llm.generate(prompts, sampling_params=sampling_params)    # return List[RequestOutput]
    # len(prompts) = 1,  we take the first one RequestOutput. 
    output = outputs[0]
    completion_outputs = output.outputs                                 # return List[CompletionOutput], where len() = sampling_params.n
    if with_value:
        return completion_outputs, output.value_estimate  # for sbs, mcts
    else:
        return [co.text for co in completion_outputs]


def server_generator(
    prompts: List[str],
    engine: Any,
):
    vllm_outputs = []
    for prompt in prompts:
        responses = engine(prompt)
        output = RequestOutput(request_id=str(time.time()),
                               prompt=prompt,
                               prompt_token_ids=[],
                               prompt_logprobs=-1,
                               outputs=[CompletionOutput(index=idx, text=response, token_ids=[], cumulative_logprob=-1, logprobs=-1) 
                                        for idx, response in enumerate(responses)],
                               finished=True)
        vllm_outputs.append(output)
    return vllm_outputs


def local_generator(
    prompts: List[str],
    sampling_params: SamplingParams,
    engine: LLM,
):
    
    if "</step>\n\nHere is your question:" in prompts[0]:
        direct_prompts = []
        step_prompts = []
        for prompt in prompts:
            if prompt.split('@@ Response')[1].count('<step>')==1:
                step_prompts.append(prompt)
                question = prompt.split("</step>\n\nHere is your question:")[1].split('@@ Response')[0].strip()
                partial_solution = '```python'
                direct_prompts.append(QWEN_DIRECT_PROMPT.format(question=question, partial_solution=partial_solution))
    
    
        if len(prompts)==len(step_prompts) and sampling_params.max_tokens>1 and sampling_params.n>1:
            sampling_number = sampling_params.n
    
            step_sampling_params = copy.deepcopy(sampling_params)
            step_sampling_params.n=math.ceil(sampling_number/2)
            step_outputs = engine.generate(step_prompts, sampling_params=step_sampling_params)
    
            direct_sampling_params = copy.deepcopy(sampling_params)
            direct_sampling_params.n=math.floor(sampling_number/2)
            direct_sampling_params.stop = ['```','\n```','```\n','```\n\n','```\n\n\n']
            direct_outputs = engine.generate(direct_prompts, sampling_params=direct_sampling_params)
            
            outputs = []
            for i in direct_outputs:
                for idx in range(len(i.outputs)):
                    i.outputs[idx].text = '<code>\n```python\n'+ i.outputs[idx].text.strip()+'\n```\n</code>'
            for i,j in zip(direct_outputs,step_outputs):
                i.outputs+=j.outputs
                outputs.append(i)
        else:
            outputs = engine.generate(prompts, sampling_params=sampling_params)
    else:
        direct_prompts = []
        step_prompts = []
        for prompt in prompts:
            if prompt.split('@@ Response')[1].count('<step>')==1:
                step_prompts.append(prompt)
                direct_prompts.append(prompt.strip()+'\n<code>\n```python\n')
    
    
        if len(prompts)==len(step_prompts) and sampling_params.max_tokens>1 and sampling_params.n>1:
            sampling_number = sampling_params.n
    
            step_sampling_params = copy.deepcopy(sampling_params)
            step_sampling_params.n=math.ceil(sampling_number/2)
            step_outputs = engine.generate(step_prompts, sampling_params=step_sampling_params)
    
            direct_sampling_params = copy.deepcopy(sampling_params)
            direct_sampling_params.n=math.floor(sampling_number/2)
            direct_outputs = engine.generate(direct_prompts, sampling_params=direct_sampling_params)
            
            outputs = []
            for i in direct_outputs:
                for idx in range(len(i.outputs)):
                    i.outputs[idx].text = '<code>\n```python\n'+i.outputs[idx].text.strip()
            for i,j in zip(direct_outputs,step_outputs):
                i.outputs+=j.outputs
                outputs.append(i)
        else:
            outputs = engine.generate(prompts, sampling_params=sampling_params)



    outputs = engine.generate(prompts, sampling_params=sampling_params)    
    
    
    return outputs
