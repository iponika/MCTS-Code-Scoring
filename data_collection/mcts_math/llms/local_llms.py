import os
import time

from typing import Optional, Any, Dict, List, Callable, Type, Tuple

from vllm import LLM, SamplingParams
from vllm.outputs import CompletionOutput, RequestOutput


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


def chat_template_thinking_enabled(prompt: str, config: Any) -> bool:
    prompt_tail = str(prompt or "").rstrip().lower()
    if prompt_tail.endswith("/no_think"):
        return False
    if prompt_tail.endswith("/think"):
        return True
    return bool(getattr(config, "chat_template_enable_thinking", True))


def maybe_apply_chat_template(prompts: List[str], engine: LLM, config: Any | None) -> List[str]:
    if config is None or not getattr(config, "use_chat_template", False):
        return prompts
    tokenizer = engine.get_tokenizer()
    rendered = []
    for prompt in prompts:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": chat_template_thinking_enabled(prompt, config),
        }
        try:
            rendered.append(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs))
        except TypeError:
            kwargs.pop("enable_thinking", None)
            rendered.append(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs))
    return rendered


def local_generator(
    prompts: List[str],
    sampling_params: SamplingParams,
    engine: LLM,
    config: Any | None = None,
):
    prompts = maybe_apply_chat_template(prompts, engine, config)
    if config is not None and getattr(config, "use_chat_template", False):
        return engine.generate(prompts, sampling_params=sampling_params)
    if not prompts or any('@@ Response' not in prompt for prompt in prompts):
        return engine.generate(prompts, sampling_params=sampling_params)
    return engine.generate(prompts, sampling_params=sampling_params)
