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


def strip_thinking_switch(prompt: str) -> str:
    stripped = str(prompt or "").rstrip()
    lower = stripped.lower()
    for suffix in ("/no_think", "/think"):
        if lower.endswith(suffix):
            return stripped[: -len(suffix)].rstrip()
    return str(prompt or "")


def chat_messages_for_prompt(prompt: str) -> list[dict[str, str]]:
    content = strip_thinking_switch(prompt)
    if "@@ Response" not in content:
        return [{"role": "user", "content": content}]

    instruction, response = content.split("@@ Response", 1)
    user_content = instruction.rstrip() + "\n\n@@ Response"
    response_prefix = response.lstrip("\n").rstrip()
    if not response_prefix:
        return [{"role": "user", "content": user_content}]

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response_prefix},
    ]


def prompt_parts_for_active_assistant_prefix(prompt: str) -> tuple[str, str]:
    """Split a raw ``@@ Response`` prompt into user content and active prefix.

    ``@@ Response`` is an internal separator, not a model API feature.  For
    chat-template models, response text after the separator must be appended
    after the assistant generation marker so vLLM continues from it.  Passing it
    as a completed assistant message would open a new assistant turn and allow
    the model to restart or repeat the prefix.
    """
    content = strip_thinking_switch(prompt)
    if "@@ Response" not in content:
        return content, ""

    instruction, response = content.split("@@ Response", 1)
    user_content = instruction.rstrip() + "\n\n@@ Response"
    response_prefix = response.lstrip("\n").rstrip()
    return user_content, response_prefix


def maybe_apply_chat_template(prompts: List[str], engine: LLM, config: Any | None) -> List[str]:
    if config is None or not getattr(config, "use_chat_template", False):
        return prompts
    tokenizer = engine.get_tokenizer()
    rendered = []
    for prompt in prompts:
        user_content, response_prefix = prompt_parts_for_active_assistant_prefix(prompt)
        messages = [{"role": "user", "content": user_content}]
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": chat_template_thinking_enabled(prompt, config),
        }
        try:
            rendered_prompt = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            rendered_prompt = tokenizer.apply_chat_template(messages, **kwargs)
        if response_prefix:
            rendered_prompt += response_prefix
        rendered.append(rendered_prompt)
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
