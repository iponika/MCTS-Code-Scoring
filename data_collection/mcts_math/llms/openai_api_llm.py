from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from vllm.outputs import CompletionOutput, RequestOutput


@dataclass
class ApiSamplingParams:
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 512
    n: int = 1
    best_of: int = 1
    stop: list[str] | str | None = None


def build_api_sampling_params(config: Any) -> ApiSamplingParams:
    stop = config.stop
    if stop is not None and not isinstance(stop, (str, list)):
        stop = list(stop)
    return ApiSamplingParams(
        temperature=float(config.temperature),
        top_p=float(config.top_p),
        max_tokens=int(config.max_tokens),
        n=int(config.n_generate_sample),
        best_of=int(config.n_generate_sample),
        stop=stop,
    )


class OpenAICompatibleGenerator:
    def __init__(self, config: Any) -> None:
        base_url = os.environ.get(config.api_base_url_env, "").rstrip("/")
        api_key = os.environ.get(config.api_key_env, "")
        model = os.environ.get(config.api_model_env) or config.model_dir
        if not base_url:
            raise RuntimeError(f"Missing API base URL env var: {config.api_base_url_env}")
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {config.api_key_env}")
        if not model:
            raise RuntimeError(f"Missing API model; set model_dir or {config.api_model_env}")

        self.url = f"{base_url}/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = int(config.api_timeout)
        self.max_retries = int(config.api_max_retries)
        self.retry_sleep = float(config.api_retry_sleep)

    def __call__(self, prompts: list[str], sampling_params: ApiSamplingParams) -> list[RequestOutput]:
        return [self._generate_one(prompt, sampling_params) for prompt in prompts]

    def _generate_one(self, prompt: str, sampling_params: ApiSamplingParams) -> RequestOutput:
        n = max(1, int(getattr(sampling_params, "n", 1) or 1))
        try:
            texts = self._request(prompt, sampling_params, n=n)
        except RuntimeError:
            if n == 1:
                raise
            texts = []
            for _ in range(n):
                texts.extend(self._request(prompt, sampling_params, n=1))

        outputs = [
            CompletionOutput(
                index=index,
                text=text,
                token_ids=[],
                cumulative_logprob=None,
                logprobs=None,
            )
            for index, text in enumerate(texts[:n])
        ]
        return RequestOutput(
            request_id=str(time.time()),
            prompt=prompt,
            prompt_token_ids=[],
            prompt_logprobs=None,
            outputs=outputs,
            finished=True,
        )

    def _request(self, prompt: str, sampling_params: ApiSamplingParams, n: int) -> list[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(getattr(sampling_params, "temperature", 0.7)),
            "top_p": float(getattr(sampling_params, "top_p", 1.0)),
            "max_tokens": int(getattr(sampling_params, "max_tokens", 512)),
            "n": n,
        }
        stop = getattr(sampling_params, "stop", None)
        if stop:
            payload["stop"] = stop

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error = None
        for attempt in range(self.max_retries):
            request = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                parsed = json.loads(body)
                choices = parsed.get("choices") or []
                texts = []
                for choice in choices:
                    message = choice.get("message") if isinstance(choice, dict) else None
                    content = message.get("content") if isinstance(message, dict) else None
                    if content is None:
                        content = choice.get("text") if isinstance(choice, dict) else ""
                    texts.append(str(content or ""))
                if texts:
                    return texts
                raise RuntimeError(f"API response has no choices: {body[:500]}")
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"API HTTP {exc.code}: {error_body[:500]}")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
            if attempt + 1 < self.max_retries:
                time.sleep(self.retry_sleep * (attempt + 1))
        raise RuntimeError(f"OpenAI-compatible API request failed after {self.max_retries} attempts: {last_error}")
