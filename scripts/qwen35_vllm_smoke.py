from vllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(
        model="Qwen/Qwen3.5-9B",
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.55,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    outputs = llm.generate(
        ["用一句话介绍你自己。"],
        SamplingParams(max_tokens=32, temperature=0.0),
    )
    print("SMOKE_OUTPUT", outputs[0].outputs[0].text.replace("\n", " "))


if __name__ == "__main__":
    main()
