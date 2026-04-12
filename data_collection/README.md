# Reproduce the Experiments

This document provides comprehensive instructions for reproducing the experiments presented in our paper.

> [!IMPORTANT]
> **General Requirements**
>
> We use VLLM to accelerate model inference. Note that VLLM may have environment conflicts with the training environment. We recommend using Anaconda to create a separate environment specifically for the data collection process.
>
> You can install the specific VLLM version used in this paper as follows:
>
> ```bash
> git clone https://github.com/MARIO-Math-Reasoning/vllm
> cd vllm
> pip install -e .
> ```
>
> Ensure you set `CUDA_VISIBLE_DEVICES` to specify the 1 or 2 GPUs you wish to use for the experiments.

## Reproducing the Data Collection Process

To generate the initial dataset using Monte Carlo Tree Search, execute the following command. This step creates synthetic samples based on our seed data and generates solution paths:

```bash
python solver_demo.py \
--custom_cfg configs/mcts_code.yaml \
--qaf ../data/dsc_collection/data_seed_test.json
```

## Code Review Data Generation

For the code-evaluation adaptation, use the dedicated review MCTS entrypoint. It currently consumes `CodeCriticBench` code-generation samples, expands one subtree per checklist dimension, and labels leaf rewards by comparing the model's structured review against dataset checklist scores plus test-derived correctness signals.

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=data_collection HF_HUB_OFFLINE=1 UV_CACHE_DIR=/tmp/uv-cache \
uv run python data_collection/solver_review.py \
  --custom_cfg data_collection/configs/mcts_code_review.yaml \
  --dataset datasets/CodeCriticBench/data/CodeCriticBench.jsonl \
  --start 0 \
  --limit 32 \
  --output data_collection/review_mcts_runs/codecriticbench_0_32/aggregate.jsonl \
  --output_dir data_collection/review_mcts_runs/codecriticbench_0_32/samples
```

The aggregate output is a `.jsonl` file containing the original review seed, objective scoring metadata, one `best_reviews_by_dimension` summary, and a pure-node `react` tree whose nodes carry `value/q_value/visit_count` fields for downstream preprocessing. If `--output_dir` is set, the same records are also written as one pretty JSON file per sample for inspection. Under each dimension subtree, MCTS explores native thinking nodes first; after the configured search iterations, a review-specific finalization pass selects one open leaf per dimension that still lacks a scored terminal and forces `n_generate_sample` terminal `<review>` JSON leaves. Terminal review rewards are stored in the leaf's `q_value` and `reward_details`.

Long MCTS collection should be run inside `tmux` so it keeps running after SSH disconnects:

```bash
tmux new -s review-mcts-0-32
```

Then run the `uv run python data_collection/solver_review.py ...` command inside that session. Detach with `Ctrl-b d`, list sessions with `tmux ls`, and reattach with:

```bash
tmux attach -t review-mcts-0-32
```

`solver_review.py` enables `--resume` by default. If the process is interrupted, rerun the same command and it will append to the existing aggregate `.jsonl`, skip completed `dataset_index` values, and repair missing aggregate rows from complete per-sample JSON files under `--output_dir`. Use `--no-resume` only when you intentionally want to overwrite the aggregate output. Resume granularity is per sample, so an interruption inside one sample's MCTS search restarts that sample; keep `batch_size: 1` for long unattended runs if minimizing lost in-progress work is more important than throughput.

For a quick smoke test, reduce the config at runtime by copying `data_collection/configs/mcts_code_review.yaml` and setting:

```yaml
iterations: 2
n_generate_sample: 1
max_review_dimensions: 2
max_depth: 1
batch_size: 1
max_tokens: 256
enforce_eager: True
```

After completing the data preprocessing steps outlined in `../model_training/README.md`, you will obtain `all_pass.jsonl` and `all_fail.jsonl` files that document samples lacking incorrect and correct paths respectively. You can then run the following commands for path refinement and perturbation:

```bash
python correction.py ../data/dsc_collection

python solver_demo.py \
--custom_cfg configs/correction.yaml \
--qaf ../data/dsc_collection/all_pass.json
```

## Reproducing the Inference Stage

Please first modify the `model_dir` parameter in `configs/inference.yaml` to point to your trained model before proceeding.

To evaluate the model's performance on the HumanEval benchmark:

```bash
python solver_demo.py \
--custom_cfg configs/inference.yaml \
--qaf ../data/code_gen_eval/humaneval.json
```

To evaluate the model's performance on the MBPP benchmark:

```bash
python solver_demo.py \
--custom_cfg configs/inference.yaml \
--qaf ../data/code_gen_eval/mbpp.json
```

To evaluate the model's performance on the LiveCodeBench benchmark:

```bash
python solver_demo.py \
--custom_cfg configs/inference.yaml \
--qaf ../data/code_gen_eval/full_problems.json
```

## Calculating Benchmark Pass Rates

For computing pass rates on HumanEval and MBPP benchmarks, we recommend using EvalPlus with the following command:

```bash
evalplus.evaluate --dataset $DATASET --samples $SAVE_PATH
```

Where:
- `$DATASET` should be set to either "humaneval" or "mbpp"
- `$SAVE_PATH` is the path to your generated solutions

You can find the generated predictions in the `../data/code_gen_eval` directory.

For calculating pass rates on LiveCodeBench, we recommend using the official [LiveCodeBench](https://github.com/livecodebench/livecodebench) repository and its evaluation pipeline.
