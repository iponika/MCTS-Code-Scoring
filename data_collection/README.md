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
