# Reproduce the Experiments

This document provides comprehensive instructions for reproducing the experiments presented in our paper.

## Data Preprocessing

After MCTS data collection (see `../data_collection/README.md`), follow these steps to prepare the training data:

### Step 1: Merge and Filter MCTS Data

```bash
RAW_DATA_PATH=../../data/dsc_collection
PREPROCESS_OUTPUT_PATH=../../data/dsc_collection/data_mcts.jsonl

python magicoder/preprocess_mcts_data.py \
--dataset_path json \
--raw_dataset_path ${RAW_DATA_PATH} \
--output_file ${PREPROCESS_OUTPUT_PATH} \
--stage mcts \
--key src-instruct
```

### Step 2: Combine Path Refinement and Perturbation Data (First Stage)

```bash
RAW_DATA_PATH=../../data/dsc_collection
PREPROCESS_OUTPUT_PATH=../../data/dsc_collection/data_mcts_pr.jsonl

python magicoder/preprocess_mcts_data.py \
--dataset_path json \
--raw_dataset_path ${RAW_DATA_PATH} \
--output_file ${PREPROCESS_OUTPUT_PATH} \
--stage pr \
--key src-instruct
```

### Step 3: Create Training Data for Adaptive CoT Reasoning (Second Stage)

```bash
RAW_DATA_PATH=../../data/dsc_collection
PREPROCESS_OUTPUT_PATH=../../data/dsc_collection/data_mcts_s2.jsonl

python magicoder/preprocess_mcts_data.py \
--dataset_path json \
--raw_dataset_path ${RAW_DATA_PATH} \
--output_file ${PREPROCESS_OUTPUT_PATH} \
--stage s2 \
--key src-instruct
```

## Model Training

First, set `CUDA_VISIBLE_DEVICES` to specify which 1-2 GPUs to use.

### Code Review MCTS Data

The code-evaluation adaptation uses a dedicated converter because review rewards are continuous and terminal leaves are `<review>` JSON blocks rather than generated code.

```bash
python model_training/src/magicoder/preprocess_review_mcts_data.py \
  --input data_collection/review_mcts_runs/codecriticbench_2_1/samples/2_mbpp_mbpp.json \
  --output_file model_training/review_mcts_train_data/codecriticbench_2_1_train_multi.jsonl \
  --policy_min_q 0.5
```

Each output item contains `instruction`, segmented `response`, continuous `q_value`, and `train_lm`.
Items with `train_lm=true` train both the policy tokens and value head; items with `train_lm=false` mask LM labels and train only the value head.

Run review policy/value training from `model_training/src` to avoid the repository-level `datasets/` directory shadowing Hugging Face `datasets`:

Qwen3.5 requires a Transformers build that recognizes `model_type=qwen3_5`. If your installed release does not, install Transformers from the official main branch:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv pip install --upgrade --no-deps "git+https://github.com/huggingface/transformers.git"
UV_CACHE_DIR=/tmp/uv-cache uv pip install --upgrade "huggingface-hub>=1.5.0,<2.0"
```

When running offline, prefer the local Hugging Face snapshot path for `--model_name_or_path`; this avoids adapter-config probes against the Hub.

```bash
cd model_training/src
CUDA_VISIBLE_DEVICES=0 HF_DATASETS_CACHE=/tmp/hf-datasets-cache uv run accelerate launch --num_processes=1 -m magicoder.train_multi \
  --task review \
  --model_key Qwen/Qwen3.5-9B \
  --model_name_or_path /path/to/models--Qwen--Qwen3.5-9B/snapshots/<snapshot-id> \
  --datafile_paths ../review_mcts_train_data/codecriticbench_2_1_train_multi.jsonl \
  --output_dir ./output/qwen3_5_9b-review-s1 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --bf16 True \
  --logging_steps 1 \
  --optim adafactor \
  --learning_rate 5e-5 \
  --lr_scheduler_type linear \
  --peft lora \
  --value_weight 0.025 \
  --num_proc 1
```

For a one-step smoke test, add `--max_steps 1 --save_strategy no --skip_save True --report_to none`.
For a small real checkpoint, remove `--skip_save`, keep `--save_strategy no`, and use a small `--max_steps` such as `20`. LoRA runs now save the adapter plus `value_head.pth` by default; add `--save_merged_model True` only when you explicitly need a merged full-backbone checkpoint.
If the installed Transformers version does not recognize `model_type=qwen3_5`, upgrade Transformers to a build that supports Qwen3.5 before running the real target model.

After training a policy/value checkpoint, inspect value scoring on an existing path:

```bash
cd model_training/src
HF_HUB_OFFLINE=1 uv run python -m magicoder.review_policy_value_inference \
  --policy_model_path ./output/qwen3_5_9b-review-s1 \
  --value_model_path ./output/qwen3_5_9b-review-s1 \
  --datafile_path ../review_mcts_train_data/codecriticbench_2_1_train_multi.jsonl \
  --item_index 7 \
  --use_gold_response
```

Run a minimal value-guided review loop, where the value model scores policy candidates at each step and the highest-value continuation is selected:

```bash
cd model_training/src
HF_HUB_OFFLINE=1 uv run python -m magicoder.review_value_guided_evaluator \
  --policy_model_path ./output/qwen3_5_9b-review-s1 \
  --value_model_path ./output/qwen3_5_9b-review-s1 \
  --datafile_path ../review_mcts_train_data/codecriticbench_2_1_train_multi.jsonl \
  --item_index 7 \
  --max_steps 3 \
  --num_candidates 4 \
  --max_new_tokens 256 \
  --final_max_new_tokens 512 \
  --temperature 0.7 \
  --top_p 0.95
```

For the structured code-review evaluation flow, use `review_evaluator`. It loops over review dimensions, samples policy candidates, selects continuations with the value model, triggers rethink when the selected value is below `--rethink_threshold`, and writes a full trace:

```bash
cd model_training/src
HF_HUB_OFFLINE=1 uv run python -m magicoder.review_evaluator \
  --policy_model_path ./output/qwen3_5_9b-review-s1 \
  --value_model_path ./output/qwen3_5_9b-review-s1 \
  --input_record ../../data_collection/review_mcts_runs/codecriticbench_2_1/samples/2_mbpp_mbpp.json \
  --output_file ./output/review-eval/2_mbpp.json \
  --max_dimensions 10 \
  --max_steps 3 \
  --num_candidates 4 \
  --max_new_tokens 256 \
  --temperature 0.7 \
  --top_p 0.95 \
  --rethink_threshold -0.2 \
  --max_rethinks 1 \
  --max_final_retries 1
```

If `--policy_model_path` and `--value_model_path` are the same path, `review_evaluator` loads one value-head model and uses its `pretrained_model` for policy generation to avoid loading two 9B backbones. Local relative checkpoint paths are resolved before model loading so TRL/PEFT does not misinterpret them as Hub repo ids. Each dimension output includes `reference_score`, `parsed_score`, and `score_delta` when dataset scores are available; the top-level `evaluation_summary` reports valid review rate and mean absolute score delta. A run that ends without a valid `<review>...</review>` is marked in `final_review_parse`; increase `--final_max_new_tokens`, `--max_new_tokens`, or `--max_final_retries` if this happens with a trained checkpoint.

### First Stage: Dual Model Training

```bash
MODEL_KEY=deepseek-ai/deepseek-coder-6.7b-instruct
MODEL_PATH=deepseek-ai/deepseek-coder-6.7b-instruct
MAGICODER_OUTPUT_DIR=./output/deepseek-coder-s1
PATH_TO_OSS_INSTRUCT=../../data/dsc_collection/data_mcts_pr.jsonl

accelerate launch --num_processes=1 -m magicoder.train_multi \
--model_key $MODEL_KEY \
--model_name_or_path $MODEL_PATH \
--use_flash_attention True \
--max_training_seq_length 2048 \
--datafile_paths ${PATH_TO_OSS_INSTRUCT} \
--output_dir $MAGICODER_OUTPUT_DIR \
--num_train_epochs 2 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 128 \
--group_by_length False \
--bf16 True \
--ddp_find_unused_parameters False \
--logging_steps 1 \
--log_level info \
--optim adafactor \
--warmup_steps 15 \
--learning_rate 5e-5 \
--lr_scheduler_type linear \
--peft lora \
--value_weight 0.025
```

### Second Stage: Adaptive CoT Reasoning Training

```bash
MODEL_KEY=deepseek-ai/deepseek-coder-6.7b-instruct
MODEL_PATH=deepseek-ai/deepseek-coder-6.7b-instruct
MAGICODER_OUTPUT_DIR=./output/deepseek-coder-s2 # Note: Changed output directory
PATH_TO_OSS_INSTRUCT=../../data/dsc_collection/data_mcts_s2.jsonl

accelerate launch --num_processes=1 -m magicoder.train_multikl \
--model_key $MODEL_KEY \
--model_name_or_path $MODEL_PATH \
--use_flash_attention True \
--max_training_seq_length 2048 \
--datafile_paths ${PATH_TO_OSS_INSTRUCT} \
--output_dir $MAGICODER_OUTPUT_DIR \
--num_train_epochs 2 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 128 \
--group_by_length False \
--bf16 True \
--ddp_find_unused_parameters False \
--logging_steps 1 \
--log_level info \
--optim adafactor \
--warmup_steps 15 \
--learning_rate 5e-5 \
--lr_scheduler_type linear \
--peft lora \
--value_weight 0.025
```

## Model Inference

Update the `MODEL` variable to your saved model path. The code below runs inference for the baseline model. To use self-planning or code CoT templates, modify the prompt_template in `../experiments/text2code.py`. For SPEAR inference, see `../data_collection/README.md`.

```bash
MODEL_KEY=deepseek-ai/deepseek-coder-6.7b-instruct
MODEL=./output/deepseek-coder-s1 # Change to your model path

DATASET=lcb # Options: humaneval, mbpp, lcb
SAVE_PATH=$MODEL/evalplus-$DATASET.jsonl

python -m experiments.text2code \
--model_key $MODEL_KEY \
--model_name_or_path $MODEL \
--save_path $SAVE_PATH \
--dataset $DATASET \
--temperature 0 \
--top_p 1.0 \
--max_new_tokens 2048 \
--n_problems_per_batch 5 \
--n_samples_per_problem 1 \
--n_batches 1 \
--prompt direct # Options: direct, self-planning, code-cot
```

## Calculating Benchmark Pass Rates

### For HumanEval and MBPP

Use EvalPlus with the following command:

```bash
evalplus.evaluate --dataset $DATASET --samples $SAVE_PATH
```

Where:
- `$DATASET` is either "humaneval" or "mbpp"
- `$SAVE_PATH` is the path to your generated solutions

Generated predictions can be found in the `../data/code_gen_eval` directory.

### For LiveCodeBench

Use the official [LiveCodeBench repository](https://github.com/livecodebench/LiveCodeBench) and follow its evaluation pipeline to calculate pass rates.
