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
