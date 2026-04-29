# Review Policy/Value Training

Run commands in this file from `model_training/src` unless stated otherwise. This avoids the repository-level `datasets/` directory shadowing Hugging Face `datasets`.

## Convert Review Trajectories

Convert direct or MCTS trajectory JSONL files into `train_multi` JSONL:

```bash
cd model_training/src
PYTHONPATH=.:../../data_collection uv run python -m magicoder.preprocess_review_mcts_data \
  --input ../../data_collection/review_mcts_runs/example/mcts_bootstrap_raw.jsonl \
  --output_file ../review_mcts_train_data/example_mcts_train.jsonl \
  --policy_min_q 0.5 \
  --policy_response_mode path
```

Each output row contains:

- `instruction`: task and candidate-code scoring prompt.
- `response`: one or more `<step>` / `<review>` segments.
- `q_value`: value target in the training range.
- `train_lm`: whether the response tokens also train the policy head.
- optional interval fields such as `q_min` / `q_max` for weak labels.

Use `--policy_response_mode final_review` when exporting only the best final `<review>` for final-only policy training.

## Static AXIOM-Style Score Data

Use AXIOM and CodeCriticBench as the stable first-stage score sources:

```bash
cd model_training/src
uv run python -m magicoder.preprocess_score_datasets \
  --axiom_dir ../../datasets/axiom-llm-judge \
  --codecriticbench ../../datasets/CodeCriticBench/data/CodeCriticBench.jsonl \
  --drop_axiom_grade_zero \
  --output_file ../review_mcts_train_data/axiom_codecritic_static.jsonl \
  --shuffle_seed 42
```

The internal target is AXIOM grade 0-5, treated as an ordinal repair-effort score. User-facing 0-100 scores are derived as `grade / 5 * 100`; they are not the primary label.

## Train LoRA Policy/Value Model

Example single-node command:

```bash
cd model_training/src
CUDA_VISIBLE_DEVICES=0,1 \
uv run accelerate launch --num_processes=2 -m magicoder.train_multi \
  --task review \
  --model_key Qwen/Qwen3.5-9B \
  --model_name_or_path Qwen/Qwen3.5-9B \
  --datafile_paths ../review_mcts_train_data/example_mcts_train.jsonl \
  --output_dir ./output/qwen35-review-example \
  --max_training_seq_length 2048 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --bf16 True \
  --logging_steps 1 \
  --optim adafactor \
  --learning_rate 5e-5 \
  --lr_scheduler_type linear \
  --peft lora \
  --lora_target_scope attention \
  --lora_rank 8 \
  --value_weight 0.025 \
  --save_strategy no \
  --report_to none
```

For a smoke test, add `--max_steps 1 --skip_save True`. For a real checkpoint, remove `--skip_save` and set a finite `--max_steps`.

LoRA runs save the adapter plus `value_head.pth`. Add `--save_merged_model True` only when a merged full-backbone checkpoint is explicitly needed.

## Evaluate

Value-head inspection on a known training row:

```bash
cd model_training/src
uv run python -m magicoder.review_policy_value_inference \
  --policy_model_path ./output/qwen35-review-example \
  --value_model_path ./output/qwen35-review-example \
  --datafile_path ../review_mcts_train_data/example_mcts_train.jsonl \
  --item_index 0 \
  --use_gold_response
```

Structured held-out evaluation:

```bash
cd model_training/src
uv run python -m magicoder.review_evaluator \
  --policy_model_path ./output/qwen35-review-example \
  --value_model_path ./output/qwen35-review-example \
  --share_policy_value_model \
  --input_record ../../datasets/axiom-llm-judge/axiombench/apps.jsonl \
  --output_file ./output/review-eval-example/apps_0.json \
  --record_index 0 \
  --max_steps 3 \
  --num_candidates 2 \
  --max_new_tokens 192 \
  --final_max_new_tokens 256 \
  --score_key last_value
```

Use `--final_only_json` for final-only scoring. Omit it for stepwise value-guided evaluation.
