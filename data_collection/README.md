# Review Data Collection

This directory contains the active data-generation path for code scoring. The legacy code-generation reproduction entrypoints were removed; use the review-specific scripts below.

## Seed Preparation

Create AXIOM-aligned seeds from CodeCriticBench:

```bash
PYTHONPATH=data_collection uv run python data_collection/prepare_codecritic_axiom_seedset.py \
  --output data_collection/review_mcts_runs/example/seed_codecritic_axiom.jsonl \
  --metadata data_collection/review_mcts_runs/example/seed_codecritic_axiom.metadata.json \
  --per_grade 4 \
  --min_grade 1 \
  --max_grade 5
```

Grade 0 is excluded by default in recent experiments because those samples are often outliers and can over-teach extreme failure behavior.

## Direct Bootstrap

Generate non-MCTS reviews or non-MCTS sequential steps:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=data_collection \
uv run python data_collection/direct_bootstrap_review.py \
  --custom_cfg data_collection/configs/mcts_code_review.yaml \
  --dataset data_collection/review_mcts_runs/example/seed_codecritic_axiom.jsonl \
  --output data_collection/review_mcts_runs/example/direct_bootstrap_raw.jsonl \
  --dimension "Correctness Verification" \
  --response_mode review \
  --repeats 2
```

Use `--response_mode stepwise --reasoning_steps -1` for MCTS-like direct stepwise generation: it generates `max_depth` reasoning steps before the final review, matching the usual MCTS final-review path length. Pass an explicit non-negative `--reasoning_steps` only for fixed-step ablations.

## Review MCTS

Generate review-MCTS trajectories:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=data_collection \
uv run python data_collection/solver_review.py \
  --custom_cfg data_collection/configs/mcts_code_review.yaml \
  --dataset data_collection/review_mcts_runs/example/seed_codecritic_axiom.jsonl \
  --output data_collection/review_mcts_runs/example/mcts_bootstrap_raw.jsonl \
  --output_dir data_collection/review_mcts_runs/example/mcts_samples
```

`solver_review.py` resumes by default. Re-running the same command skips completed `dataset_index` values in the aggregate output and repairs missing aggregate rows from complete per-sample JSON files under `--output_dir`.

## Maintained Wrappers

DeepSeek-R1-Distill-Qwen-7B is the default model for current experiments. The
Qwen-specific wrappers remain for reproducing old runs, but new runs should use
the DeepSeek wrappers unless a comparison explicitly requires another model.

- `scripts/run_bootstrap_comparison_deepseek7b.sh`: default static/direct/MCTS comparison.
- `scripts/run_direct_stepcount_vs_review_deepseek7b.sh`: default direct-review vs 1-step/2-step comparison.
- `scripts/run_direct_stepwise_vs_review_deepseek7b.sh`: default diagnostic comparing final-review SFT and stepwise SFT.
- `scripts/run_bootstrap_comparison_qwen3_4b.sh`: legacy Qwen3-4B static/direct/MCTS comparison.
- `scripts/run_qwen35_9b_direct_stepwise_vs_review_smoke.sh`: 9B direct-review vs direct-stepwise comparison, intended for larger-memory servers.
- `scripts/run_qwen35_9b_fsdp_smoke.sh`: isolated 9B training smoke.
- `scripts/run_direct_stepwise_vs_review_qwen3_4b.sh`: legacy 4B diagnostic comparing direct final-review SFT and non-MCTS stepwise SFT.
- `scripts/run_axiom_clean_eval.sh`: AXIOM held-out evaluation.
- `configs/mcts_code_review_qwen3_4b_thinking.yaml`: Qwen3-4B native `/think` review-MCTS smoke configuration.
- `configs/mcts_code_review_qwen3_4b_no_think.yaml`: Qwen3-4B `/no_think` comparison configuration.
- `configs/mcts_code_review_deepseek_r1_distill_qwen_7b.yaml`: DeepSeek-R1-Distill-Qwen-7B reasoning review-MCTS configuration.

All wrappers are resumable at file/stage level where practical and write logs under `data_collection/review_mcts_runs/<RUN_NAME>/logs`.
