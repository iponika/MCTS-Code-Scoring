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

Use `--response_mode stepwise --reasoning_steps 3` when testing whether explicit `<step>` supervision helps.

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

- `scripts/run_bootstrap_comparison_qwen3_4b.sh`: current 4B static/direct/MCTS comparison.
- `scripts/run_qwen35_9b_direct_stepwise_vs_review_smoke.sh`: 9B direct-review vs direct-stepwise comparison, intended for larger-memory servers.
- `scripts/run_qwen35_9b_fsdp_smoke.sh`: isolated 9B training smoke.
- `scripts/run_direct_stepwise_vs_review_qwen3_4b.sh`: 4B diagnostic comparing direct final-review SFT and non-MCTS stepwise SFT.
- `scripts/run_axiom_clean_eval.sh`: AXIOM held-out evaluation.

All wrappers are resumable at file/stage level where practical and write logs under `data_collection/review_mcts_runs/<RUN_NAME>/logs`.
