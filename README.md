# MCTS Code Scoring

[中文 README](README.zh-CN.md)

This repository is a code-evaluation adaptation of the SEER MCTS pipeline. The current target is not code generation: it trains and evaluates models that assign stable AXIOM-style scores to candidate code. Text comments are auxiliary evidence for the score.

## Active Scope

- Generate CodeCriticBench-based review trajectories with direct bootstrap or review-MCTS.
- Convert trajectories into policy/value training data.
- Train a policy/value model with LoRA and a value head.
- Evaluate on AXIOM-style held-out code-scoring tasks.

The original code-generation datasets, Open-R1 reproduction code, old perturbation training, and historical ablation wrappers have been removed from the tracked repository. Large datasets, model checkpoints, and run outputs are intentionally ignored.

## Repository Layout

```text
data_collection/
  solver_review.py                         # review-MCTS data generation
  direct_bootstrap_review.py               # non-MCTS direct bootstrap data
  prepare_codecritic_axiom_seedset.py      # CodeCriticBench -> AXIOM-aligned seeds
  prepare_static_review_train_data.py      # static seed -> exact value labels
  rebalance_review_train_data.py           # balance policy/value samples
  configs/                                 # review-MCTS configs
  scripts/                                 # maintained experiment wrappers

model_training/src/magicoder/
  preprocess_review_mcts_data.py           # MCTS/direct trajectories -> train JSONL
  preprocess_score_datasets.py             # AXIOM/CodeCritic static score data
  train_multi.py                           # policy/value LoRA training
  review_evaluator.py                      # final-only and stepwise eval
  review_policy_value_inference.py         # value-head inspection
  review_value_guided_evaluator.py         # value-guided single-sample loop

tests/                                     # regression tests for review path
tools/mcts_tree_viewer.html                # local MCTS sample viewer
docs/                                      # current design notes and reports
```

## Data Expected After Clone

Datasets are not tracked. Put them under these paths, or override the paths in commands:

```text
datasets/CodeCriticBench/data/CodeCriticBench.jsonl
datasets/axiom-llm-judge/axiombench/*.jsonl
```

Optional future datasets should also stay outside Git under `datasets/` or `benchmarks/`.

## Environment

Use `uv` in the Python environment provided by the target server. On a cluster, install the CUDA/PyTorch/vLLM stack according to the local driver first, then install project dependencies:

```bash
uv pip install -r requirements.txt
```

For Qwen3.5, the environment must include a Transformers build that recognizes `model_type=qwen3_5`. The default `requirements.txt` uses the Hugging Face main branch for this reason. If the server already has a compatible Transformers version, replacing that line with a pinned release is fine.

## Smoke Checks

Run these from the repository root:

```bash
PYTHONPATH=data_collection:model_training/src uv run pytest tests
```

Prepare a tiny CodeCritic seed set:

```bash
PYTHONPATH=data_collection uv run python data_collection/prepare_codecritic_axiom_seedset.py \
  --output /tmp/codecritic_seed.jsonl \
  --metadata /tmp/codecritic_seed.meta.json \
  --per_grade 1 \
  --min_grade 1 \
  --max_grade 5
```

## Maintained Workflows

4B full comparison wrapper:

```bash
tmux new -s review_4b_cmp
RUN_NAME=bootstrap_cmp_qwen3_4b_server \
SEED_PER_GRADE=8 \
MAX_STEPS=120 \
bash data_collection/scripts/run_bootstrap_comparison_qwen3_4b.sh
```

Qwen3.5-9B direct-review vs direct-stepwise wrapper for larger-memory servers:

```bash
tmux new -s qwen35_9b_stepwise
RUN_NAME=qwen35_9b_direct_stepwise_server \
SEED_PER_GRADE=8 \
DIRECT_REPEATS=3 \
MAX_STEPS=120 \
MAX_TRAINING_SEQ_LENGTH=2048 \
bash data_collection/scripts/run_qwen35_9b_direct_stepwise_vs_review_smoke.sh
```

AXIOM held-out evaluation for an existing checkpoint:

```bash
RUN_NAME=axiom_eval_server \
TRAINED_MODEL_PATH=/path/to/review-lora-checkpoint \
bash data_collection/scripts/run_axiom_clean_eval.sh
```

All maintained wrappers derive `ROOT` from their own location by default, so they should work after `git clone` without editing absolute paths. Set `ROOT=...` only if you intentionally run scripts from a different checkout.

## Long Jobs

Use `tmux` for long-running data generation and training:

```bash
tmux new -s mcts_job
# run the command
# detach: Ctrl-b d
tmux attach -t mcts_job
```

Experiment outputs are written under ignored directories:

```text
data_collection/review_mcts_runs/
model_training/review_mcts_train_data/
model_training/src/output/
```
