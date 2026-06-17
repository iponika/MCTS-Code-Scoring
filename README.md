# SPARC: MCTS-Guided Self-Training for Code Scoring

[中文 README](README.zh-CN.md)

This repository contains the implementation artifacts for **SPARC**, a
policy-value self-training framework for criterion-based code scoring. SPARC
uses Monte-Carlo Tree Search (MCTS) to construct reward-filtered reasoning
trajectories, trains a LoRA-adapted policy head and value head, and performs
value-guided inference for stable code-correctness scoring.

The repository is organized as a research artifact for paper review. Large
datasets, generated search trees, training JSONL files, and model checkpoints
are intentionally excluded from Git and should be restored separately when
reproducing experiments.

## Repository Layout

```text
data_collection/
  solver_review.py                         # MCTS trajectory generation
  direct_bootstrap_review.py               # direct reasoning baseline data
  prepare_codecritic_axiom_seedset.py      # CodeCriticBench seed preparation
  prepare_static_review_train_data.py      # exact-label baseline preparation
  rebalance_review_train_data.py           # policy/value sample balancing
  configs/                                 # model and MCTS configuration files
  scripts/                                 # experiment wrappers

model_training/src/magicoder/
  preprocess_review_mcts_data.py           # tree/trajectory export to train JSONL
  preprocess_score_datasets.py             # static score-data preprocessing
  train_multi.py                           # LoRA policy/value training
  review_evaluator.py                      # code-scoring evaluator
  review_policy_value_inference.py         # value-head inspection utilities
  review_value_guided_evaluator.py         # single-sample value-guided inference

paper/
  69fdb1f6e7574fa1c601d7fa/JASE/          # manuscript source and figures

tests/                                     # regression tests for the review path
tools/mcts_tree_viewer.html                # local MCTS-tree inspection UI
docs/                                      # reviewer-facing notes and case evidence
```

## Data

Datasets are not tracked in this repository. By default, scripts expect the
following local paths:

```text
datasets/CodeCriticBench/data/CodeCriticBench.jsonl
datasets/axiom-llm-judge/axiombench/*.jsonl
```

The main paper experiments use the CodeGen subset of CodeCriticBench after
preprocessing into a 2,631-record seed set, partitioned into 1,316 training
records and 1,315 held-out evaluation records. The split IDs used by the current
artifact are recorded in:

```text
docs/codecriticbench_1316_1315_split_id_record_20260617.md
```

## Environment

Use the Python/CUDA environment provided by the target server. After installing
a compatible PyTorch and CUDA stack, install project dependencies with `uv`:

```bash
uv pip install -r requirements.txt
```

Qwen3.5-family checkpoints require a Transformers build that recognizes
`model_type=qwen3_5`. If the target server already provides a compatible
Transformers installation, using that pinned environment is acceptable.

## Quick Checks

Run regression tests from the repository root:

```bash
PYTHONPATH=data_collection:model_training/src uv run pytest tests
```

Prepare a small CodeCriticBench seed sample:

```bash
PYTHONPATH=data_collection uv run python data_collection/prepare_codecritic_axiom_seedset.py \
  --output /tmp/codecritic_seed.jsonl \
  --metadata /tmp/codecritic_seed.meta.json \
  --per_grade 1 \
  --min_grade 1 \
  --max_grade 5
```

Run a one-record MCTS trajectory smoke test after restoring the required model
checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=data_collection \
uv run python data_collection/solver_review.py \
  --custom_cfg data_collection/configs/mcts_code_review.yaml \
  --dataset datasets/CodeCriticBench/data/CodeCriticBench.jsonl \
  --start 0 \
  --limit 1 \
  --output data_collection/review_mcts_runs/smoke/aggregate.jsonl \
  --output_dir data_collection/review_mcts_runs/smoke/samples
```

## Reproduction Workflow

The paper workflow consists of four stages:

1. Prepare CodeCriticBench scoring seeds.
2. Generate reward-filtered MCTS reasoning trajectories.
3. Export policy/value training data and train the LoRA policy-value model.
4. Evaluate Base, Direct, ablation variants, and SPARC on the held-out split.

Long-running jobs should be launched in `tmux`:

```bash
tmux new -s sparc_job
# run the experiment command
# detach with Ctrl-b d
tmux attach -t sparc_job
```

Generated artifacts are written under ignored directories:

```text
data_collection/review_mcts_runs/
model_training/review_mcts_train_data/
model_training/src/output/
```
