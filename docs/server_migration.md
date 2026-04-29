# Server Migration Notes

This checklist is for moving the project to a different GPU server.

## Clone

```bash
git clone <repo-url> mcts-code-review
cd mcts-code-review
```

The clone intentionally does not include datasets, model checkpoints, run outputs, or local reference repositories.

## Environment

Use the server's CUDA module or conda environment first. Then install project Python dependencies with uv:

```bash
uv pip install -r requirements.txt
```

If the server already provides PyTorch/vLLM, keep those versions and install only missing packages:

```bash
uv pip install -r requirements.txt --no-deps
```

For Qwen3.5, verify that Transformers can load `Qwen/Qwen3.5-9B`. If not, install the Hugging Face main branch as shown in `requirements.txt`.

## Data Placement

Expected default paths:

```text
datasets/CodeCriticBench/data/CodeCriticBench.jsonl
datasets/axiom-llm-judge/axiombench/*.jsonl
```

These directories are ignored by Git. Copy or symlink datasets into place on each server.

## Model Cache

Set `HF_HOME` if the cluster uses a shared model cache:

```bash
export HF_HOME=/path/to/shared/huggingface
```

All maintained scripts accept `MODEL_PATH=...` and `MODEL_KEY=...` overrides.

## Quick Validation

```bash
PYTHONPATH=data_collection:model_training/src uv run pytest tests
```

Run one seed-preparation smoke:

```bash
PYTHONPATH=data_collection uv run python data_collection/prepare_codecritic_axiom_seedset.py \
  --output /tmp/codecritic_seed.jsonl \
  --metadata /tmp/codecritic_seed.meta.json \
  --per_grade 1 \
  --min_grade 1 \
  --max_grade 5
```

## Long Jobs

Run long jobs inside `tmux`:

```bash
tmux new -s qwen35_9b_review
RUN_NAME=qwen35_9b_server_test bash data_collection/scripts/run_qwen35_9b_direct_stepwise_vs_review_smoke.sh
```

Detach with `Ctrl-b d`, reattach with:

```bash
tmux attach -t qwen35_9b_review
```

Outputs are written under ignored paths:

```text
data_collection/review_mcts_runs/
model_training/review_mcts_train_data/
model_training/src/output/
```
