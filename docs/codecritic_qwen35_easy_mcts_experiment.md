# CodeCriticBench Easy Correctness MCTS Experiment

Date: 2026-05-13

## Goal

Run a small-model multi-level code-scoring experiment on CodeCriticBench using only Code Generation Easy samples. The task is single-dimension `Correctness Verification` scoring on the original CodeCriticBench 1-10 scale. Code QA samples are excluded because they are answer-quality QA tasks rather than direct candidate-code scoring tasks.

## Data Scope

Use `datasets/CodeCriticBench/data/CodeCriticBench.jsonl`.

Included samples:

- `source in {mbpp, codeforce, live-code-bench, debug}`
- `difficulty == "Easy"`
- sample has a `Correctness Verification` checklist, checklist score, and checklist text

Excluded samples:

- `source == stackoverflow`
- non-Easy samples
- Easy CodeGen rows missing `Correctness Verification` advanced labels

Observed local counts:

| Split basis | Count |
|---|---:|
| CodeGen Easy total | 1164 |
| Missing Correctness Verification advanced label | 57 |
| Eligible CodeGen Easy | 1107 |

Layering uses the original CodeCriticBench overall `score`, not the per-dimension correctness score:

| Layer | Overall score range | Original Easy count | Eligible count |
|---|---:|---:|---:|
| Low | 0-3 | 809 | 753 |
| Mid | 4-6 | 17 | 16 |
| High | 7-10 | 338 | 338 |

## Train/Eval Split

Create a fixed MCTS seed split with:

```bash
PYTHONPATH=data_collection python data_collection/prepare_codecritic_easy_correctness_splits.py \
  --input datasets/CodeCriticBench/data/CodeCriticBench.jsonl \
  --train_output data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/seed_train.jsonl \
  --eval_output data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/seed_eval.jsonl \
  --metadata data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/split_metadata.json \
  --target_train 500 \
  --seed 20260513
```

Training seed allocation:

- Low/high counts use `floor(500 * original_layer_count / 1164)`.
- Mid is scarce, so all eligible Mid samples are retained.
- Expected train seeds: Low 347, Mid 16, High 145, total 508.
- Remaining eligible Easy samples form the evaluation set, expected total 599.

## MCTS Generation

Model: `Qwen/Qwen3.5-9B` in no-think mode.

Config:

```text
data_collection/configs/mcts_codecritic_qwen35_9b_no_think.yaml
```

Important prompt/scoring choices:

- `review_prompt_variant: codecritic_correctness`
- `show_tests_in_prompt: True`
- output schema uses `correctness_score`, not `axiom_grade`
- only `Correctness Verification` is scored
- final score range is integer 1-10

Run MCTS over the 508 training seeds. If all four GPUs are idle, split the seed file into four contiguous ranges and run one tmux worker per GPU. Each worker writes one JSONL shard and one per-sample output directory.

## Policy Selection Rule

After MCTS generation, inspect terminal review nodes. A seed contributes one policy sample if at least one valid terminal review satisfies:

```text
abs(predicted_correctness_score - target_correctness_score) <= 2
```

For each such seed, choose the best terminal path by:

1. highest `q_value`
2. smaller absolute score distance as a tie-breaker
3. valid parseable final review with no reward error

If no terminal review is within the ±2 range, keep the explored paths as value-only candidates but do not train LM policy on that seed.

## Training Plan After Policy Count

Let `P` be the number of selected MCTS policy samples.

- Train `MCTS` on selected policy paths plus value-only paths from the generated trees.
- Train `Static` with exactly `P` seed-label examples generated from the same training seed pool.
- Do not train `Direct` in this round.
- Evaluate on the held-out eligible Easy set.

The first required result before training is the policy availability count `P`.
