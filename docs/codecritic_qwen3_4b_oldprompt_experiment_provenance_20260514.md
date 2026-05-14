# Qwen3-4B CodeCritic Old-Prompt Optimistic Experiment Provenance

Date: 2026-05-14

This note records the provenance of the old-prompt Qwen3-4B CodeCriticBench Easy correctness experiment whose stopped-eval overlap was 490 valid examples. It is intended as the source-of-truth trail for reporting the experiment in a paper or appendix.

## Code State

The closest clean commit for this experiment is:

```text
ba27cf0215a997cf48c003ae822fbb8b73e28911
2026-05-14 10:49:54 +0800
feat: add qwen3 4b codecritic optimistic experiment
```

The MCTS generation and training started before this commit was created, while the repository was a dirty worktree based on:

```text
cf36605f291a62edf9d3b38212f3e8cd5aeb2520
2026-05-13 23:45:29 +0800
add docs
```

The dirty worktree changes used for the old experiment were then committed as `ba27cf0`. The later promptfix series should be treated as a different code path:

```text
6d06ae6 fix codecritic mcts score-only steps
3257df6 fix review mcts no-value expansion
212dd5f add resume control to codecritic mcts worker
e27ee91 handle missing value outputs in review mcts
5607fbc avoid stop-suffix empty review steps
dac5d0b strip score-only lines from review steps
dcc182d relax codecritic optimistic seed selection
d0d0456 borrow no-policy negatives for value sampling
```

For old-prompt reruns, use an isolated worktree checked out at `ba27cf0`. In the current server session this was:

```text
/tmp/cc-oldprompt-ba27cf0
```

## Source Dataset And Initial Filtering

Input dataset:

```text
datasets/CodeCriticBench/data/CodeCriticBench.jsonl
```

Seed file:

```text
data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514/seed_all_nonqa_easy.jsonl
```

Seed construction metadata:

```text
data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514/all_seed_metadata.json
```

Initial filter:

- CodeGen sources only: `mbpp`, `codeforce`, `live-code-bench`, `debug`.
- Difficulty: `Easy`.
- Excluded QA-style rows.
- Kept rows with a usable first raw `Correctness Verification` score.
- Removed label/score conflicts:
  - `correctness_label == Error` with target score greater than 5.
  - `correctness_label == Correct` with target score less than 5.

Initial counts:

| Item | Count |
|---|---:|
| all Easy layer rows before eligibility filtering | 1164 |
| skipped: missing Correctness Verification | 57 |
| skipped: label/score conflict | 51 |
| eligible non-QA CodeGen Easy seeds | 1056 |

Eligible layer counts:

| Layer | Count |
|---|---:|
| high | 334 |
| mid | 14 |
| low | 708 |

## MCTS Generation

Run directory:

```text
data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514
```

Model:

```text
Qwen/Qwen3-4B
```

Mode:

- Qwen no-think mode.
- CodeCritic correctness prompt.
- Visible tests enabled.
- Two RTX 4090 GPUs, one shard per GPU.

MCTS output shards:

| Shard | Records |
|---|---:|
| `mcts_gpu0_000_528.jsonl` | 528 |
| `mcts_gpu1_528_528.jsonl` | 528 |
| total | 1056 |

Generation time from logs:

```text
2026-05-14 00:57-01:18 +0800
```

## Optimistic Seed Selection

Optimistic selection artifacts:

```text
data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514/mcts_optimistic_over_gt_under_records.jsonl
data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514/seed_optimistic_over_gt_under.jsonl
data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514/optimistic_seed_indices.json
data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514/optimistic_selection_stats.json
```

Selection rule:

```text
over_count  = number of terminal leaves with predicted_correctness_score > target_correctness_score
under_count = number of terminal leaves with predicted_correctness_score < target_correctness_score
exact_count = number of terminal leaves with predicted_correctness_score == target_correctness_score

keep seed iff over_count > under_count
```

Selection counts:

| Item | Count |
|---|---:|
| MCTS records considered | 1056 |
| optimistic records kept | 197 |
| pessimistic records | 705 |
| tied records | 154 |
| optimistic seed rows | 197 |

The full selected seed index list is stored in `optimistic_seed_indices.json`. The previous experiment note `docs/codecritic_qwen3_4b_optimistic_mcts_experiment_20260514.md` also contains a human-readable table of selected indices.

## Train/Eval Split

Training seeds:

```text
data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514/seed_optimistic_over_gt_under.jsonl
```

Evaluation seeds:

```text
data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514/seed_eval_nonoptimistic.jsonl
data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514/eval_nonoptimistic_indices.json
data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514/eval_nonoptimistic_metadata.json
```

Split counts:

| Split | Count |
|---|---:|
| all eligible seeds | 1056 |
| optimistic train seeds | 197 |
| non-optimistic eval seeds | 859 |

Eval seed distribution:

| Field | Counts |
|---|---|
| source | `mbpp=194`, `codeforce=240`, `live-code-bench=223`, `debug=202` |
| score layer | `high=269`, `mid=8`, `low=582` |
| target score | `1=13`, `2=179`, `3=219`, `4=152`, `5=23`, `6=3`, `7=20`, `8=43`, `9=149`, `10=58` |

## Training Data

MCTS training data artifacts:

```text
data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514/mcts_optimistic_policy_pm2_value_pm2_aligned_train.jsonl
data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514/mcts_optimistic_policy_pm2_value_pm2_aligned_train_stats.json
model_training/review_mcts_train_data/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514_mcts_optimistic_policy_pm2_value_pm2_aligned_train.jsonl
```

Current file checksum:

```text
25407ea3d14d111c78b788f418aab760a11da0fcb6f4d0faf1871e9b31d50877
```

Important historical note: the original training log reports `629` MCTS training rows loaded and used. The current JSONL file at the same path has `624` rows and was overwritten by a later postprocess rerun. Therefore, for the model actually evaluated in the 490-overlap old result, the training log is the reliable source for the loaded item count.

MCTS training log evidence:

```text
Generating train split: 629 examples
Dataset size after filtering: 629
train_runtime: 414.4
train_loss: 0.9916
epoch: 2.025
```

Static training data:

```text
model_training/review_mcts_train_data/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514_static_optimistic197.jsonl
sha256: 05cb877b8120a1703f960fc7d023cc3d6ef243abfb3b8bad520aac33d5f44fda
```

Static training log evidence:

```text
Generating train split: 197 examples
Dataset size after filtering: 197
train_runtime: 391.6
train_loss: 3.009
epoch: 6.406
```

Training configuration:

- `max_steps=160`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=8`
- `max_training_seq_length=2048`
- `learning_rate=3e-5`
- LoRA fine-tuning with value head.
- `value_weight=0.05`
- `boundary_value_weight=0.02`
- deterministic sequential sampling, seed `20260513`.

## Trained Models

The old trained models already exist and were reused for old-prompt rerun evaluation.

```text
model_training/src/output/review-lora-qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514-static_optimistic-160step
model_training/src/output/review-lora-qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514-mcts_optimistic-160step
```

Both model directories contain:

```text
adapter_model.safetensors
adapter_config.json
value_head.pth
tokenizer.json
tokenizer_config.json
training_args.bin
checkpoint-80/
checkpoint-160/
```

## Original Old-Prompt Evaluation

Original eval output directories:

```text
model_training/src/output/review-eval-qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514-static_optimistic-nonoptimistic_eval
model_training/src/output/review-eval-qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514-mcts_optimistic-nonoptimistic_eval
```

Original eval raw file counts:

| Method | Raw files | Valid scored rows |
|---|---:|---:|
| Static | 514 | 513 |
| MCTS | 492 | 491 |
| Common valid overlap | - | 490 |

Original 490-overlap metrics:

| Method | Spearman | Kendall | Pearson | MAE | mean error | boundary@5 |
|---|---:|---:|---:|---:|---:|---:|
| Static | 0.8128 | 0.6501 | 0.8555 | 1.3408 | -1.1367 | 0.9306 |
| MCTS | 0.8046 | 0.6444 | 0.8426 | 1.2878 | -0.8347 | 0.9122 |

Additional ordinal/agreement metrics on the same 490 examples:

| Method | Alpha | quadratic kappa | ICC2 | RMSE | EMD |
|---|---:|---:|---:|---:|---:|
| Static | 68.86 | 78.71 | 78.74 | 1.91 | 1.137 |
| MCTS | 73.10 | 80.95 | 80.99 | 1.84 | 0.943 |

Interpretation:

- Static is slightly better on rank correlation: Spearman/Kendall/Pearson.
- MCTS is better on ordinal agreement, RMSE, EMD, and mean signed error.
- Both methods are pessimistic on average, but MCTS is less pessimistic.

## Old-Prompt Full Rerun

To avoid overwriting the original eval outputs, the old-prompt rerun uses new output directories:

```text
model_training/src/output/review-eval-qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514-static_oldprompt_rerun-nonoptimistic_eval
model_training/src/output/review-eval-qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514-mcts_oldprompt_rerun-nonoptimistic_eval
```

Rerun code path:

```text
/tmp/cc-oldprompt-ba27cf0
```

Rerun launch time:

```text
2026-05-14 19:24 +0800
```

Rerun tmux sessions:

```text
cc_oldprompt_eval_static
cc_oldprompt_eval_mcts
```

Final rerun status:

| Method | Raw files completed |
|---|---:|
| Static old-prompt rerun | 859 |
| MCTS old-prompt rerun | 859 |

Full rerun common-valid overlap:

```text
857 examples
```

Full rerun metrics on the 857 common-valid examples:

| Method | Alpha | QWK | ICC2 | RMSE | EMD | MAE |
|---|---:|---:|---:|---:|---:|---:|
| Static | 60.75 | 76.66 | 76.68 | 2.0035 | 1.3162 | 1.4749 |
| MCTS | 65.45 | 78.83 | 78.85 | 1.9396 | 1.0860 | 1.4026 |

Notes:

- EMD is lower-is-better and uses the `1..10` score support.
- The MCTS rerun has two parsed predictions equal to `0`; the EMD computation follows the earlier metric code and does not include `0` in the `1..10` frequency support.
- The full rerun is worse than the original stopped 490-example subset in absolute metric values, but preserves the same direction: MCTS is better on Alpha, QWK, ICC2, RMSE, EMD, and MAE.

## Reproduction Pointers

Use `ba27cf0` for the old-prompt code path:

```bash
git worktree add --detach /tmp/cc-oldprompt-ba27cf0 ba27cf0
```

Then use the old eval worker with the existing trained models:

```bash
RUN_NAME=qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514 \
TAG=mcts_oldprompt \
MODEL_DIR=/data1/xianzhiwei/mcts-code-review/model_training/src/output/review-lora-qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514-mcts_optimistic-160step \
GPU_ID=1 \
EVAL_DATA=/data1/xianzhiwei/mcts-code-review/data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514/seed_eval_nonoptimistic.jsonl \
INDICES_FILE=/data1/xianzhiwei/mcts-code-review/data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514/eval_nonoptimistic_indices.json \
OUTPUT_DIR=/data1/xianzhiwei/mcts-code-review/model_training/src/output/review-eval-qwen3_4b_codecritic_easy_correctness_all_optimistic_20260514-mcts_oldprompt_rerun-nonoptimistic_eval \
/tmp/cc-oldprompt-ba27cf0/data_collection/scripts/run_codecritic_easy_qwen3_4b_eval_worker.sh
```

Use the analogous Static command with `TAG=static_oldprompt`, `GPU_ID=0`, the static model directory, and the static output directory.
