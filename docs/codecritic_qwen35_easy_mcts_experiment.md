# CodeCriticBench Easy Correctness MCTS Experiment

Date: 2026-05-14

This document records the current optimistic-seed experiment design and the exact data production steps. It is written so the same experiment can be replayed on another server, including a Qwen3-4B server, without mixing it with older AXIOM runs or the older 508-seed/357-policy CodeCritic run.

## Goal

Train and compare review models on CodeCriticBench Code Generation Easy samples for the single dimension `Correctness Verification`.

Current active variants:

- `MCTS optimistic`: run MCTS on all eligible non-QA Easy CodeGen seeds, keep only optimistic trees, then train on selected MCTS policy paths plus sampled value-only paths from those trees.
- `Static optimistic`: train on exact reference labels for the same optimistic seed items.
- `Direct`: not trained in this round.

The current Qwen3.5-9B run directory is:

```bash
export RUN_DIR=data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513
```

For a Qwen3-4B replay, use a new run directory, for example:

```bash
export RUN_DIR=data_collection/review_mcts_runs/qwen3_4b_codecritic_easy_correctness_20260513
```

Do not reuse files from older runs such as AXIOM seed sets or old `review_mcts_train_data` outputs.

## Dataset Scope

Source file:

```bash
datasets/CodeCriticBench/data/CodeCriticBench.jsonl
```

Included rows:

- `source in {mbpp, codeforce, live-code-bench, debug}`
- `difficulty == "Easy"`
- row has `Correctness Verification` in `checklist_dimensions`
- that dimension has both checklist score and checklist text

Excluded rows:

- `source == stackoverflow`
- non-Easy rows
- Easy CodeGen rows missing `Correctness Verification` advanced labels
- label/score conflicts:
  - `correctness == "Error"` and `Correctness Verification` score is greater than `5`
  - `correctness == "Correct"` and `Correctness Verification` score is less than `5`

Correctness score extraction uses the first raw `Correctness Verification` entry in `checklist_dimensions`. This avoids silently overwriting repeated dimensions.

Observed local counts after the 2026-05-14 filtering change:

| Split basis | Count |
|---|---:|
| CodeGen Easy total | 1164 |
| Missing Correctness Verification advanced label | 57 |
| Label/score conflicts | 51 |
| Eligible CodeGen Easy | 1056 |

Layering uses the first `Correctness Verification` score:

| Layer | Correctness score range | Eligible count |
|---|---:|---:|
| Low | 0-3 | 525 |
| Mid | 4-6 | 204 |
| High | 7-10 | 327 |

## All Eligible Seed Set

Create the all-eligible seed file with seed `20260513`:

```bash
mkdir -p "${RUN_DIR}"

PYTHONPATH=data_collection python data_collection/prepare_codecritic_easy_correctness_splits.py \
  --input datasets/CodeCriticBench/data/CodeCriticBench.jsonl \
  --train_output "${RUN_DIR}/seed_all_nonqa_easy.jsonl" \
  --eval_output "${RUN_DIR}/seed_eval.jsonl" \
  --metadata "${RUN_DIR}/split_metadata.json" \
  --all_train \
  --seed 20260513
```

Expected output:

| Output | Low | Mid | High | Total |
|---|---:|---:|---:|---:|
| `seed_all_nonqa_easy.jsonl` | 525 | 204 | 327 | 1056 |
| `seed_eval.jsonl` | 0 | 0 | 0 | 0 |

Selection rule:

- Every eligible seed is used for MCTS generation.
- The training set is selected after MCTS generation using the optimistic-tree rule below.
- Use a separate held-out file for evaluation if needed; do not reuse the older 599-row eval split as if it matched this optimistic training design.

## MCTS Generation

Current Qwen3.5-9B config:

```bash
data_collection/configs/mcts_codecritic_qwen35_9b_no_think.yaml
```

Important MCTS config values:

```yaml
mode: "mcts"
model_dir: "Qwen/Qwen3.5-9B"
temperature: 0.7
top_p: 0.8
top_k: 20
max_tokens: 1536
max_model_len: 16384
use_chat_template: True
chat_template_enable_thinking: False
qwen_thinking_mode: "no_think"
review_prompt_variant: "codecritic_correctness"
show_tests_in_prompt: True
max_review_dimensions: 1
max_depth: 3
n_generate_sample: 3
iterations: 18
batch_size: 2
review_explore_depth: 2
review_max_exploration_nodes: 24
review_finalize_frontier_limit: 24
review_target_leaf_count: 24
remove_duplicate: True
disable_process_pool: True
```

The prompt/scoring contract is:

- score only `Correctness Verification`
- expose reviewer-visible tests in the prompt
- output schema uses `correctness_score`, not `axiom_grade`
- final score range is integer `1..10`
- Qwen no-think mode

For Qwen3-4B, copy the config to a new file and change at least `model_dir`, for example:

```bash
cp data_collection/configs/mcts_codecritic_qwen35_9b_no_think.yaml \
  data_collection/configs/mcts_codecritic_qwen3_4b_no_think.yaml
# Edit model_dir to the local/HF Qwen3-4B model path.
```

Run 4 MCTS shards over the 1056 all-eligible seeds. A 4-GPU replay can use contiguous ranges of 264 rows:

```bash
mkdir -p "${RUN_DIR}/mcts_samples" "${RUN_DIR}/logs"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=data_collection \
python data_collection/solver_review.py \
  --custom_cfg data_collection/configs/mcts_codecritic_qwen35_9b_no_think.yaml \
  --dataset "${RUN_DIR}/seed_all_nonqa_easy.jsonl" \
  --start 0 --limit 264 \
  --output "${RUN_DIR}/mcts_shard_0.jsonl" \
  --output_dir "${RUN_DIR}/mcts_samples/shard_0"

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=data_collection \
python data_collection/solver_review.py \
  --custom_cfg data_collection/configs/mcts_codecritic_qwen35_9b_no_think.yaml \
  --dataset "${RUN_DIR}/seed_all_nonqa_easy.jsonl" \
  --start 264 --limit 264 \
  --output "${RUN_DIR}/mcts_shard_1.jsonl" \
  --output_dir "${RUN_DIR}/mcts_samples/shard_1"

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=data_collection \
python data_collection/solver_review.py \
  --custom_cfg data_collection/configs/mcts_codecritic_qwen35_9b_no_think.yaml \
  --dataset "${RUN_DIR}/seed_all_nonqa_easy.jsonl" \
  --start 528 --limit 264 \
  --output "${RUN_DIR}/mcts_shard_2.jsonl" \
  --output_dir "${RUN_DIR}/mcts_samples/shard_2"

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=data_collection \
python data_collection/solver_review.py \
  --custom_cfg data_collection/configs/mcts_codecritic_qwen35_9b_no_think.yaml \
  --dataset "${RUN_DIR}/seed_all_nonqa_easy.jsonl" \
  --start 792 --limit 264 \
  --output "${RUN_DIR}/mcts_shard_3.jsonl" \
  --output_dir "${RUN_DIR}/mcts_samples/shard_3"
```

On the Qwen3-4B server, replace the config path with `mcts_codecritic_qwen3_4b_no_think.yaml`. Keep the seed-generation command and shard ranges unchanged if you want a matched experiment.

## Optimistic MCTS Policy And Value Dataset

After all `mcts_shard_*.jsonl` files finish, first select optimistic trees:

```text
over_count  = count(predicted_correctness_score > target_correctness_score)
under_count = count(predicted_correctness_score < target_correctness_score)
keep tree iff over_count > under_count
```

Then convert only those optimistic trees into training data.

Policy selection:

- Inspect terminal review nodes.
- A seed contributes one policy sample only if at least one valid terminal review satisfies:

```text
abs(predicted_correctness_score - target_correctness_score) <= 2
```

- Pick the best valid terminal path by highest `q_value`, then smaller score distance.
- If no terminal review is within ±2, do not train LM policy on that seed.

Value-only sampling:

- Restrict value-only paths to seed trees that produced a policy sample.
- High-quality value paths are terminal paths with `abs(predicted_correctness_score - target_correctness_score) <= 2`.
- Low-quality value paths are terminal paths outside ±2, errored paths, or invalid parsed paths.
- Within each policy-producing seed tree, sample at most `3` high-quality value-only paths.
- Low-quality value-only paths are capped to `min(3, sampled_high_quality_count)`.
- Drop value-only paths from seed trees with no policy sample.
- Use deterministic `--value_sampling_seed 20260513`.

Command:

```bash
RUN_NAME=qwen35_codecritic_easy_correctness_20260513 \
RUN_DIR="${PWD}/${RUN_DIR}" \
SEED_DATA="${PWD}/${RUN_DIR}/seed_all_nonqa_easy.jsonl" \
SHARD_GLOB="${PWD}/${RUN_DIR}/mcts_shard_*.jsonl" \
data_collection/scripts/finalize_codecritic_optimistic_mcts.sh
```

Main outputs:

- `mcts_optimistic_over_gt_under_records.jsonl`
- `seed_optimistic_over_gt_under.jsonl`
- `optimistic_selection_stats.json`
- `mcts_optimistic_policy_pm2_value_pm2_aligned_train.jsonl`
- `mcts_optimistic_policy_pm2_value_pm2_aligned_train_stats.json`

Final 2026-05-14 selection result:

| Item | Count |
|---|---:|
| all eligible normalized MCTS trees | 1056 |
| optimistic trees / seeds | 278 |
| pessimistic trees | 697 |
| tied trees | 81 |

Normalization safeguards:

- The older 508-seed MCTS run was not reused blindly.
- 27 old trees were removed because their seeds fail the new label/score conflict filter.
- 29 otherwise eligible old trees had terminal `reward_details.target_correctness_score` from the previous target extraction logic; these were regenerated.
- The final merged tree file contains `452` target-consistent old trees, `575` newly generated missing trees, and `29` regenerated target-mismatch trees.
- Final integrity check: `1056` unique seed keys, no duplicate trees, no terminal target mismatch.

Scripts used for seed selection and MCTS dataset construction:

- `data_collection/prepare_codecritic_easy_correctness_splits.py`
- `data_collection/scripts/normalize_codecritic_mcts_records.py`
- `data_collection/scripts/finalize_codecritic_optimistic_mcts.sh`
- `data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/finalize_optimistic_after_missing_mcts.sh`

## Token Filtering

Filter the optimistic MCTS training set with the training tokenizer and `max_tokens=6200`.

Use the local tokenizer path for the model being trained:

```bash
export MODEL_PATH=/path/to/Qwen-model
```

Current Qwen3.5-9B path:

```bash
export MODEL_PATH=/data1/ruiqi/hf_home/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

Reusable filtering snippet:

```bash
PYTHONPATH="${PWD}/model_training/src:${PWD}" python - <<'PY'
import json
import os
from pathlib import Path
from transformers import AutoTokenizer

run_dir = Path(os.environ["RUN_DIR"])
model_path = os.environ["MODEL_PATH"]
src = run_dir / "mcts_optimistic_policy_pm2_value_pm2_aligned_train.jsonl"
dst = run_dir / "mcts_optimistic_policy_pm2_value_pm2_aligned_le6200_train.jsonl"
stats_path = run_dir / "mcts_optimistic_policy_pm2_value_pm2_aligned_le6200_stats.json"
max_tokens = 6200

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
kept, dropped, lengths = [], [], []

def text_for_item(item):
    if item.get("messages"):
        return tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
    return str(item.get("instruction", "")) + "\n" + "\n".join(map(str, item.get("response", [])))

with src.open(encoding="utf-8") as handle:
    for line_no, line in enumerate(handle, start=1):
        item = json.loads(line)
        n = len(tokenizer(text_for_item(item), add_special_tokens=False)["input_ids"])
        lengths.append(n)
        if n <= max_tokens:
            kept.append(item)
        else:
            dropped.append({
                "line_no": line_no,
                "dataset_index": item.get("dataset_index"),
                "source": item.get("source"),
                "subset": item.get("subset"),
                "terminal_tag": item.get("terminal_tag"),
                "train_lm": item.get("train_lm"),
                "token_length": n,
            })

with dst.open("w", encoding="utf-8") as writer:
    for item in kept:
        writer.write(json.dumps(item, ensure_ascii=False) + "\n")

stats = {
    "source_train_jsonl": str(src),
    "output_train_jsonl": str(dst),
    "model_tokenizer": model_path,
    "max_tokens_filter": max_tokens,
    "input_items": len(lengths),
    "kept_items": len(kept),
    "dropped_too_long": len(dropped),
    "kept_policy_items": sum(1 for x in kept if x.get("train_lm")),
    "kept_value_only_items": sum(1 for x in kept if not x.get("train_lm")),
    "dropped_examples": dropped,
}
stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(stats, ensure_ascii=False, indent=2))
PY
```

Final token-filtered MCTS training set:

| Item | Count |
|---|---:|
| raw MCTS train items | 1141 |
| raw policy items | 224 |
| raw value-only items | 917 |
| kept items after `<=6200` filter | 1132 |
| kept policy items | 223 |
| kept value-only items | 909 |
| dropped too long | 9 |
| max raw tokens | 7589 |
| p50 raw tokens | 1117 |
| p95 raw tokens | 3748 |

One policy item exceeded 6200 tokens and was dropped. The final MCTS training file is:

```text
data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/mcts_optimistic_policy_pm2_value_pm2_aligned_le6200_train.jsonl
```

## Static Matched Dataset

The Static baseline should use the optimistic seed items selected from MCTS, not the old 357-policy subset.

The finalize script already writes:

```text
seed_optimistic_over_gt_under.jsonl
```

Use that file for static exact-label training:

```bash
PYTHONPATH="${PWD}/model_training/src:data_collection:${PWD}" \
python data_collection/prepare_static_review_train_data.py \
  --input "${RUN_DIR}/seed_optimistic_over_gt_under.jsonl" \
  --output "${RUN_DIR}/static_optimistic_codecritic_train.jsonl" \
  --prompt_variant base_static \
  --output_schema codecritic_correctness
```

Token-audit/filter static data with the same tokenizer:

```bash
cp "${RUN_DIR}/static_optimistic_codecritic_train.jsonl" "${RUN_DIR}/static_optimistic_codecritic_le6200_train.jsonl"
```

If any static sample exceeds 6200 tokens on the target tokenizer, write a filtered file and record the dropped examples.

Final static optimistic training set:

| Item | Count |
|---|---:|
| optimistic static items | 278 |
| kept after `<=6200` filter | 278 |
| dropped too long | 0 |
| max tokens | 1526 |
| p50 tokens | 759 |
| p95 tokens | 1153 |

The final static training file is:

```text
data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/static_optimistic_codecritic_le6200_train.jsonl
```

## Training Configuration

### MCTS Training

Current Qwen3.5-9B training script:

```bash
data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/train_optimistic_le6200_3gpu.sh
```

Key settings:

```text
train data: mcts_optimistic_policy_pm2_value_pm2_aligned_le6200_train.jsonl
items: 1132 = 223 policy + 909 value-only
max_training_seq_length: 6200
gpus: 0,1,2
mixed precision: bf16
FSDP: enabled
FSDP offload: --fsdp_offload_params true
FSDP activation checkpointing: true
LoRA: rank 8, alpha 16, dropout 0.05, attention targets
batch: per_device_train_batch_size 1, gradient_accumulation_steps 8
steps: 160
optimizer: adafactor
lr: 2e-5 constant
shuffle: disabled, sequential sampling
value loss: value_weight 0.05, boundary_value_weight 0.02, pairwise 0.0
seed: 20260513
```

For Qwen3.5-9B the FSDP layer class is:

```bash
--fsdp_transformer_layer_cls_to_wrap Qwen3_5DecoderLayer
```

For Qwen3-4B use the Qwen3 layer class:

```bash
--fsdp_transformer_layer_cls_to_wrap Qwen3DecoderLayer
```

Also replace:

```bash
MODEL_KEY="Qwen/Qwen3-4B"
MODEL_PATH="/path/to/Qwen3-4B"
```

Keep `--bf16 True`, `--mixed_precision bf16`, and `--fsdp_offload_params true`. The 9B run OOMed without offload, and the current successful run uses bf16 plus FSDP parameter offload.

### Static Training

Current Qwen3.5-9B static script:

```bash
data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/train_static_optimistic_gpu3.sh
```

Key settings:

```text
train data: static_optimistic_codecritic_le6200_train.jsonl
items: 278
output schema: codecritic_correctness
max_training_seq_length: 2048
gpu: 3
mixed precision: bf16
LoRA: rank 8, alpha 16, dropout 0.05, attention targets
batch: per_device_train_batch_size 1, gradient_accumulation_steps 8
steps: 160
optimizer: adafactor
lr: 2e-5 constant
shuffle: disabled, sequential sampling
seed: 20260513
```

When pinning static training to one physical GPU, use only `CUDA_VISIBLE_DEVICES=<gpu>` and do not pass `accelerate --gpu_ids`. Passing `--gpu_ids 0` while another multi-GPU job is running can be interpreted as physical GPU 0 and cause an OOM collision.

## Current Qwen3.5-9B Jobs

The older 357-policy training jobs were stopped before this optimistic dataset rebuild. New tmux sessions should use names and logs that include `optimistic` to avoid mixing runs:

| Variant | tmux session | Log |
|---|---|---|
| MCTS train | `cc_easy_optimistic_mcts_3gpu` | `${RUN_DIR}/logs/train_optimistic_mcts_le6200_3gpu_<timestamp>.log` |
| Static train | `cc_easy_optimistic_static_gpu3` | `${RUN_DIR}/logs/train_optimistic_static_gpu3_<timestamp>.log` |

The MCTS job should use GPUs 0/1/2 with bf16 and FSDP offload. The Static job can use physical GPU 3, bf16, CodeCritic `correctness_score` labels, and `max_training_seq_length=2048`.

## Qwen3-4B Replay Checklist

Use the same all-eligible seed generation, optimistic selection, and data production logic, but write to a new run directory.

Required changes:

- Set `RUN_DIR` to a Qwen3-4B-specific path.
- Copy the MCTS YAML and set `model_dir` to Qwen3-4B.
- In training scripts set `MODEL_KEY="Qwen/Qwen3-4B"` and `MODEL_PATH` to the local Qwen3-4B snapshot.
- For MCTS training with FSDP, use `--fsdp_transformer_layer_cls_to_wrap Qwen3DecoderLayer`.
- Retain `--bf16 True`, `--mixed_precision bf16`, and `--fsdp_offload_params true` for the MCTS policy/value training.
- Recompute token-filter stats with the Qwen3-4B tokenizer, even if the source JSONL is the same.
- Generate Static from `seed_optimistic_over_gt_under.jsonl`, not by copying the 9B static file.
- Use `--output_schema codecritic_correctness`; the old AXIOM static schema is not comparable to this CodeCritic correctness experiment.

Expected matched outputs are not guaranteed. The number of optimistic seeds and usable policy paths may differ because MCTS generations and score parses are model-dependent. Always set Static to exactly that run's `seed_optimistic_over_gt_under.jsonl`.
