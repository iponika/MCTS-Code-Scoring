# CodeCriticBench Easy Correctness MCTS Experiment

Date: 2026-05-13

This document records the current experiment design and the exact data production steps. It is written so the same experiment can be replayed on another server, including a Qwen3-4B server, without mixing it with older AXIOM or full-dataset runs.

## Goal

Train and compare review models on CodeCriticBench Code Generation Easy samples for the single dimension `Correctness Verification`.

Current active variants:

- `MCTS`: train on selected MCTS policy paths plus sampled value-only paths from the same MCTS trees.
- `Static`: train on exact reference labels for the same seed items that produced usable MCTS policy samples.
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
- rows where the advanced `Correctness Verification` score conflicts with the basic correctness label:
  - `correctness == "Error"` and first `Correctness Verification` score `> 5`
  - `correctness == "Correct"` and first `Correctness Verification` score `< 5`

`target_correctness_score` uses the first `Correctness Verification` checklist score in the raw row. Some raw rows contain duplicate `Correctness Verification` dimensions; later duplicates are treated as auxiliary checklist questions and must not overwrite the first score.

Observed local counts:

| Split basis | Count |
|---|---:|
| CodeGen Easy total | 1164 |
| Missing Correctness Verification advanced label | 57 |
| Label/score conflicts filtered | 51 |
| Eligible CodeGen Easy after filtering | 1056 |

Layering uses the original CodeCriticBench overall `score`, not the per-dimension correctness score:

| Layer | Overall score range | Original Easy count | Eligible count after filtering |
|---|---:|---:|---:|
| Low | 0-3 | 809 | 708 |
| Mid | 4-6 | 17 | 14 |
| High | 7-10 | 338 | 334 |

## Fixed Train/Eval Split

Create the seed split with seed `20260513`:

```bash
mkdir -p "${RUN_DIR}"

PYTHONPATH=data_collection python data_collection/prepare_codecritic_easy_correctness_splits.py \
  --input datasets/CodeCriticBench/data/CodeCriticBench.jsonl \
  --train_output "${RUN_DIR}/seed_train.jsonl" \
  --eval_output "${RUN_DIR}/seed_eval.jsonl" \
  --metadata "${RUN_DIR}/split_metadata.json" \
  --target_train 500 \
  --seed 20260513
```

Expected split:

| Split | Low | Mid | High | Total |
|---|---:|---:|---:|---:|
| Train seeds | 347 | 14 | 145 | 506 |
| Eval seeds | 361 | 0 | 189 | 550 |

Selection rule:

- Low/high counts use `floor(500 * original_layer_count / 1164)`.
- Mid is scarce, so all eligible Mid rows are retained.
- Eval is all remaining eligible Easy rows.

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

Run 4 MCTS shards over the 508 train seeds. The current run used contiguous ranges:

```bash
mkdir -p "${RUN_DIR}/mcts_samples" "${RUN_DIR}/logs"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=data_collection \
python data_collection/solver_review.py \
  --custom_cfg data_collection/configs/mcts_codecritic_qwen35_9b_no_think.yaml \
  --dataset "${RUN_DIR}/seed_train.jsonl" \
  --start 0 --limit 127 \
  --output "${RUN_DIR}/mcts_shard_0.jsonl" \
  --output_dir "${RUN_DIR}/mcts_samples/shard_0"

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=data_collection \
python data_collection/solver_review.py \
  --custom_cfg data_collection/configs/mcts_codecritic_qwen35_9b_no_think.yaml \
  --dataset "${RUN_DIR}/seed_train.jsonl" \
  --start 127 --limit 127 \
  --output "${RUN_DIR}/mcts_shard_1.jsonl" \
  --output_dir "${RUN_DIR}/mcts_samples/shard_1"

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=data_collection \
python data_collection/solver_review.py \
  --custom_cfg data_collection/configs/mcts_codecritic_qwen35_9b_no_think.yaml \
  --dataset "${RUN_DIR}/seed_train.jsonl" \
  --start 254 --limit 127 \
  --output "${RUN_DIR}/mcts_shard_2.jsonl" \
  --output_dir "${RUN_DIR}/mcts_samples/shard_2"

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=data_collection \
python data_collection/solver_review.py \
  --custom_cfg data_collection/configs/mcts_codecritic_qwen35_9b_no_think.yaml \
  --dataset "${RUN_DIR}/seed_train.jsonl" \
  --start 381 --limit 127 \
  --output "${RUN_DIR}/mcts_shard_3.jsonl" \
  --output_dir "${RUN_DIR}/mcts_samples/shard_3"
```

On the Qwen3-4B server, replace the config path with `mcts_codecritic_qwen3_4b_no_think.yaml`. Keep the split seed and shard ranges unchanged if you want a matched experiment.

## MCTS Policy And Value Dataset

After all `mcts_shard_*.jsonl` files finish, convert them into training data.

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
- Within each such seed tree, sample at most `3` correct value-only paths and at most `3` incorrect value-only paths.
- If there are fewer than 3 in either label bucket, keep all available paths.
- Drop value-only paths from seed trees with no policy sample.
- Use deterministic `--value_sampling_seed 20260513`.

Command:

```bash
PYTHONPATH="${PWD}/model_training/src:${PWD}" \
python model_training/src/magicoder/preprocess_review_mcts_data.py \
  --input "${RUN_DIR}"/mcts_shard_*.jsonl \
  --output_file "${RUN_DIR}/mcts_policy_pm2_train.jsonl" \
  --policy_min_q -1.0 \
  --policy_max_grade_delta 2 \
  --max_value_paths_per_dimension 0 \
  --max_value_paths_per_sample_label 3 \
  --value_only_from_policy_records_only \
  --value_sampling_seed 20260513 \
  --no-shuffle \
  > "${RUN_DIR}/mcts_policy_pm2_train_stats.json"
```

Current Qwen3.5-9B result before token filtering:

| Item | Count |
|---|---:|
| MCTS seed records | 508 |
| terminal reviews | 6700 |
| raw seeds with ±2 terminal | 401 |
| usable policy paths after quality filters | 357 |
| exact-score policy paths | 74 |
| weak ±2 policy paths | 283 |
| sampled value-only paths | 1146 |
| output train items | 1503 |

Sampled value-only paths:

| Bucket | Count |
|---|---:|
| correct | 131 |
| incorrect | 1015 |
| dropped because seed tree had no policy path | 2004 |

## Token Filtering

Filter the MCTS training set with the training tokenizer and `max_tokens=6200`. This preserves all 357 policy samples and drops only too-long value-only items.

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
src = run_dir / "mcts_policy_pm2_train.jsonl"
dst = run_dir / "mcts_policy_pm2_train_policy357_value3x3_le6200.jsonl"
stats_path = run_dir / "mcts_policy_pm2_train_policy357_value3x3_le6200_stats.json"
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

Current Qwen3.5-9B result after filtering:

| Item | Count |
|---|---:|
| input items | 1503 |
| kept items | 1500 |
| kept policy items | 357 |
| kept value-only items | 1143 |
| dropped too long | 3 |
| max kept tokens | 6087 |

## Static Matched Dataset

The Static baseline must use exactly the seed items that produced policy samples in the MCTS dataset. For the current run this is 357 seed items.

Create `seed_train_policy357.jsonl` by selecting the `(source, subset, dataset_index)` keys with `train_lm=true` in the filtered MCTS dataset:

```bash
PYTHONPATH="${PWD}/model_training/src:${PWD}" python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
mcts_train = run_dir / "mcts_policy_pm2_train_policy357_value3x3_le6200.jsonl"
seed_train = run_dir / "seed_train.jsonl"
seed_policy = run_dir / "seed_train_policy357.jsonl"

keys = set()
with mcts_train.open(encoding="utf-8") as handle:
    for line in handle:
        item = json.loads(line)
        if item.get("train_lm"):
            keys.add((item.get("source"), item.get("subset"), item.get("dataset_index")))

rows = []
with seed_train.open(encoding="utf-8") as handle:
    for line in handle:
        item = json.loads(line)
        key = (item.get("source"), item.get("subset"), item.get("dataset_index"))
        if key in keys:
            rows.append(item)

with seed_policy.open("w", encoding="utf-8") as writer:
    for row in rows:
        writer.write(json.dumps(row, ensure_ascii=False) + "\n")

print(json.dumps({"policy_keys": len(keys), "seed_rows": len(rows), "output": str(seed_policy)}, indent=2))
PY
```

Generate static exact-label training items:

```bash
PYTHONPATH="${PWD}/model_training/src:data_collection:${PWD}" \
python data_collection/prepare_static_review_train_data.py \
  --input "${RUN_DIR}/seed_train_policy357.jsonl" \
  --output "${RUN_DIR}/static_policy357_train.jsonl" \
  --prompt_variant base_static
```

Token-audit/filter static data with the same tokenizer. In the current run no static samples exceed 6200 tokens:

```bash
cp "${RUN_DIR}/static_policy357_train.jsonl" "${RUN_DIR}/static_policy357_le6200_train.jsonl"
```

Current static stats:

| Item | Count |
|---|---:|
| input items | 357 |
| kept items | 357 |
| dropped too long | 0 |
| max tokens | 2023 |
| p50 tokens | 845 |
| p95 tokens | 1439 |

## Training Configuration

### MCTS Training

Current Qwen3.5-9B training script:

```bash
data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/train_policy357_le6200_3gpu.sh
```

Key settings:

```text
train data: mcts_policy_pm2_train_policy357_value3x3_le6200.jsonl
items: 1500 = 357 policy + 1143 value-only
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
data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/train_static357_gpu3.sh
```

Key settings:

```text
train data: static_policy357_le6200_train.jsonl
items: 357
prompt_variant: base_static
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

## Current Running Qwen3.5-9B Jobs

These are the live runs started from the current data:

| Variant | tmux session | Log |
|---|---|---|
| MCTS train | `cc_easy_policy357_3gpu` | `${RUN_DIR}/logs/train_policy357_le6200_3gpu_20260513_231103.log` |
| Static train | `cc_easy_static357_gpu3` | `${RUN_DIR}/logs/train_static357_gpu3_20260513_232659.log` |

The MCTS job uses 0/1/2 with bf16 and FSDP offload. The Static job uses physical GPU 3, bf16, and `max_training_seq_length=2048`.

## Qwen3-4B Replay Checklist

Use the same dataset split and data production logic, but write to a new run directory.

Required changes:

- Set `RUN_DIR` to a Qwen3-4B-specific path.
- Copy the MCTS YAML and set `model_dir` to Qwen3-4B.
- In training scripts set `MODEL_KEY="Qwen/Qwen3-4B"` and `MODEL_PATH` to the local Qwen3-4B snapshot.
- For MCTS training with FSDP, use `--fsdp_transformer_layer_cls_to_wrap Qwen3DecoderLayer`.
- Retain `--bf16 True`, `--mixed_precision bf16`, and `--fsdp_offload_params true` for the MCTS policy/value training.
- Recompute token-filter stats with the Qwen3-4B tokenizer, even if the source JSONL is the same.
- Generate Static from the policy-producing seed keys of that run, not by copying the 9B static file.

Expected matched outputs if the Qwen3-4B MCTS behavior matches the 9B run are not guaranteed. The number of policy seeds `P` may differ because MCTS generations and score parses are model-dependent. Always set Static to exactly that run's final usable policy count `P`.
