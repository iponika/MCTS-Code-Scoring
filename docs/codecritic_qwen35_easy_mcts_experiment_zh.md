# CodeCriticBench Easy Correctness MCTS 实验（中文）

日期：2026-05-14

本文记录当前 Qwen3.5-9B CodeCriticBench Easy correctness optimistic-seed 实验，避免和旧的 AXIOM 实验、旧的 508 seed / 357 policy 样本实验混淆。

## 目标

只训练和评估 `Correctness Verification` 这一维度。

由于base组不用训练，当前训练两个变体：

- `MCTS optimistic`：对全量 eligible non-QA Easy CodeGen seed 生成 MCTS 树，筛选乐观树，再从乐观树中抽取 policy 和 value-only 样本。
- `Static optimistic`：对同一批乐观 seed 使用 CodeCritic 静态 `correctness_score` 标签训练。

当前 run 目录：

```bash
data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513
```

## Seed 选取原则（数据清洗）

源数据：

```bash
datasets/CodeCriticBench/data/CodeCriticBench.jsonl
```

纳入条件：

- `source in {mbpp, codeforce, live-code-bench, debug}`
- `difficulty == "Easy"`
- 含有 `Correctness Verification` checklist 维度
- 该维度有 checklist score 和 checklist text

排除条件：

- `source == stackoverflow`
- 非 Easy 样本
- 缺失 `Correctness Verification` advanced label
- 标签和分数冲突：
  - `correctness == "Error"` 且 `Correctness Verification` 分数大于 5
  - `correctness == "Correct"` 且 `Correctness Verification` 分数小于 5

`Correctness Verification` 分数抽取规则：使用原始 `checklist_dimensions` 中第一个 `Correctness Verification` 对应的 score，不使用 dict 覆盖。这样避免重复维度时后一个分数覆盖第一个分数。

最终 seed 统计：

| 项目 | 数量 |
|---|---:|
| CodeGen Easy 总数 | 1164 |
| 缺失 Correctness Verification | 57 |
| 标签/分数冲突 | 51 |
| eligible seed | 1056 |

eligible 分层使用第一个 `Correctness Verification` 分数：

| 层 | 分数范围 | 数量 |
|---|---:|---:|
| low | 0-3 | 525 |
| mid | 4-6 | 204 |
| high | 7-10 | 327 |

生成全量 seed 的脚本：

```bash
PYTHONPATH=data_collection python data_collection/prepare_codecritic_easy_correctness_splits.py \
  --input datasets/CodeCriticBench/data/CodeCriticBench.jsonl \
  --train_output data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/seed_all_nonqa_easy.jsonl \
  --eval_output data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/seed_eval_optimistic_empty.jsonl \
  --metadata data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/split_all_nonqa_easy_metadata.json \
  --all_train \
  --seed 20260513
```

## MCTS 树生成规则

先生成清洗后所有1056个种子的MCTS树，然后合并检查：

- `mcts_all_eligible_normalized.jsonl` 共 1056 棵树
- 1056 个唯一 seed key
- 无重复树
- 无 terminal target mismatch

相关脚本：

- `data_collection/scripts/normalize_codecritic_mcts_records.py`
- `data_collection/scripts/finalize_codecritic_optimistic_mcts.sh`
- `data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/finalize_optimistic_after_missing_mcts.sh`

## 乐观树选择原则

为减轻MCTS的低估倾向，挑选偏乐观种子作为训练集，其他作为评测集。
对每棵 MCTS 树统计 terminal CodeCritic review 的预测分数和目标分数：

```text
over_count  = count(predicted_correctness_score > target_correctness_score)
under_count = count(predicted_correctness_score < target_correctness_score)
保留条件：over_count > under_count
```

最终 optimistic selection：

| 项目 | 数量 |
|---|---:|
| 全量 normalized MCTS 树 | 1056 |
| optimistic trees / seeds | 278 |
| pessimistic trees | 697 |
| tied trees | 81 |

输出文件：

- `mcts_optimistic_over_gt_under_records.jsonl`
- `seed_optimistic_over_gt_under.jsonl`
- `optimistic_selection_stats.json`

## MCTS Policy/Value 样本抽取原则

Policy 样本：

- 只从 optimistic trees 中抽取。
- terminal review 满足 `abs(predicted_correctness_score - target_correctness_score) <= 2` 才可作为 policy。
- 若同一树有多个可用 terminal，按已有 preprocess 逻辑选 best path。
- 弱 policy（±1/±2）保留，但使用较低 LM 权重。

Value-only 样本：

- 只从产出了 policy 样本的 seed tree 中抽取。
- 高质量 value path 定义为 `abs(predicted_correctness_score - target_correctness_score) <= 2`。
- 低质量 value path 为 ±2 之外、解析错误或有 terminal error 的路径。
- 每棵 policy-producing tree 最多抽 3 条高质量 value path。
- 低质量 value path 最多抽 `min(3, sampled_high_quality_count)` 条。

preprocess 命令使用：

```bash
PYTHONPATH=/data1/ruiqi/MCTS-Code-Scoring/model_training/src:/data1/ruiqi/MCTS-Code-Scoring \
python -m magicoder.preprocess_review_mcts_data \
  --input data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/mcts_optimistic_over_gt_under_records.jsonl \
  --output_file data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/mcts_optimistic_policy_pm2_value_pm2_aligned_train.jsonl \
  --policy_min_q -1.0 \
  --policy_max_grade_delta 2 \
  --max_value_paths_per_dimension 0 \
  --max_value_paths_per_sample_label 3 \
  --value_only_from_policy_records_only \
  --value_sampling_seed 20260513 \
  --no-shuffle
```

最终 MCTS 训练样本：

| 项目 | 数量 |
|---|---:|
| 原始 MCTS train items | 1141 |
| 原始 policy items | 224 |
| 原始 value-only items | 917 |
| `<=6200 tokens` 后保留 | 1132 |
| 保留 policy items | 223 |
| 保留 value-only items | 909 |
| 超长丢弃 | 9 |
| p50 tokens | 1117 |
| p95 tokens | 3748 |
| max raw tokens | 7589 |

最终 MCTS 训练文件：

```text
data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/mcts_optimistic_policy_pm2_value_pm2_aligned_le6200_train.jsonl
```

## Static Optimistic 样本

Static 使用乐观 seed 作为训练集，数量与MCTS的policy样本数对齐：

```text
data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/seed_optimistic_over_gt_under.jsonl
```

生成命令：

```bash
PYTHONPATH="${PWD}/model_training/src:data_collection:${PWD}" \
python data_collection/prepare_static_review_train_data.py \
  --input data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/seed_optimistic_over_gt_under.jsonl \
  --output data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/static_optimistic_codecritic_train.jsonl \
  --prompt_variant base_static \
  --output_schema codecritic_correctness
```

最终 static 训练样本：

| 项目 | 数量 |
|---|---:|
| static optimistic items | 278 |
| `<=6200 tokens` 后保留 | 278 |
| 超长丢弃 | 0 |
| p50 tokens | 759 |
| p95 tokens | 1153 |
| max tokens | 1526 |

最终 static 训练文件：

```text
data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/static_optimistic_codecritic_le6200_train.jsonl
```

## 训练配置

MCTS 训练脚本：

```text
data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/train_optimistic_le6200_3gpu.sh
```

关键配置：

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

Static 训练脚本：

```text
data_collection/review_mcts_runs/qwen35_codecritic_easy_correctness_20260513/train_static_optimistic_gpu3.sh
```

关键配置：

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

训练 tmux session：

- MCTS: `cc_easy_optimistic_mcts_3gpu`
- Static: `cc_easy_optimistic_static_gpu3`
