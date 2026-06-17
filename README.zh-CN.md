# SPARC：基于 MCTS 引导自训练的代码评分框架

[English README](README.md)

本仓库包含 **SPARC** 的实现材料。SPARC 面向基于评价准则的代码评分任务，使用 Monte-Carlo Tree Search (MCTS) 生成经过奖励过滤的推理轨迹，并训练带 policy head 和 value head 的 LoRA 模型，在推理阶段通过 value-guided candidate selection 提升代码正确性评分的稳定性。

本仓库作为论文审稿 artifact 组织。大型数据集、搜索树、训练 JSONL 和模型 checkpoint 不进入 Git；复现实验时需要按本地服务器环境单独恢复。

## 项目结构

```text
data_collection/
  solver_review.py                         # MCTS 推理轨迹生成入口
  direct_bootstrap_review.py               # Direct baseline 数据生成
  prepare_codecritic_axiom_seedset.py      # CodeCriticBench 种子准备
  prepare_static_review_train_data.py      # 精确标签 baseline 数据准备
  rebalance_review_train_data.py           # policy/value 样本平衡
  configs/                                 # 模型与 MCTS 配置
  scripts/                                 # 实验 wrapper

model_training/src/magicoder/
  preprocess_review_mcts_data.py           # 搜索树/轨迹导出为训练 JSONL
  preprocess_score_datasets.py             # 静态评分数据预处理
  train_multi.py                           # LoRA policy/value 训练
  review_evaluator.py                      # 代码评分评测器
  review_policy_value_inference.py         # value head 检查工具
  review_value_guided_evaluator.py         # 单样本 value-guided 推理

paper/
  69fdb1f6e7574fa1c601d7fa/JASE/          # 论文正文与图片

tests/                                     # review 主线回归测试
tools/mcts_tree_viewer.html                # 本地 MCTS 树查看器
docs/                                      # 面向审稿的说明与 case 证据
```

## 数据

数据集不随 Git 跟踪。默认脚本路径如下：

```text
datasets/CodeCriticBench/data/CodeCriticBench.jsonl
datasets/axiom-llm-judge/axiombench/*.jsonl
```

论文主实验使用 CodeCriticBench 的 CodeGen 子集，并在预处理后得到 2,631 条种子样本，划分为 1,316 条训练样本和 1,315 条保留评测样本。当前 artifact 使用的划分 ID 记录在：

```text
docs/codecriticbench_1316_1315_split_id_record_20260617.md
```

## 环境

优先使用目标服务器提供的 Python/CUDA 环境。确认 PyTorch 和 CUDA 版本匹配后，用 `uv` 安装项目依赖：

```bash
uv pip install -r requirements.txt
```

Qwen3.5 系列 checkpoint 需要 Transformers 支持 `model_type=qwen3_5`。如果服务器已有兼容版本，也可以继续使用服务器固定环境。

## 快速检查

在仓库根目录运行回归测试：

```bash
PYTHONPATH=data_collection:model_training/src uv run pytest tests
```

准备一个极小 CodeCriticBench seed 样本：

```bash
PYTHONPATH=data_collection uv run python data_collection/prepare_codecritic_axiom_seedset.py \
  --output /tmp/codecritic_seed.jsonl \
  --metadata /tmp/codecritic_seed.meta.json \
  --per_grade 1 \
  --min_grade 1 \
  --max_grade 5
```

恢复所需模型 checkpoint 后，可运行单条 MCTS 轨迹 smoke test：

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

## 复现实验流程

论文实验流程包含四个阶段：

1. 准备 CodeCriticBench 评分种子。
2. 生成经过奖励过滤的 MCTS 推理轨迹。
3. 导出 policy/value 训练数据并训练 LoRA policy-value 模型。
4. 在保留评测集上比较 Base、Direct、消融模型与 SPARC。

长任务建议放入 `tmux`：

```bash
tmux new -s sparc_job
# 在 tmux 中运行实验命令
# Ctrl-b d 断开
tmux attach -t sparc_job
```

生成产物默认写入以下被忽略目录：

```text
data_collection/review_mcts_runs/
model_training/review_mcts_train_data/
model_training/src/output/
```
