# MCTS 代码评分项目

[English README](README.md)

本仓库是对 SEER 风格 MCTS 流程的代码评估改编版本。项目目标是训练和评测能够稳定给候选代码打 AXIOM 风格分数的模型。文本评论只作为评分证据和可解释性辅助。

## 当前范围

- 基于 CodeCriticBench 生成 direct bootstrap 或 review-MCTS 代码评审轨迹。
- 将生成轨迹转换成 policy/value 训练数据。
- 使用 LoRA 和 value head 训练 policy/value 模型。
- 在 AXIOM 风格 held-out 代码评分任务上评测模型。

大型数据集、模型 checkpoint 和实验输出都通过 `.gitignore` 排除。

## 项目结构

```text
data_collection/
  solver_review.py                         # review-MCTS 数据生成入口
  direct_bootstrap_review.py               # 非 MCTS direct bootstrap 数据生成
  prepare_codecritic_axiom_seedset.py      # CodeCriticBench -> AXIOM 对齐种子
  prepare_static_review_train_data.py      # 静态种子 -> 精确 value 标签
  rebalance_review_train_data.py           # 平衡 policy/value 样本
  configs/                                 # review-MCTS 配置
  scripts/                                 # 当前维护的实验 wrapper

model_training/src/magicoder/
  preprocess_review_mcts_data.py           # MCTS/direct 轨迹 -> 训练 JSONL
  preprocess_score_datasets.py             # AXIOM/CodeCritic 静态评分数据
  train_multi.py                           # policy/value LoRA 训练
  review_evaluator.py                      # final-only 和 stepwise 评测
  review_policy_value_inference.py         # value head 检查
  review_value_guided_evaluator.py         # 单样本 value-guided 推理

tests/                                     # review 主线回归测试
tools/mcts_tree_viewer.html                # 本地 MCTS 树查看器
docs/                                      # 设计说明、报告和迁移文档
```

## Clone 后需要准备的数据

数据集不进入 Git。默认路径如下，也可以在命令中用参数覆盖：

```text
datasets/CodeCriticBench/data/CodeCriticBench.jsonl
datasets/axiom-llm-judge/axiombench/*.jsonl
```

后续新增数据集也建议放在 `datasets/` 或 `benchmarks/` 下，这两个目录默认不跟踪。

## 环境安装

在目标服务器上优先使用服务器提供的 CUDA / PyTorch / vLLM 环境。确认 GPU 驱动和 PyTorch 版本匹配后，再用 `uv` 安装项目依赖：

```bash
uv pip install -r requirements.txt
```

如果服务器已经预装了 PyTorch 和 vLLM，并且不希望依赖文件覆盖它们，可以只补齐缺失包：

```bash
uv pip install -r requirements.txt --no-deps
```

Qwen3.5 需要 Transformers 支持 `model_type=qwen3_5`。当前 `requirements.txt` 默认使用 Hugging Face Transformers main 分支。如果服务器已有兼容版本，可以把这一行改成固定 release。

## 快速检查

在仓库根目录运行：

```bash
PYTHONPATH=data_collection:model_training/src uv run pytest tests
```

准备一个极小 CodeCritic seed 集合：

```bash
PYTHONPATH=data_collection uv run python data_collection/prepare_codecritic_axiom_seedset.py \
  --output /tmp/codecritic_seed.jsonl \
  --metadata /tmp/codecritic_seed.meta.json \
  --per_grade 1 \
  --min_grade 1 \
  --max_grade 5
```

## 当前维护的工作流

4B static/direct/MCTS 完整对比：

```bash
tmux new -s review_4b_cmp
RUN_NAME=bootstrap_cmp_qwen3_4b_server \
SEED_PER_GRADE=8 \
MAX_STEPS=120 \
bash data_collection/scripts/run_bootstrap_comparison_qwen3_4b.sh
```

适合大显存服务器的 Qwen3.5-9B direct-review 与 direct-stepwise 对比：

```bash
tmux new -s qwen35_9b_stepwise
RUN_NAME=qwen35_9b_direct_stepwise_server \
SEED_PER_GRADE=8 \
DIRECT_REPEATS=3 \
MAX_STEPS=120 \
MAX_TRAINING_SEQ_LENGTH=2048 \
bash data_collection/scripts/run_qwen35_9b_direct_stepwise_vs_review_smoke.sh
```

对已有 checkpoint 进行 AXIOM held-out 评测：

```bash
RUN_NAME=axiom_eval_server \
TRAINED_MODEL_PATH=/path/to/review-lora-checkpoint \
bash data_collection/scripts/run_axiom_clean_eval.sh
```

保留的 wrapper 默认会根据脚本位置自动推导 `ROOT`，所以 `git clone` 后通常不需要修改绝对路径。只有在特殊目录结构下运行时才需要手动设置 `ROOT=...`。

## 长任务运行方式

数据生成和训练任务建议放在 `tmux` 中运行，避免 SSH 断连导致进程停止：

```bash
tmux new -s mcts_job
# 在 tmux 中运行实验命令
# detach: Ctrl-b d
tmux attach -t mcts_job
```

实验输出默认写入以下被忽略目录：

```text
data_collection/review_mcts_runs/
model_training/review_mcts_train_data/
model_training/src/output/
```

这些目录不会随 Git 提交迁移。如果需要复现实验结果，应单独复制对应 run 目录和 checkpoint。
