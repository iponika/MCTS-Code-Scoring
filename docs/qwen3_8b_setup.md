# Qwen3-8B 适配

默认 8B 模型 ID 为 `Qwen/Qwen3-8B`。该 workflow 复用现有 AXIOM-heavy principle-generalization 流程，但把训练 token 上限提高到 4096，以减少完整样本被过滤掉的比例。

## 下载

```bash
export HF_HOME=/data1/xianzhiwei/model/huggingface
hf download Qwen/Qwen3-8B
```

也可以用 vLLM 首次启动触发下载：

```bash
tmux new-session -d -s qwen3_8b_vllm \
  /data1/xianzhiwei/mcts-code-review/data_collection/scripts/serve_qwen3_8b.sh
```

## 训练/评测

```bash
tmux new-session -d -s principle_generalization_qwen3_8b_full_context_no_axiom0_20260423 \
  /data1/xianzhiwei/mcts-code-review/data_collection/scripts/run_principle_generalization_qwen3_8b.sh
```

默认参数：

```bash
MODEL_KEY=Qwen/Qwen3-8B
MODEL_PATH=Qwen/Qwen3-8B
MAX_TRAINING_SEQ_LENGTH=4096
MAX_STEPS=600
EXACT_PER_GRADE=220
WEAK_INTERVAL_ITEMS=120
CODEJUDGE_PAIRS=60
DROP_AXIOM_GRADE_ZERO=1
AXIOM_EXACT_FRACTION=0.7
```

注意：当前训练仍是 Transformers 单进程 LoRA + value head，`run_principle_generalization_eval.sh` 内部训练阶段只使用单卡；vLLM serve helper 才默认使用双卡 tensor parallel。8B 的 4096 token 训练是否能稳定跑完，需要先用小步 smoke 验证。

## CodeCritic -> AXIOM 对齐过夜实验

如果当前目标是先验证 `CodeCriticBench` 的 AXIOM 对齐程度，而不是继续混合多数据源，可以使用专门的过夜脚本：

```bash
tmux new-session -d -s codecritic_axiom_alignment_qwen3_8b_overnight_20260423 \
  /data1/xianzhiwei/mcts-code-review/data_collection/scripts/run_codecritic_axiom_alignment_qwen3_8b_overnight.sh
```

这条脚本会执行：

1. 仅用 `CodeCriticBench` 构造 8B 全上下文训练集
2. 默认丢弃 `AXIOM grade 0` 对齐档位，只保留 `1-5`
3. 训练 `Qwen/Qwen3-8B` LoRA + value head
4. 训练完成后自动跑 `AXIOM clean heldout` 外评

适合做“CodeCritic 映射是否真的对齐 AXIOM”这类外部语义验证。
