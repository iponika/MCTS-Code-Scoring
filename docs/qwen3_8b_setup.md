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
