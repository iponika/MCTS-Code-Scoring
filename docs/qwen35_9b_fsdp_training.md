# Qwen3.5-9B 双卡 FSDP 训练方案

目标是在两张 4090 上训练 `Qwen/Qwen3.5-9B` 的 LoRA + value-head 版本，尽量把 `max_training_seq_length` 提高到 4096/6144/8192，减少短样本过滤和截断压力。

## 为什么不用普通双卡 DDP

普通 DDP 会在每张卡上各复制一份完整 9B 模型，不能降低单卡显存占用。当前需要的是把模型参数、梯度和优化器状态分片，因此优先使用 FSDP。

## 当前实现

脚本：

```bash
data_collection/scripts/run_qwen35_9b_fsdp_smoke.sh
```

默认行为：

- `MODEL_KEY=Qwen/Qwen3.5-9B`
- `MODEL_PATH=/data1/xianzhiwei/model/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- `MAX_TRAINING_SEQ_LENGTH=4096`
- `MAX_STEPS=1`
- `SKIP_SAVE=True`
- `SAVE_STRATEGY=no`
- `FSDP_OFFLOAD_PARAMS=false`，如果 24GB 显存仍不足，可以临时设为 `true` 验证链路，代价是明显变慢。
- `LORA_RANK=8`
- `LORA_ALPHA=16`
- `LORA_TARGET_SCOPE=attention`
- `FSDP_ACTIVATION_CHECKPOINTING=true`
- `FORCE_GRADIENT_CHECKPOINTING=False`
- 双卡 `accelerate launch --use_fsdp --num_processes 2`
- FSDP wrap class: `Qwen3_5DecoderLayer`
- LoRA + value head 仍复用 `magicoder.train_multi`
- 9B smoke 默认使用 FSDP activation checkpointing，并关闭 Trainer gradient checkpointing；其他训练脚本默认仍保持原 Trainer gradient checkpointing。
- 默认 `ACCELERATE_MIXED_PRECISION=no` 且 `TRAINING_BF16=False`，模型加载路径本身会把训练 wrapper 转成 bf16。这样避免 Accelerate/FSDP 在 prepare 阶段把 flat parameters upcast 到 fp32，后者在双 4090 上会接近满显存并 OOM。
- `RLTrainer` 对没有 LM labels 的 value-only batch 会使用 `logits_to_keep=1` 跳过完整 LM logits，避免 Qwen 大词表在 value-only 训练中浪费显存。

## Smoke 运行

```bash
tmux new-session -d -s qwen35_9b_fsdp_smoke \
  'cd /data1/xianzhiwei/mcts-code-review && bash data_collection/scripts/run_qwen35_9b_fsdp_smoke.sh'
```

查看进度：

```bash
tail -f data_collection/review_mcts_runs/qwen35_9b_fsdp_smoke_20260427/logs/train_*.log
```

如果 4096 通过，可以继续试：

```bash
MAX_TRAINING_SEQ_LENGTH=6144 MAX_STEPS=1 RUN_NAME=qwen35_9b_fsdp_6144_smoke_20260427 \
  bash data_collection/scripts/run_qwen35_9b_fsdp_smoke.sh
```

再试 8192：

```bash
MAX_TRAINING_SEQ_LENGTH=8192 MAX_STEPS=1 RUN_NAME=qwen35_9b_fsdp_8192_smoke_20260427 \
  bash data_collection/scripts/run_qwen35_9b_fsdp_smoke.sh
```

## 真实训练

smoke 通过后再开启保存和更多步数：

```bash
RUN_NAME=qwen35_9b_fsdp_mcts_train_20260427 \
MAX_TRAINING_SEQ_LENGTH=6144 \
MAX_STEPS=240 \
SAVE_STRATEGY=steps \
SKIP_SAVE=False \
bash data_collection/scripts/run_qwen35_9b_fsdp_smoke.sh
```

## 风险

- FSDP 能分片参数和训练状态，但长上下文 activation 仍会随序列长度增长。8192 不保证一定能跑。
- 这里优先选择 bf16 参数常驻而不是 FSDP fp32 master 参数，是为了适配 24GB 显存；稳定性需要后续用更长训练观察。
- 如果 `FSDP_OFFLOAD_PARAMS=true` 才能通过 smoke，说明 9B 在双 4090 上可训练但吞吐会受 CPU/GPU 传输限制；后续应再评估低 LoRA rank、DeepSpeed ZeRO-3 或更小 batch/更短 step 数据。
- 9B smoke 默认使用轻量 LoRA，而不是原先 `rank=64 + attention/MLP/额外投影全覆盖`。后者在双 4090 上即使 FSDP 也会在 backward 阶段 OOM。
- 当前 value-head wrapper 是 TRL 的 `AutoModelForCausalLMWithValueHead`，FSDP 保存 checkpoint 可能需要单独验证；smoke 默认 `SKIP_SAVE=True` 是为了先验证训练前向/反向。
- 如果 FSDP 在 value-head wrapper 上不稳定，下一备选是安装并使用 DeepSpeed ZeRO-3。
