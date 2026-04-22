# Qwen3-4B-Instruct-2507 小模型适配

## 选择理由

默认小模型切换为 `Qwen/Qwen3-4B-Instruct-2507`。它是 2025 年发布的 Qwen3 4B instruct 刷新版，参数量约 4B，用于替代 9B 训练时的显存权宜策略，让主流程可以保留完整 review prompt 和 pairwise value ranking。

需要注意：该模型是 non-thinking instruct 模型，不会原生输出 `<think></think>`。因此它更适合当前以 `<review>` 标量评分为主的训练和 value-guided rerank；如果后续要继续把“原生深度思考片段”作为 MCTS 节点，应另行评估 thinking 版本。

## 下载

模型使用现有 Hugging Face 缓存目录：

```bash
export HF_HOME=/data1/xianzhiwei/model/huggingface
hf download Qwen/Qwen3-4B-Instruct-2507
```

## vLLM 服务

单卡启动：

```bash
tmux new-session -d -s qwen3_4b_vllm \
  /data1/xianzhiwei/mcts-code-review/data_collection/scripts/serve_qwen3_4b_instruct_2507.sh
```

默认参数：

```bash
CUDA_VISIBLE_DEVICES=0
MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
VLLM_HOST=0.0.0.0
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.85
PORT=8000
```

如果当前 shell 设置了 `http_proxy`，本机请求需要绕过代理：

```bash
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

## 训练/评测 workflow

小模型版 full-context workflow：

```bash
tmux new-session -d -s principle_generalization_qwen3_4b_full_context_no_axiom0_20260422 \
  /data1/xianzhiwei/mcts-code-review/data_collection/scripts/run_principle_generalization_qwen3_4b.sh
```

该脚本继承 `run_principle_generalization_eval.sh`，但默认：

```bash
MODEL_KEY=Qwen/Qwen3-4B-Instruct-2507
MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
MAX_TRAINING_SEQ_LENGTH=3072
MAX_STEPS=600
EXACT_PER_GRADE=180
WEAK_INTERVAL_ITEMS=90
CODEJUDGE_PAIRS=45
DROP_AXIOM_GRADE_ZERO=1
```

主流程使用完整 `QWEN_REVIEW_STEP_PROMPT`，不再使用短训练 prompt；`CodeJudgeBench` 正负样本对保持相邻，并启用 batch-local pairwise value ranking。AXIOM 数据源默认剔除 0 分样本，因为这部分样本经数据集开发者反馈质量较离谱；CodeCritic 映射出的 0 分样本仍保留。若 3072 token OOM，优先确认是否必须回退到 2048，而不是重新引入短 prompt 或截断 instruction。
