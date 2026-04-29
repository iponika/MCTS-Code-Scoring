#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUN_NAME="${RUN_NAME:-qwen35_9b_fsdp_smoke_20260427}"
MODEL_KEY="${MODEL_KEY:-Qwen/Qwen3.5-9B}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
TRAIN_DATA="${TRAIN_DATA:-}"
OUTPUT_MODEL="${OUTPUT_MODEL:-${ROOT}/model_training/src/output/review-lora-${RUN_NAME}}"
MAX_TRAINING_SEQ_LENGTH="${MAX_TRAINING_SEQ_LENGTH:-4096}"
MAX_STEPS="${MAX_STEPS:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
SKIP_SAVE="${SKIP_SAVE:-True}"
ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-no}"
TRAINING_BF16="${TRAINING_BF16:-False}"
FSDP_OFFLOAD_PARAMS="${FSDP_OFFLOAD_PARAMS:-false}"
FSDP_ACTIVATION_CHECKPOINTING="${FSDP_ACTIVATION_CHECKPOINTING:-true}"
FORCE_GRADIENT_CHECKPOINTING="${FORCE_GRADIENT_CHECKPOINTING:-False}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_SCOPE="${LORA_TARGET_SCOPE:-attention}"
NTFY_URL="${NTFY_URL:-https://ntfy.sh/iponika_mcts}"

RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_MODEL}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE="init"

notify() {
  local status="$1"
  local message="${RUN_NAME} ${status}: stage=${CURRENT_STAGE}, host=$(hostname), time=$(date -Is), log=${LOG_FILE}, output=${OUTPUT_MODEL}"
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    curl --noproxy '*' --connect-timeout 5 --max-time 20 -fsS -d "${message}" "${NTFY_URL}" >/dev/null 2>&1 || true
}

on_exit() {
  local code=$?
  if [[ "${CURRENT_STAGE}" != "finished" ]]; then
    notify "FAILED(code=${code})"
  fi
  exit "${code}"
}
trap on_exit EXIT

audit_lengths() {
  CURRENT_STAGE="audit_lengths"
  cd "${ROOT}"
  PYTHONPATH="${ROOT}/model_training/src" \
  UV_CACHE_DIR=/tmp/uv-cache \
  HF_HUB_OFFLINE=1 \
  uv run python - <<PY
import json
from pathlib import Path

from magicoder.prompt_template import QWEN_REVIEW_STEP_PROMPT
from transformers import AutoTokenizer

path = Path("${TRAIN_DATA}")
max_len = int("${MAX_TRAINING_SEQ_LENGTH}")
tokenizer = AutoTokenizer.from_pretrained("${MODEL_PATH}", use_fast=True)
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

lengths = []
for row in rows:
    prompt = QWEN_REVIEW_STEP_PROMPT.format(instruction=row.get("instruction", ""), response="")
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    for response in row.get("response", []):
        ids += tokenizer.encode(str(response).strip() + "\\n", add_special_tokens=False)
    ids += [tokenizer.eos_token_id]
    lengths.append(len(ids))

lengths.sort()
def pct(q):
    if not lengths:
        return 0
    idx = min(len(lengths) - 1, int(round((len(lengths) - 1) * q)))
    return lengths[idx]

print(json.dumps({
    "train_data": str(path),
    "examples": len(lengths),
    "max_training_seq_length": max_len,
    "fit_examples": sum(length <= max_len for length in lengths),
    "too_long_examples": sum(length > max_len for length in lengths),
    "min": lengths[0] if lengths else 0,
    "p50": pct(0.50),
    "p90": pct(0.90),
    "p95": pct(0.95),
    "p99": pct(0.99),
    "max": lengths[-1] if lengths else 0,
}, ensure_ascii=False, indent=2))
PY
}

train_fsdp() {
  CURRENT_STAGE="train_fsdp"
  cd "${ROOT}/model_training/src"

  resume_args=()
  latest_checkpoint="$(find "${OUTPUT_MODEL}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "${latest_checkpoint}" ]]; then
    resume_args=(--resume_from_checkpoint "${latest_checkpoint}")
  fi

  CUDA_VISIBLE_DEVICES=0,1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_DATASETS_CACHE=/tmp/hf-datasets-cache \
  HF_HUB_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  TRL_EXPERIMENTAL_SILENCE=1 \
  UV_CACHE_DIR=/tmp/uv-cache \
  accelerate launch \
    --num_processes 2 \
    --gpu_ids 0,1 \
    --mixed_precision "${ACCELERATE_MIXED_PRECISION}" \
    --use_fsdp \
    --fsdp_version 1 \
    --fsdp_offload_params "${FSDP_OFFLOAD_PARAMS}" \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap Qwen3_5DecoderLayer \
    --fsdp_backward_prefetch BACKWARD_PRE \
    --fsdp_state_dict_type FULL_STATE_DICT \
    --fsdp_use_orig_params true \
    --fsdp_activation_checkpointing "${FSDP_ACTIVATION_CHECKPOINTING}" \
    -m magicoder.train_multi \
      --task review \
      --model_key "${MODEL_KEY}" \
      --model_name_or_path "${MODEL_PATH}" \
      --lora_rank "${LORA_RANK}" \
      --lora_alpha "${LORA_ALPHA}" \
      --lora_dropout "${LORA_DROPOUT}" \
      --lora_target_scope "${LORA_TARGET_SCOPE}" \
      --datafile_paths "${TRAIN_DATA}" \
      --output_dir "${OUTPUT_MODEL}" \
      --max_steps "${MAX_STEPS}" \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
      --max_training_seq_length "${MAX_TRAINING_SEQ_LENGTH}" \
      --bf16 "${TRAINING_BF16}" \
      --logging_steps 1 \
      --save_strategy "${SAVE_STRATEGY}" \
      --report_to none \
      --optim adafactor \
      --learning_rate 2e-5 \
      --lr_scheduler_type constant \
      --peft lora \
      --force_gradient_checkpointing "${FORCE_GRADIENT_CHECKPOINTING}" \
      --value_weight 0.05 \
      --boundary_value_weight 0.02 \
      --pairwise_value_weight 0.0 \
      --disable_train_shuffle True \
      --train_sampling_strategy sequential \
      --num_proc 1 \
      --seed 20260427 \
      --skip_save "${SKIP_SAVE}" \
      "${resume_args[@]}"
}

echo "[run] ${RUN_NAME}"
echo "[log] ${LOG_FILE}"
echo "[model] ${MODEL_PATH}"
echo "[train_data] ${TRAIN_DATA}"
if [[ -z "${TRAIN_DATA}" || ! -f "${TRAIN_DATA}" ]]; then
  echo "TRAIN_DATA must point to an existing review training JSONL file." >&2
  exit 2
fi
audit_lengths
train_fsdp
CURRENT_STAGE="finished"
notify "FINISHED"
