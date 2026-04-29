#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUN_NAME="${RUN_NAME:-qwen35_9b_direct_stepwise_vs_review_smoke_20260428}"
CFG="${CFG:-${ROOT}/data_collection/configs/mcts_code_review.yaml}"
MODEL_KEY="${MODEL_KEY:-Qwen/Qwen3.5-9B}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"

SEED_PER_GRADE="${SEED_PER_GRADE:-4}"
DIRECT_REPEATS="${DIRECT_REPEATS:-2}"
DIRECT_BOOTSTRAP_REASONING_STEPS="${DIRECT_BOOTSTRAP_REASONING_STEPS:-3}"
DIRECT_BATCH_SIZE="${DIRECT_BATCH_SIZE:-2}"
POLICY_MIN_Q="${POLICY_MIN_Q:-0.8}"
MAX_TRAINING_SEQ_LENGTH="${MAX_TRAINING_SEQ_LENGTH:-2048}"
MAX_STEPS="${MAX_STEPS:-40}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
EVAL_PER_GRADE="${EVAL_PER_GRADE:-3}"
STEPWISE_MAX_STEPS="${STEPWISE_MAX_STEPS:-3}"
STEPWISE_NUM_CANDIDATES="${STEPWISE_NUM_CANDIDATES:-2}"
STEPWISE_MAX_RETHINKS="${STEPWISE_MAX_RETHINKS:-1}"
STEPWISE_RETHINK_THRESHOLD="${STEPWISE_RETHINK_THRESHOLD:--0.2}"
STEPWISE_RETHINK_SPREAD_THRESHOLD="${STEPWISE_RETHINK_SPREAD_THRESHOLD:-0.0}"
MAX_PROBLEM_CHARS="${MAX_PROBLEM_CHARS:-5000}"
MAX_CODE_CHARS="${MAX_CODE_CHARS:-5000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
FINAL_MAX_NEW_TOKENS="${FINAL_MAX_NEW_TOKENS:-256}"
NTFY_URL="${NTFY_URL:-https://ntfy.sh/iponika_mcts}"

ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-no}"
TRAINING_BF16="${TRAINING_BF16:-False}"
FSDP_OFFLOAD_PARAMS="${FSDP_OFFLOAD_PARAMS:-false}"
FSDP_ACTIVATION_CHECKPOINTING="${FSDP_ACTIVATION_CHECKPOINTING:-true}"
FORCE_GRADIENT_CHECKPOINTING="${FORCE_GRADIENT_CHECKPOINTING:-False}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_SCOPE="${LORA_TARGET_SCOPE:-attention}"

RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
SEED_DATA="${RUN_DIR}/seed_codecritic_axiom.jsonl"
SEED_META="${RUN_DIR}/seed_codecritic_axiom.metadata.json"
REVIEW_RAW="${RUN_DIR}/direct_review_bootstrap_raw.jsonl"
STEPWISE_RAW="${RUN_DIR}/direct_stepwise_bootstrap_raw.jsonl"

TRAIN_DIR="${ROOT}/model_training/review_mcts_train_data"
REVIEW_TRAIN_RAW="${TRAIN_DIR}/${RUN_NAME}_direct_review_raw.jsonl"
STEPWISE_TRAIN_RAW="${TRAIN_DIR}/${RUN_NAME}_direct_stepwise_raw.jsonl"
REVIEW_TRAIN="${TRAIN_DIR}/${RUN_NAME}_direct_review.jsonl"
STEPWISE_TRAIN="${TRAIN_DIR}/${RUN_NAME}_direct_stepwise.jsonl"
BALANCE_META="${RUN_DIR}/train_balance_summary.json"
SUMMARY_JSON="${RUN_DIR}/summary.json"

OUTPUT_ROOT="${ROOT}/model_training/src/output"
REVIEW_MODEL="${OUTPUT_ROOT}/review-lora-${RUN_NAME}-direct-review-${MAX_STEPS}step"
STEPWISE_MODEL="${OUTPUT_ROOT}/review-lora-${RUN_NAME}-direct-stepwise-${MAX_STEPS}step"

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${TRAIN_DIR}"
LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE="init"

notify() {
  local status="$1"
  local message="${RUN_NAME} ${status}: stage=${CURRENT_STAGE}, host=$(hostname), time=$(date -Is), log=${LOG_FILE}, run_dir=${RUN_DIR}"
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

prepare_seed_data() {
  CURRENT_STAGE="prepare_seed_data"
  if [[ -f "${SEED_DATA}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${SEED_DATA}"
    return
  fi
  cd "${ROOT}"
  PYTHONPATH="${ROOT}/data_collection" UV_CACHE_DIR=/tmp/uv-cache \
    uv run python data_collection/prepare_codecritic_axiom_seedset.py \
      --output "${SEED_DATA}" \
      --metadata "${SEED_META}" \
      --per_grade "${SEED_PER_GRADE}" \
      --min_grade 1 \
      --max_grade 5 \
      --max_objective_assertions_per_split 4 \
      --assertion_timeout_seconds 1.0 \
      --progress_every 100
}

generate_direct() {
  local mode="$1"
  local output="$2"
  CURRENT_STAGE="generate_${mode}"
  if [[ -f "${output}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${output}"
    return
  fi
  cd "${ROOT}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  PYTHONPATH="${ROOT}/data_collection" \
  UV_CACHE_DIR=/tmp/uv-cache \
  HF_HUB_OFFLINE=1 \
  uv run python data_collection/direct_bootstrap_review.py \
    --custom_cfg "${CFG}" \
    --dataset "${SEED_DATA}" \
    --start 0 \
    --limit "$(wc -l < "${SEED_DATA}" | tr -d ' ')" \
    --output "${output}" \
    --dimension "Correctness Verification" \
    --batch_size "${DIRECT_BATCH_SIZE}" \
    --repeats "${DIRECT_REPEATS}" \
    --response_mode "${mode}" \
    --reasoning_steps "${DIRECT_BOOTSTRAP_REASONING_STEPS}"
}

prepare_train() {
  local mode="$1"
  local raw="$2"
  local out="$3"
  CURRENT_STAGE="prepare_${mode}_train"
  if [[ -f "${out}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${out}"
    return
  fi
  cd "${ROOT}"
  PYTHONPATH="${ROOT}:${ROOT}/model_training/src:${ROOT}/data_collection" UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.preprocess_review_mcts_data \
      --input "${raw}" \
      --output_file "${out}" \
      --policy_min_q "${POLICY_MIN_Q}" \
      --policy_response_mode path \
      --max_value_paths_per_dimension 0
}

count_train() {
  local path="$1"
  python - <<PY
import json
from pathlib import Path
rows = [json.loads(line) for line in Path("${path}").read_text(encoding="utf-8").splitlines() if line.strip()]
print(json.dumps({
    "total": len(rows),
    "policy": sum(1 for row in rows if row.get("train_lm")),
    "value": sum(1 for row in rows if not row.get("train_lm")),
}))
PY
}

balance_train_sets() {
  CURRENT_STAGE="balance_train_sets"
  if [[ -f "${REVIEW_TRAIN}" && -f "${STEPWISE_TRAIN}" && -f "${BALANCE_META}" ]]; then
    echo "[stage:${CURRENT_STAGE}] balanced train files exist"
    return
  fi
  local review_counts
  local stepwise_counts
  review_counts="$(count_train "${REVIEW_TRAIN_RAW}")"
  stepwise_counts="$(count_train "${STEPWISE_TRAIN_RAW}")"
  local target_policy
  local target_value
  local target_total
  target_policy="$(python - <<PY
import json
print(min(json.loads('''${review_counts}''')["policy"], json.loads('''${stepwise_counts}''')["policy"]))
PY
)"
  target_value="$(python - <<PY
import json
print(min(json.loads('''${review_counts}''')["value"], json.loads('''${stepwise_counts}''')["value"]))
PY
)"
  target_total=$((target_policy + target_value))
  python - <<PY > "${BALANCE_META}"
import json
print(json.dumps({
    "direct_review_raw": json.loads('''${review_counts}'''),
    "direct_stepwise_raw": json.loads('''${stepwise_counts}'''),
    "target_policy": int("${target_policy}"),
    "target_value": int("${target_value}"),
    "target_total": int("${target_total}"),
}, indent=2))
PY
  cd "${ROOT}"
  for input in "${REVIEW_TRAIN_RAW}" "${STEPWISE_TRAIN_RAW}"; do
    if [[ "${input}" == "${REVIEW_TRAIN_RAW}" ]]; then
      output="${REVIEW_TRAIN}"
    else
      output="${STEPWISE_TRAIN}"
    fi
    PYTHONPATH="${ROOT}" UV_CACHE_DIR=/tmp/uv-cache \
      uv run python data_collection/rebalance_review_train_data.py \
        --input "${input}" \
        --output "${output}" \
        --target_policy_count "${target_policy}" \
        --target_value_count "${target_value}" \
        --target_total_count "${target_total}" \
        --stratify_by_dataset \
        --stratify_by_delta_bucket
  done
}

audit_lengths() {
  local train_data="$1"
  CURRENT_STAGE="audit_lengths"
  cd "${ROOT}"
  PYTHONPATH="${ROOT}/model_training/src" UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 \
    uv run python - <<PY
import json
from pathlib import Path
from transformers import AutoTokenizer
from magicoder.prompt_template import QWEN_REVIEW_STEP_PROMPT
tokenizer = AutoTokenizer.from_pretrained("${MODEL_PATH}", use_fast=True)
path = Path("${train_data}")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
lengths = []
for row in rows:
    ids = tokenizer.encode(QWEN_REVIEW_STEP_PROMPT.format(instruction=row.get("instruction", ""), response=""), add_special_tokens=True)
    for segment in row.get("response", []):
        ids += tokenizer.encode(str(segment).strip() + "\\n", add_special_tokens=False)
    ids += [tokenizer.eos_token_id]
    lengths.append(len(ids))
lengths.sort()
def pct(q):
    if not lengths:
        return 0
    return lengths[min(len(lengths) - 1, int(round((len(lengths) - 1) * q)))]
print(json.dumps({
    "train_data": str(path),
    "examples": len(lengths),
    "max_training_seq_length": int("${MAX_TRAINING_SEQ_LENGTH}"),
    "fit_examples": sum(x <= int("${MAX_TRAINING_SEQ_LENGTH}") for x in lengths),
    "too_long_examples": sum(x > int("${MAX_TRAINING_SEQ_LENGTH}") for x in lengths),
    "min": lengths[0] if lengths else 0,
    "p50": pct(0.5),
    "p90": pct(0.9),
    "max": lengths[-1] if lengths else 0,
}, ensure_ascii=False, indent=2))
PY
}

train_fsdp() {
  local tag="$1"
  local train_data="$2"
  local output_model="$3"
  CURRENT_STAGE="train_${tag}"
  local log_file="${LOG_DIR}/train_${tag}.log"
  if [[ -f "${output_model}/adapter_model.safetensors" && -f "${output_model}/value_head.pth" ]]; then
    echo "[stage:${CURRENT_STAGE}] final checkpoint exists: ${output_model}"
    return
  fi
  audit_lengths "${train_data}"
  cd "${ROOT}/model_training/src"
  resume_args=()
  latest_checkpoint="$(find "${output_model}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
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
      --datafile_paths "../review_mcts_train_data/$(basename "${train_data}")" \
      --output_dir "${output_model}" \
      --max_steps "${MAX_STEPS}" \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
      --max_training_seq_length "${MAX_TRAINING_SEQ_LENGTH}" \
      --bf16 "${TRAINING_BF16}" \
      --logging_steps 5 \
      --save_strategy no \
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
      --seed 20260428 \
      --skip_save False \
      "${resume_args[@]}" >"${log_file}" 2>&1
}

eval_model() {
  local tag="$1"
  local model="$2"
  CURRENT_STAGE="eval_${tag}"
  RUN_NAME="${RUN_NAME}_${tag}_axiom_eval" \
  BASE_MODEL_PATH="${MODEL_PATH}" \
  TRAINED_MODEL_PATH="${model}" \
  TRAIN_DATA="${SEED_DATA}" \
  PER_GRADE="${EVAL_PER_GRADE}" \
  DROP_AXIOM_GRADE_ZERO=1 \
  FINAL_ONLY_JSON=0 \
  EVAL_BASE_DIRECT=0 \
  TRAINED_VALUE_MAX_STEPS="${STEPWISE_MAX_STEPS}" \
  TRAINED_VALUE_NUM_CANDIDATES="${STEPWISE_NUM_CANDIDATES}" \
  MAX_RETHINKS="${STEPWISE_MAX_RETHINKS}" \
  RETHINK_THRESHOLD="${STEPWISE_RETHINK_THRESHOLD}" \
  RETHINK_SPREAD_THRESHOLD="${STEPWISE_RETHINK_SPREAD_THRESHOLD}" \
  MAX_PROBLEM_CHARS="${MAX_PROBLEM_CHARS}" \
  MAX_CODE_CHARS="${MAX_CODE_CHARS}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  FINAL_MAX_NEW_TOKENS="${FINAL_MAX_NEW_TOKENS}" \
  NTFY_URL="${NTFY_URL}" \
  "${ROOT}/data_collection/scripts/run_axiom_clean_eval.sh"
}

write_summary() {
  CURRENT_STAGE="write_summary"
  python - <<PY
import json
from pathlib import Path
root = Path("${ROOT}")
run = "${RUN_NAME}"
def load_eval(tag):
    path = root / "model_training" / "src" / "output" / f"review-eval-{run}_{tag}_axiom_eval" / "comparison.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
payload = {
    "run_name": run,
    "model_key": "${MODEL_KEY}",
    "model_path": "${MODEL_PATH}",
    "seed_data": "${SEED_DATA}",
    "direct_review_train": "${REVIEW_TRAIN}",
    "direct_stepwise_train": "${STEPWISE_TRAIN}",
    "review_model": "${REVIEW_MODEL}",
    "stepwise_model": "${STEPWISE_MODEL}",
    "max_steps": int("${MAX_STEPS}"),
    "max_training_seq_length": int("${MAX_TRAINING_SEQ_LENGTH}"),
    "direct_repeats": int("${DIRECT_REPEATS}"),
    "reasoning_steps": int("${DIRECT_BOOTSTRAP_REASONING_STEPS}"),
    "balance": json.loads(Path("${BALANCE_META}").read_text(encoding="utf-8")) if Path("${BALANCE_META}").exists() else None,
    "direct_review_eval": load_eval("direct_review"),
    "direct_stepwise_eval": load_eval("direct_stepwise"),
}
Path("${SUMMARY_JSON}").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}

main() {
  echo "[start] run=${RUN_NAME} time=$(date -Is) log=${LOG_FILE}"
  prepare_seed_data
  generate_direct "review" "${REVIEW_RAW}"
  generate_direct "stepwise" "${STEPWISE_RAW}"
  prepare_train "review" "${REVIEW_RAW}" "${REVIEW_TRAIN_RAW}"
  prepare_train "stepwise" "${STEPWISE_RAW}" "${STEPWISE_TRAIN_RAW}"
  balance_train_sets
  if [[ "${STOP_AFTER_DATA:-0}" == "1" ]]; then
    CURRENT_STAGE="finished"
    echo "[finished] STOP_AFTER_DATA=1 run_dir=${RUN_DIR}"
    exit 0
  fi
  train_fsdp "direct_review" "${REVIEW_TRAIN}" "${REVIEW_MODEL}"
  train_fsdp "direct_stepwise" "${STEPWISE_TRAIN}" "${STEPWISE_MODEL}"
  if [[ "${STOP_AFTER_TRAIN:-0}" == "1" ]]; then
    CURRENT_STAGE="finished"
    echo "[finished] STOP_AFTER_TRAIN=1"
    exit 0
  fi
  eval_model "direct_review" "${REVIEW_MODEL}"
  eval_model "direct_stepwise" "${STEPWISE_MODEL}"
  write_summary
  CURRENT_STAGE="finished"
  notify "FINISHED"
  echo "[finished] summary=${SUMMARY_JSON}"
}

main "$@"
