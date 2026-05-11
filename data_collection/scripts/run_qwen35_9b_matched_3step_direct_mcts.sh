#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# Keep Qwen3.5 dependencies, caches, and model files outside /home.
# shellcheck disable=SC1091
source "${ROOT}/scripts/qwen35_9b_env.sh"

RUN_NAME="${RUN_NAME:-qwen35_9b_matched_3step_direct_mcts_$(date +%Y%m%d)}"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-qwen35_9b_base_static_direct_mcts_full_20260509}"
CFG="${CFG:-${ROOT}/data_collection/configs/mcts_code_review_qwen35_9b_no_think.yaml}"
MODEL_KEY="${MODEL_KEY:-Qwen/Qwen3.5-9B}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
CHAT_TEMPLATE_ENABLE_THINKING="${CHAT_TEMPLATE_ENABLE_THINKING:-false}"

REASONING_STEPS="${REASONING_STEPS:-3}"
DIRECT_REPEATS="${DIRECT_REPEATS:-3}"
DIRECT_BATCH_SIZE="${DIRECT_BATCH_SIZE:-4}"
POLICY_MIN_Q="${POLICY_MIN_Q:-0.8}"
POLICY_MAX_GRADE_DELTA="${POLICY_MAX_GRADE_DELTA:-1}"
WEAK_POLICY_LM_WEIGHT="${WEAK_POLICY_LM_WEIGHT:-0.35}"
MAX_VALUE_PATHS_PER_DIMENSION="${MAX_VALUE_PATHS_PER_DIMENSION:-0}"
BALANCED_TRAIN_TOTAL="${BALANCED_TRAIN_TOTAL:-800}"
MAX_PROBLEM_CHARS="${MAX_PROBLEM_CHARS:-0}"
MAX_CODE_CHARS="${MAX_CODE_CHARS:-0}"
MAX_TRAINING_SEQ_LENGTH="${MAX_TRAINING_SEQ_LENGTH:-8192}"
MAX_STEPS="${MAX_STEPS:-180}"
SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
SAVE_STEPS="${SAVE_STEPS:-$((MAX_STEPS + 1))}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
EVAL_PER_GRADE="${EVAL_PER_GRADE:-20}"
CHECK_GPUS_IDLE="${CHECK_GPUS_IDLE:-1}"
GEN_CUDA_VISIBLE_DEVICES="${GEN_CUDA_VISIBLE_DEVICES:-0,1}"
DIRECT_TRAIN_CUDA_VISIBLE_DEVICES="${DIRECT_TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
MCTS_TRAIN_CUDA_VISIBLE_DEVICES="${MCTS_TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
NTFY_URL="${NTFY_URL:-https://ntfy.sh/iponika_mcts}"
REUSE_SOURCE_MCTS_RAW="${REUSE_SOURCE_MCTS_RAW:-1}"

ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-no}"
TRAINING_BF16="${TRAINING_BF16:-False}"
DIRECT_FSDP_OFFLOAD_PARAMS="${DIRECT_FSDP_OFFLOAD_PARAMS:-false}"
MCTS_FSDP_OFFLOAD_PARAMS="${MCTS_FSDP_OFFLOAD_PARAMS:-true}"
FSDP_ACTIVATION_CHECKPOINTING="${FSDP_ACTIVATION_CHECKPOINTING:-true}"
FORCE_GRADIENT_CHECKPOINTING="${FORCE_GRADIENT_CHECKPOINTING:-False}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_SCOPE="${LORA_TARGET_SCOPE:-attention}"

RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
SOURCE_RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${SOURCE_RUN_NAME}"
SEED_DATA="${RUN_DIR}/seed_axiom.jsonl"
SEED_META="${RUN_DIR}/seed_axiom.metadata.json"
DIRECT_RAW="${RUN_DIR}/direct_stepwise_${REASONING_STEPS}step_bootstrap_raw.jsonl"
MCTS_RAW="${RUN_DIR}/mcts_bootstrap_raw.jsonl"
BALANCE_META="${RUN_DIR}/train_balance_summary.json"
SUMMARY_JSON="${RUN_DIR}/summary.json"

TRAIN_DIR="${ROOT}/model_training/review_mcts_train_data"
DIRECT_TRAIN_RAW="${TRAIN_DIR}/${RUN_NAME}_direct_3step_raw.jsonl"
MCTS_TRAIN_RAW="${TRAIN_DIR}/${RUN_NAME}_mcts_3step_raw.jsonl"
DIRECT_TRAIN="${TRAIN_DIR}/${RUN_NAME}_direct_3step.jsonl"
MCTS_TRAIN="${TRAIN_DIR}/${RUN_NAME}_mcts_3step.jsonl"

OUTPUT_ROOT="${ROOT}/model_training/src/output"
DIRECT_MODEL="${OUTPUT_ROOT}/review-lora-${RUN_NAME}-direct-3step-${MAX_STEPS}step"
MCTS_MODEL="${OUTPUT_ROOT}/review-lora-${RUN_NAME}-mcts-3step-${MAX_STEPS}step"

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

require_idle_gpus() {
  if [[ "${CHECK_GPUS_IDLE}" == "1" || "${CHECK_GPUS_IDLE,,}" == "true" ]]; then
    "${ROOT}/scripts/check_gpus_idle.sh"
  fi
}

count_train_json() {
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

prepare_reused_inputs() {
  CURRENT_STAGE="prepare_reused_inputs"
  if [[ ! -f "${SEED_DATA}" ]]; then
    cp "${SOURCE_RUN_DIR}/seed_axiom.jsonl" "${SEED_DATA}"
  fi
  if [[ ! -f "${SEED_META}" && -f "${SOURCE_RUN_DIR}/seed_axiom.metadata.json" ]]; then
    cp "${SOURCE_RUN_DIR}/seed_axiom.metadata.json" "${SEED_META}"
  fi
  if [[ "${REUSE_SOURCE_MCTS_RAW}" == "1" && ! -f "${MCTS_RAW}" ]]; then
    cp "${SOURCE_RUN_DIR}/mcts_bootstrap_raw.jsonl" "${MCTS_RAW}"
  fi
}

generate_direct_3step() {
  CURRENT_STAGE="generate_direct_3step"
  if [[ -f "${DIRECT_RAW}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${DIRECT_RAW}"
    return
  fi
  require_idle_gpus
  cd "${ROOT}"
  CUDA_VISIBLE_DEVICES="${GEN_CUDA_VISIBLE_DEVICES}" \
  PYTHONPATH="${ROOT}/data_collection" \
  HF_HUB_OFFLINE=1 \
  python data_collection/direct_bootstrap_review.py \
    --custom_cfg "${CFG}" \
    --dataset "${SEED_DATA}" \
    --start 0 \
    --limit "$(wc -l < "${SEED_DATA}" | tr -d ' ')" \
    --output "${DIRECT_RAW}" \
    --dimension "Correctness Verification" \
    --batch_size "${DIRECT_BATCH_SIZE}" \
    --repeats "${DIRECT_REPEATS}" \
    --response_mode stepwise \
    --reasoning_steps "${REASONING_STEPS}"
}

generate_mcts() {
  CURRENT_STAGE="generate_mcts"
  if [[ -f "${MCTS_RAW}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${MCTS_RAW}"
    return
  fi
  require_idle_gpus
  cd "${ROOT}"
  CUDA_VISIBLE_DEVICES="${GEN_CUDA_VISIBLE_DEVICES}" \
  PYTHONPATH="${ROOT}/data_collection" \
  HF_HUB_OFFLINE=1 \
  python data_collection/solver_review.py \
    --custom_cfg "${CFG}" \
    --dataset "${SEED_DATA}" \
    --start 0 \
    --limit "$(wc -l < "${SEED_DATA}" | tr -d ' ')" \
    --output "${MCTS_RAW}" \
    --output_dir "${RUN_DIR}/mcts_samples"
}

prepare_bootstrap_train() {
  local tag="$1"
  local raw_file="$2"
  local output_file="$3"
  CURRENT_STAGE="prepare_${tag}_train"
  if [[ -f "${output_file}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${output_file}"
    return
  fi
  cd "${ROOT}"
  PYTHONPATH="${ROOT}:${ROOT}/model_training/src:${ROOT}/data_collection" \
    python -m magicoder.preprocess_review_mcts_data \
      --input "${raw_file}" \
      --output_file "${output_file}" \
      --policy_min_q "${POLICY_MIN_Q}" \
      --policy_max_grade_delta "${POLICY_MAX_GRADE_DELTA}" \
      --weak_policy_lm_weight "${WEAK_POLICY_LM_WEIGHT}" \
      --policy_response_mode path \
      --max_value_paths_per_dimension "${MAX_VALUE_PATHS_PER_DIMENSION}" \
      --max_problem_chars "${MAX_PROBLEM_CHARS}" \
      --max_code_chars "${MAX_CODE_CHARS}"
}

balance_train_sets() {
  CURRENT_STAGE="balance_train_sets"
  if [[ -f "${DIRECT_TRAIN}" && -f "${MCTS_TRAIN}" && -f "${BALANCE_META}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists"
    return
  fi
  local direct_counts mcts_counts targets
  direct_counts="$(count_train_json "${DIRECT_TRAIN_RAW}")"
  mcts_counts="$(count_train_json "${MCTS_TRAIN_RAW}")"
  targets="$(
    python - <<PY
import json
direct = json.loads('''${direct_counts}''')
mcts = json.loads('''${mcts_counts}''')
total = int("${BALANCED_TRAIN_TOTAL}")
direct_policy = min(direct["policy"], total)
mcts_policy = min(mcts["policy"], total)
print(json.dumps({
    "direct_raw": direct,
    "mcts_raw": mcts,
    "target_total": total,
    "direct_target_policy": direct_policy,
    "mcts_target_policy": mcts_policy,
    "direct_target_value": max(0, total - direct_policy),
    "mcts_target_value": max(0, total - mcts_policy),
}, ensure_ascii=False))
PY
  )"
  echo "${targets}" | python -m json.tool > "${BALANCE_META}"
  local target_total direct_target_policy mcts_target_policy direct_target_value mcts_target_value
  target_total="$(python - <<PY
import json
print(json.loads('''${targets}''')["target_total"])
PY
)"
  direct_target_policy="$(python - <<PY
import json
print(json.loads('''${targets}''')["direct_target_policy"])
PY
)"
  mcts_target_policy="$(python - <<PY
import json
print(json.loads('''${targets}''')["mcts_target_policy"])
PY
)"
  direct_target_value="$(python - <<PY
import json
print(json.loads('''${targets}''')["direct_target_value"])
PY
)"
  mcts_target_value="$(python - <<PY
import json
print(json.loads('''${targets}''')["mcts_target_value"])
PY
)"
  cd "${ROOT}"
  PYTHONPATH="${ROOT}" python data_collection/rebalance_review_train_data.py \
    --input "${DIRECT_TRAIN_RAW}" \
    --output "${DIRECT_TRAIN}" \
    --target_policy_count "${direct_target_policy}" \
    --target_value_count "${direct_target_value}" \
    --target_total_count "${target_total}" \
    --stratify_by_dataset \
    --stratify_by_delta_bucket
  PYTHONPATH="${ROOT}" python data_collection/rebalance_review_train_data.py \
    --input "${MCTS_TRAIN_RAW}" \
    --output "${MCTS_TRAIN}" \
    --target_policy_count "${mcts_target_policy}" \
    --target_value_count "${mcts_target_value}" \
    --target_total_count "${target_total}" \
    --stratify_by_dataset \
    --stratify_by_delta_bucket
}

audit_lengths() {
  local train_data="$1"
  CURRENT_STAGE="audit_lengths"
  cd "${ROOT}"
  PYTHONPATH="${ROOT}/model_training/src" HF_HUB_OFFLINE=1 python - <<PY
import json
from pathlib import Path
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("${MODEL_PATH}", use_fast=True, trust_remote_code=True)
rows = [json.loads(line) for line in Path("${train_data}").read_text(encoding="utf-8").splitlines() if line.strip()]
lengths = []
for row in rows:
    if row.get("messages"):
        rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        lengths.append(len(tokenizer.encode(rendered, add_special_tokens=False)))
lengths.sort()
def pct(q):
    return lengths[min(len(lengths) - 1, int(round((len(lengths) - 1) * q)))] if lengths else 0
print(json.dumps({
    "train_data": "${train_data}",
    "examples": len(lengths),
    "max_training_seq_length": int("${MAX_TRAINING_SEQ_LENGTH}"),
    "fit_examples": sum(x <= int("${MAX_TRAINING_SEQ_LENGTH}") for x in lengths),
    "too_long_examples": sum(x > int("${MAX_TRAINING_SEQ_LENGTH}") for x in lengths),
    "p50": pct(0.5),
    "p90": pct(0.9),
    "p99": pct(0.99),
    "max": lengths[-1] if lengths else 0,
}, ensure_ascii=False, indent=2))
PY
}

train_fsdp() {
  local tag="$1"
  local train_data="$2"
  local output_model="$3"
  local cuda_devices="$4"
  local offload_params="$5"
  CURRENT_STAGE="train_${tag}"
  local log_file="${LOG_DIR}/train_${tag}.log"
  if [[ -f "${output_model}/value_head.pth" && ( -f "${output_model}/adapter_model.safetensors" || -f "${output_model}/adapter_model.bin" ) ]]; then
    echo "[stage:${CURRENT_STAGE}] final checkpoint exists: ${output_model}"
    return
  fi
  require_idle_gpus
  audit_lengths "${train_data}"
  cd "${ROOT}/model_training/src"
  local num_processes
  num_processes="$(python - <<PY
print(len([item for item in "${cuda_devices}".split(",") if item.strip()]))
PY
)"
  resume_args=()
  latest_checkpoint="$(find "${output_model}" -maxdepth 1 -type d -name 'checkpoint-*' -exec test -f '{}/trainer_state.json' ';' -print 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "${latest_checkpoint}" ]]; then
    resume_args=(--resume_from_checkpoint "${latest_checkpoint}")
  fi
  CUDA_VISIBLE_DEVICES="${cuda_devices}" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_HUB_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  TRL_EXPERIMENTAL_SILENCE=1 \
  accelerate launch \
    --num_processes "${num_processes}" \
    --gpu_ids "${cuda_devices}" \
    --mixed_precision "${ACCELERATE_MIXED_PRECISION}" \
    --use_fsdp \
    --fsdp_version 1 \
    --fsdp_offload_params "${offload_params}" \
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
      --logging_steps 10 \
      --save_strategy "${SAVE_STRATEGY}" \
      --save_steps "${SAVE_STEPS}" \
      --save_total_limit 2 \
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
      --seed 20260510 \
      --skip_save False \
      "${resume_args[@]}" >"${log_file}" 2>&1
}

eval_stepwise_model() {
  local tag="$1"
  local trained_model="$2"
  CURRENT_STAGE="eval_${tag}"
  require_idle_gpus
  RUN_NAME="${RUN_NAME}_${tag}_axiom_eval" \
  BASE_MODEL_PATH="${MODEL_PATH}" \
  TRAINED_MODEL_PATH="${trained_model}" \
  TRAIN_DATA="${SEED_DATA}" \
  PER_GRADE="${EVAL_PER_GRADE}" \
  MAX_PROBLEM_CHARS="${MAX_PROBLEM_CHARS}" \
  MAX_CODE_CHARS="${MAX_CODE_CHARS}" \
  DROP_AXIOM_GRADE_ZERO=1 \
  FINAL_ONLY_JSON=0 \
  EVAL_BASE_DIRECT=0 \
  EVAL_TRAINED_DIRECT=1 \
  EVAL_TRAINED_VALUE=1 \
  TRAINED_DIRECT_MAX_STEPS="${REASONING_STEPS}" \
  TRAINED_VALUE_MAX_STEPS="${REASONING_STEPS}" \
  TRAINED_DIRECT_NUM_CANDIDATES=1 \
  TRAINED_VALUE_NUM_CANDIDATES=4 \
  REASONING_STEPS="${REASONING_STEPS}" \
  MAX_RETHINKS=0 \
  USE_CHAT_TEMPLATE=1 \
  USE_UV=0 \
  PROMPT_VARIANT=default \
  CHAT_TEMPLATE_ENABLE_THINKING="${CHAT_TEMPLATE_ENABLE_THINKING}" \
  EVAL_CUDA_DEVICE="${EVAL_CUDA_VISIBLE_DEVICES}" \
  NTFY_URL="${NTFY_URL}" \
  "${ROOT}/data_collection/scripts/run_axiom_clean_eval.sh"
}

write_summary() {
  CURRENT_STAGE="write_summary"
  python - <<PY
import json
from pathlib import Path

root = Path("${ROOT}")
run_name = "${RUN_NAME}"

def load_eval(tag, key):
    path = root / "model_training" / "src" / "output" / f"review-eval-{run_name}_{tag}_axiom_eval" / "comparison.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get(key)

payload = {
    "run_name": run_name,
    "source_run_name": "${SOURCE_RUN_NAME}",
    "model_key": "${MODEL_KEY}",
    "reasoning_steps": int("${REASONING_STEPS}"),
    "seed_data": "${SEED_DATA}",
    "raw_data": {
        "direct_3step": "${DIRECT_RAW}",
        "mcts": "${MCTS_RAW}",
    },
    "train_data": {
        "direct_3step": "${DIRECT_TRAIN}",
        "mcts_3step": "${MCTS_TRAIN}",
    },
    "models": {
        "direct_3step": "${DIRECT_MODEL}",
        "mcts_3step": "${MCTS_MODEL}",
    },
    "balance": json.loads(Path("${BALANCE_META}").read_text(encoding="utf-8")) if Path("${BALANCE_META}").exists() else None,
    "evaluations": {
        "direct_3step_direct": load_eval("direct_3step", "trained_direct_clean"),
        "direct_3step_value": load_eval("direct_3step", "trained_value_rerank_clean"),
        "mcts_3step_direct": load_eval("mcts_3step", "trained_direct_clean"),
        "mcts_3step_value": load_eval("mcts_3step", "trained_value_rerank_clean"),
    },
}
Path("${SUMMARY_JSON}").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}

main() {
  echo "[start] run=${RUN_NAME} source=${SOURCE_RUN_NAME} reasoning_steps=${REASONING_STEPS} time=$(date -Is) log=${LOG_FILE}"
  prepare_reused_inputs
  generate_direct_3step
  generate_mcts
  prepare_bootstrap_train "direct_3step" "${DIRECT_RAW}" "${DIRECT_TRAIN_RAW}"
  prepare_bootstrap_train "mcts_3step" "${MCTS_RAW}" "${MCTS_TRAIN_RAW}"
  balance_train_sets

  if [[ "${STOP_AFTER_DATA:-0}" == "1" ]]; then
    CURRENT_STAGE="finished"
    echo "[finished] STOP_AFTER_DATA=1 run_dir=${RUN_DIR}"
    exit 0
  fi

  train_fsdp "direct_3step" "${DIRECT_TRAIN}" "${DIRECT_MODEL}" "${DIRECT_TRAIN_CUDA_VISIBLE_DEVICES}" "${DIRECT_FSDP_OFFLOAD_PARAMS}"
  train_fsdp "mcts_3step" "${MCTS_TRAIN}" "${MCTS_MODEL}" "${MCTS_TRAIN_CUDA_VISIBLE_DEVICES}" "${MCTS_FSDP_OFFLOAD_PARAMS}"

  if [[ "${STOP_AFTER_TRAIN:-0}" == "1" ]]; then
    CURRENT_STAGE="finished"
    echo "[finished] STOP_AFTER_TRAIN=1 run_dir=${RUN_DIR}"
    exit 0
  fi

  eval_stepwise_model "direct_3step" "${DIRECT_MODEL}"
  eval_stepwise_model "mcts_3step" "${MCTS_MODEL}"
  write_summary
  CURRENT_STAGE="finished"
  notify "FINISHED"
  echo "[finished] summary=${SUMMARY_JSON}"
}

main "$@"
