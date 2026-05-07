#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# Keep the Qwen3.5/vLLM stack isolated from the shared conda/home setup.
# shellcheck disable=SC1091
source "${ROOT}/scripts/qwen35_9b_env.sh"

EXPERIMENT_THINKING_MODE="${EXPERIMENT_THINKING_MODE:-no_think}"
if [[ "${EXPERIMENT_THINKING_MODE}" == "think" || "${EXPERIMENT_THINKING_MODE}" == "thinking" ]]; then
  EXPERIMENT_THINKING_MODE="think"
  DEFAULT_CFG="${ROOT}/data_collection/configs/mcts_code_review_qwen35_9b_thinking.yaml"
  DEFAULT_STEP_COUNTS="1"
  DEFAULT_RUN_STATIC="0"
  DEFAULT_RUN_NAME="direct_stepcount_axiom_qwen35_9b_think_20260507"
  DEFAULT_CHAT_TEMPLATE_ENABLE_THINKING="true"
else
  EXPERIMENT_THINKING_MODE="no_think"
  DEFAULT_CFG="${ROOT}/data_collection/configs/mcts_code_review_qwen35_9b_no_think.yaml"
  DEFAULT_STEP_COUNTS="1,2,3"
  DEFAULT_RUN_STATIC="1"
  DEFAULT_RUN_NAME="direct_stepcount_axiom_qwen35_9b_nothink_20260507"
  DEFAULT_CHAT_TEMPLATE_ENABLE_THINKING="false"
fi

RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
CFG="${CFG:-${DEFAULT_CFG}}"
MODEL_KEY="${MODEL_KEY:-Qwen/Qwen3.5-9B}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
STEP_COUNTS="${STEP_COUNTS:-${DEFAULT_STEP_COUNTS}}"
RUN_STATIC="${RUN_STATIC:-${DEFAULT_RUN_STATIC}}"
CHAT_TEMPLATE_ENABLE_THINKING="${CHAT_TEMPLATE_ENABLE_THINKING:-${DEFAULT_CHAT_TEMPLATE_ENABLE_THINKING}}"

AXIOM_PER_GRADE="${AXIOM_PER_GRADE:-30}"
AXIOM_MIN_GRADE="${AXIOM_MIN_GRADE:-1}"
AXIOM_MAX_GRADE="${AXIOM_MAX_GRADE:-5}"
DIRECT_REPEATS="${DIRECT_REPEATS:-4}"
DIRECT_BATCH_SIZE="${DIRECT_BATCH_SIZE:-2}"
FREEFORM_FINAL_REVIEW="${FREEFORM_FINAL_REVIEW:-0}"
POLICY_MIN_Q="${POLICY_MIN_Q:-0.8}"
AXIOM_MAX_CODE_CHARS="${AXIOM_MAX_CODE_CHARS:-0}"
MAX_PROBLEM_CHARS="${MAX_PROBLEM_CHARS:-0}"
MAX_CODE_CHARS="${MAX_CODE_CHARS:-0}"
MAX_TRAINING_SEQ_LENGTH="${MAX_TRAINING_SEQ_LENGTH:-8192}"
MAX_STEPS="${MAX_STEPS:-120}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
EVAL_PER_GRADE="${EVAL_PER_GRADE:-10}"
CHECK_GPUS_IDLE="${CHECK_GPUS_IDLE:-1}"
GEN_CUDA_VISIBLE_DEVICES="${GEN_CUDA_VISIBLE_DEVICES:-0,1}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
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
SEED_DATA="${RUN_DIR}/seed_axiom.jsonl"
SEED_META="${RUN_DIR}/seed_axiom.metadata.json"
DIRECT_REVIEW_RAW="${RUN_DIR}/direct_review_bootstrap_raw.jsonl"

TRAIN_DIR="${ROOT}/model_training/review_mcts_train_data"
STATIC_TRAIN_RAW="${TRAIN_DIR}/${RUN_NAME}_static_raw.jsonl"
STATIC_TRAIN="${TRAIN_DIR}/${RUN_NAME}_static.jsonl"
DIRECT_REVIEW_TRAIN_RAW="${TRAIN_DIR}/${RUN_NAME}_direct_review_raw.jsonl"
DIRECT_REVIEW_TRAIN="${TRAIN_DIR}/${RUN_NAME}_direct_review.jsonl"
ALIGN_META="${RUN_DIR}/direct_review_alignment_summary.json"
SUMMARY_JSON="${RUN_DIR}/summary.json"

OUTPUT_ROOT="${ROOT}/model_training/src/output"
STATIC_MODEL="${OUTPUT_ROOT}/review-lora-${RUN_NAME}-static-${MAX_STEPS}step"
DIRECT_REVIEW_MODEL="${OUTPUT_ROOT}/review-lora-${RUN_NAME}-direct-review-${MAX_STEPS}step"

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

step_count_list() {
  PYTHONPATH="${ROOT}" python - <<PY
from data_collection.review_experiment_utils import normalize_step_counts
raw = "${STEP_COUNTS}".replace(",", " ").split()
print(" ".join(str(item) for item in normalize_step_counts(int(item) for item in raw)))
PY
}

step_tag() {
  local step_count="$1"
  echo "direct_stepwise_${step_count}step"
}

step_raw_path() {
  local step_count="$1"
  echo "${RUN_DIR}/$(step_tag "${step_count}")_bootstrap_raw.jsonl"
}

step_train_raw_path() {
  local step_count="$1"
  echo "${TRAIN_DIR}/${RUN_NAME}_$(step_tag "${step_count}")_raw.jsonl"
}

step_train_path() {
  local step_count="$1"
  echo "${TRAIN_DIR}/${RUN_NAME}_$(step_tag "${step_count}").jsonl"
}

step_model_path() {
  local step_count="$1"
  echo "${OUTPUT_ROOT}/review-lora-${RUN_NAME}-$(step_tag "${step_count}")-${MAX_STEPS}step"
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

prepare_seed_data() {
  CURRENT_STAGE="prepare_seed_data"
  if [[ -f "${SEED_DATA}" && -f "${SEED_META}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${SEED_DATA}"
    return
  fi
  cd "${ROOT}"
  PYTHONPATH="${ROOT}/data_collection" python data_collection/prepare_axiom_seedset.py \
    --output "${SEED_DATA}" \
    --metadata "${SEED_META}" \
    --per_grade "${AXIOM_PER_GRADE}" \
    --min_grade "${AXIOM_MIN_GRADE}" \
    --max_grade "${AXIOM_MAX_GRADE}" \
    --max_code_chars "${AXIOM_MAX_CODE_CHARS}"
}

generate_direct_review() {
  CURRENT_STAGE="generate_direct_review"
  if [[ -f "${DIRECT_REVIEW_RAW}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${DIRECT_REVIEW_RAW}"
    return
  fi
  final_review_args=()
  if [[ "${FREEFORM_FINAL_REVIEW}" != "1" && "${FREEFORM_FINAL_REVIEW,,}" != "true" ]]; then
    final_review_args=(--no-freeform_final_review)
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
    --output "${DIRECT_REVIEW_RAW}" \
    --dimension "Correctness Verification" \
    --batch_size "${DIRECT_BATCH_SIZE}" \
    --repeats "${DIRECT_REPEATS}" \
    --response_mode review \
    --reasoning_steps 0 \
    "${final_review_args[@]}"
}

generate_stepwise_variant() {
  local step_count="$1"
  local output_path
  output_path="$(step_raw_path "${step_count}")"
  CURRENT_STAGE="generate_$(step_tag "${step_count}")"
  if [[ -f "${output_path}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${output_path}"
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
    --output "${output_path}" \
    --dimension "Correctness Verification" \
    --batch_size "${DIRECT_BATCH_SIZE}" \
    --repeats "${DIRECT_REPEATS}" \
    --response_mode stepwise \
    --reasoning_steps "${step_count}"
}

prepare_static_train() {
  CURRENT_STAGE="prepare_static_train"
  if [[ ! -f "${STATIC_TRAIN_RAW}" ]]; then
    cd "${ROOT}"
    PYTHONPATH="${ROOT}:${ROOT}/model_training/src:${ROOT}/data_collection" \
      python data_collection/prepare_static_review_train_data.py \
        --input "${SEED_DATA}" \
        --output "${STATIC_TRAIN_RAW}"
  fi
  cp "${STATIC_TRAIN_RAW}" "${STATIC_TRAIN}"
}

prepare_bootstrap_train() {
  local input_path="$1"
  local output_path="$2"
  local policy_mode="$3"
  CURRENT_STAGE="prepare_$(basename "${output_path}" .jsonl)"
  if [[ -f "${output_path}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${output_path}"
    return
  fi
  cd "${ROOT}"
  PYTHONPATH="${ROOT}:${ROOT}/model_training/src:${ROOT}/data_collection" \
    python -m magicoder.preprocess_review_mcts_data \
      --input "${input_path}" \
      --output_file "${output_path}" \
      --policy_min_q "${POLICY_MIN_Q}" \
      --policy_response_mode "${policy_mode}" \
      --max_value_paths_per_dimension 0 \
      --max_problem_chars "${MAX_PROBLEM_CHARS}" \
      --max_code_chars "${MAX_CODE_CHARS}"
}

align_direct_review_to_static() {
  CURRENT_STAGE="align_direct_review_to_static"
  if [[ "${RUN_STATIC}" != "1" ]]; then
    cp "${DIRECT_REVIEW_TRAIN_RAW}" "${DIRECT_REVIEW_TRAIN}"
    python - <<PY > "${ALIGN_META}"
import json
print(json.dumps({"skipped": True, "reason": "RUN_STATIC=0"}, indent=2))
PY
    return
  fi
  if [[ -f "${DIRECT_REVIEW_TRAIN}" && -f "${ALIGN_META}" ]]; then
    echo "[stage:${CURRENT_STAGE}] exists: ${DIRECT_REVIEW_TRAIN}"
    return
  fi
  local static_counts
  local direct_counts
  local targets
  static_counts="$(count_train "${STATIC_TRAIN}")"
  direct_counts="$(count_train "${DIRECT_REVIEW_TRAIN_RAW}")"
  targets="$(
    PYTHONPATH="${ROOT}" python - <<PY
import json
from data_collection.review_experiment_utils import direct_review_alignment_targets
print(json.dumps(direct_review_alignment_targets(json.loads('''${static_counts}'''), json.loads('''${direct_counts}'''))))
PY
  )"
  echo "${targets}" | python -m json.tool > "${ALIGN_META}"
  cd "${ROOT}"
  PYTHONPATH="${ROOT}" python data_collection/rebalance_review_train_data.py \
    --input "${DIRECT_REVIEW_TRAIN_RAW}" \
    --output "${DIRECT_REVIEW_TRAIN}" \
    --target_policy_count "$(python - <<PY
import json
print(json.loads('''${targets}''')["target_policy_count"])
PY
)" \
    --target_value_count -1 \
    --target_total_count -1 \
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
from magicoder.prompt_template import AXIOM_REFINEMENT_SCALE, QWEN_REVIEW_STEP_PROMPT, REVIEW_STEP_FORMAT_SECTION
tokenizer = AutoTokenizer.from_pretrained("${MODEL_PATH}", use_fast=True, trust_remote_code=True)
rows = [json.loads(line) for line in Path("${train_data}").read_text(encoding="utf-8").splitlines() if line.strip()]
lengths = []
for row in rows:
    ids = tokenizer.encode(
        QWEN_REVIEW_STEP_PROMPT.format(
            instruction=row.get("instruction", ""),
            response="",
            axiom_scale=AXIOM_REFINEMENT_SCALE,
            step_format_section=REVIEW_STEP_FORMAT_SECTION,
        ),
        add_special_tokens=True,
    )
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
    "train_data": "${train_data}",
    "examples": len(lengths),
    "max_training_seq_length": int("${MAX_TRAINING_SEQ_LENGTH}"),
    "fit_examples": sum(x <= int("${MAX_TRAINING_SEQ_LENGTH}") for x in lengths),
    "too_long_examples": sum(x > int("${MAX_TRAINING_SEQ_LENGTH}") for x in lengths),
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
  require_idle_gpus
  audit_lengths "${train_data}"
  cd "${ROOT}/model_training/src"
  resume_args=()
  latest_checkpoint="$(find "${output_model}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "${latest_checkpoint}" ]]; then
    resume_args=(--resume_from_checkpoint "${latest_checkpoint}")
  fi
  CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_HUB_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  TRL_EXPERIMENTAL_SILENCE=1 \
  accelerate launch \
    --num_processes 2 \
    --gpu_ids "${TRAIN_CUDA_VISIBLE_DEVICES}" \
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
      --logging_steps 10 \
      --save_strategy steps \
      --save_steps 80 \
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
      --seed 20260507 \
      --skip_save False \
      "${resume_args[@]}" >"${log_file}" 2>&1
}

eval_final_only_model() {
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
  FINAL_ONLY_JSON=1 \
  EVAL_BASE_DIRECT=0 \
  EVAL_TRAINED_DIRECT=1 \
  EVAL_TRAINED_VALUE=0 \
  TRAINED_DIRECT_MAX_STEPS=1 \
  TRAINED_DIRECT_NUM_CANDIDATES=1 \
  USE_CHAT_TEMPLATE=1 \
  USE_UV=0 \
  CHAT_TEMPLATE_ENABLE_THINKING="${CHAT_TEMPLATE_ENABLE_THINKING}" \
  EVAL_CUDA_DEVICE="${EVAL_CUDA_VISIBLE_DEVICES}" \
  NTFY_URL="${NTFY_URL}" \
  "${ROOT}/data_collection/scripts/run_axiom_clean_eval.sh"
}

eval_stepwise_model() {
  local step_count="$1"
  local trained_model="$2"
  local tag
  tag="$(step_tag "${step_count}")"
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
  EVAL_TRAINED_VALUE=0 \
  TRAINED_DIRECT_MAX_STEPS="${step_count}" \
  REASONING_STEPS="${step_count}" \
  TRAINED_DIRECT_NUM_CANDIDATES=1 \
  MAX_RETHINKS=0 \
  USE_CHAT_TEMPLATE=1 \
  USE_UV=0 \
  CHAT_TEMPLATE_ENABLE_THINKING="${CHAT_TEMPLATE_ENABLE_THINKING}" \
  EVAL_CUDA_DEVICE="${EVAL_CUDA_VISIBLE_DEVICES}" \
  NTFY_URL="${NTFY_URL}" \
  "${ROOT}/data_collection/scripts/run_axiom_clean_eval.sh"
}

write_summary() {
  CURRENT_STAGE="write_summary"
  local counts
  counts="$(step_count_list)"
  PYTHONPATH="${ROOT}" python - <<PY
import json
from pathlib import Path
from data_collection.review_experiment_utils import stepwise_variants

root = Path("${ROOT}")
run_name = "${RUN_NAME}"
step_counts = [int(item) for item in "${counts}".split() if item.strip()]

def load_eval(tag: str):
    path = root / "model_training" / "src" / "output" / f"review-eval-{run_name}_{tag}_axiom_eval" / "comparison.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("trained_direct_clean")

payload = {
    "run_name": run_name,
    "model_key": "${MODEL_KEY}",
    "model_path": "${MODEL_PATH}",
    "experiment_thinking_mode": "${EXPERIMENT_THINKING_MODE}",
    "chat_template_enable_thinking": "${CHAT_TEMPLATE_ENABLE_THINKING}",
    "cfg": "${CFG}",
    "seed_data": "${SEED_DATA}",
    "seed_metadata": "${SEED_META}",
    "axiom_per_grade": int("${AXIOM_PER_GRADE}"),
    "eval_per_grade": int("${EVAL_PER_GRADE}"),
    "axiom_max_code_chars": int("${AXIOM_MAX_CODE_CHARS}"),
    "max_problem_chars": int("${MAX_PROBLEM_CHARS}"),
    "max_code_chars": int("${MAX_CODE_CHARS}"),
    "max_steps": int("${MAX_STEPS}"),
    "max_training_seq_length": int("${MAX_TRAINING_SEQ_LENGTH}"),
    "direct_repeats": int("${DIRECT_REPEATS}"),
    "direct_batch_size": int("${DIRECT_BATCH_SIZE}"),
    "freeform_final_review": "${FREEFORM_FINAL_REVIEW}",
    "static_train": "${STATIC_TRAIN}" if "${RUN_STATIC}" == "1" else None,
    "direct_review_train": "${DIRECT_REVIEW_TRAIN}",
    "direct_review_alignment": json.loads(Path("${ALIGN_META}").read_text(encoding="utf-8")) if Path("${ALIGN_META}").exists() else None,
    "static_eval": load_eval("static") if "${RUN_STATIC}" == "1" else None,
    "direct_review_eval": load_eval("direct_review"),
    "stepwise_variants": {},
}
for variant in stepwise_variants(step_counts):
    tag = str(variant["tag"])
    payload["stepwise_variants"][tag] = {
        "reasoning_steps": int(variant["reasoning_steps"]),
        "train_data": "${TRAIN_DIR}/" + f"{run_name}_{tag}.jsonl",
        "model_path": "${OUTPUT_ROOT}/" + f"review-lora-{run_name}-{tag}-${MAX_STEPS}step",
        "eval": load_eval(tag),
    }
Path("${SUMMARY_JSON}").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}

main() {
  local step_counts
  step_counts="$(step_count_list)"
  echo "[start] run=${RUN_NAME} mode=${EXPERIMENT_THINKING_MODE} steps=${step_counts} static=${RUN_STATIC} time=$(date -Is) log=${LOG_FILE}"

  prepare_seed_data
  generate_direct_review
  for step_count in ${step_counts}; do
    generate_stepwise_variant "${step_count}"
  done

  if [[ "${RUN_STATIC}" == "1" ]]; then
    prepare_static_train
  fi
  prepare_bootstrap_train "${DIRECT_REVIEW_RAW}" "${DIRECT_REVIEW_TRAIN_RAW}" "final_review"
  align_direct_review_to_static
  for step_count in ${step_counts}; do
    prepare_bootstrap_train "$(step_raw_path "${step_count}")" "$(step_train_raw_path "${step_count}")" "path"
    cp "$(step_train_raw_path "${step_count}")" "$(step_train_path "${step_count}")"
  done

  if [[ "${STOP_AFTER_DATA:-0}" == "1" ]]; then
    CURRENT_STAGE="finished"
    echo "[finished] STOP_AFTER_DATA=1 run_dir=${RUN_DIR}"
    exit 0
  fi

  if [[ "${RUN_STATIC}" == "1" ]]; then
    train_fsdp "static" "${STATIC_TRAIN}" "${STATIC_MODEL}"
  fi
  train_fsdp "direct_review" "${DIRECT_REVIEW_TRAIN}" "${DIRECT_REVIEW_MODEL}"
  for step_count in ${step_counts}; do
    train_fsdp "$(step_tag "${step_count}")" "$(step_train_path "${step_count}")" "$(step_model_path "${step_count}")"
  done

  if [[ "${STOP_AFTER_TRAIN:-0}" == "1" ]]; then
    CURRENT_STAGE="finished"
    echo "[finished] STOP_AFTER_TRAIN=1 run_dir=${RUN_DIR}"
    exit 0
  fi

  if [[ "${RUN_STATIC}" == "1" ]]; then
    eval_final_only_model "static" "${STATIC_MODEL}"
  fi
  eval_final_only_model "direct_review" "${DIRECT_REVIEW_MODEL}"
  for step_count in ${step_counts}; do
    eval_stepwise_model "${step_count}" "$(step_model_path "${step_count}")"
  done
  write_summary

  CURRENT_STAGE="finished"
  notify "FINISHED"
  echo "[finished] summary=${SUMMARY_JSON}"
}

main "$@"
