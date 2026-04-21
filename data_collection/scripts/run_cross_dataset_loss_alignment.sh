#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data1/xianzhiwei/mcts-code-review}"
RUN_NAME="${RUN_NAME:-cross_dataset_loss_alignment_20260421}"
MODEL_PATH="${MODEL_PATH:-/data1/xianzhiwei/model/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
CROSS_EVAL_RUN="${CROSS_EVAL_RUN:-cross_dataset_review_eval_20260421}"
RUN_DIR="${ROOT}/data_collection/review_mcts_runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
TRAIN_DATA="${ROOT}/model_training/review_mcts_train_data/${RUN_NAME}.jsonl"
OUTPUT_MODEL="${ROOT}/model_training/src/output/review-lora-${RUN_NAME}-240step"
EVAL_ROOT="${ROOT}/model_training/src/output/review-eval-${RUN_NAME}"
CROSS_EVAL_ROOT="${ROOT}/model_training/src/output/review-eval-${CROSS_EVAL_RUN}"
MANIFEST="${ROOT}/data_collection/review_mcts_runs/${CROSS_EVAL_RUN}/manifests/all.jsonl"
INDICES="${ROOT}/data_collection/review_mcts_runs/${CROSS_EVAL_RUN}/manifests/all_indices.json"
LIMIT_PER_SOURCE="${LIMIT_PER_SOURCE:-400}"
MAX_TRAINING_SEQ_LENGTH="${MAX_TRAINING_SEQ_LENGTH:-640}"
NTFY_URL="${NTFY_URL:-https://ntfy.sh/iponika_mcts}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${EVAL_ROOT}"
LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE="init"

notify() {
  local status="$1"
  local message="${RUN_NAME} ${status}: stage=${CURRENT_STAGE}, host=$(hostname), time=$(date -Is), log=${LOG_FILE}, output=${EVAL_ROOT}"
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    curl --noproxy '*' -fsS -d "${message}" "${NTFY_URL}" >/dev/null 2>&1 || true
}

on_exit() {
  local code=$?
  if [[ "${CURRENT_STAGE}" != "finished" ]]; then
    notify "FAILED(code=${code})"
  fi
  exit "${code}"
}
trap on_exit EXIT

require_inputs() {
  CURRENT_STAGE="require_inputs"
  for path in "${MANIFEST}" "${INDICES}"; do
    if [[ ! -f "${path}" ]]; then
      echo "Missing required input: ${path}" >&2
      exit 1
    fi
  done
}

prepare_data() {
  CURRENT_STAGE="prepare_data"
  cd "${ROOT}"
  if [[ -f "${TRAIN_DATA}" ]]; then
    echo "[stage:${CURRENT_STAGE}] training data exists: ${TRAIN_DATA}"
  else
    PYTHONPATH="${ROOT}/model_training/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.preprocess_score_datasets \
      --axiom_dir "${ROOT}/datasets/axiom-llm-judge" \
      --codecriticbench "${ROOT}/datasets/CodeCriticBench/data/CodeCriticBench.jsonl" \
      --code_diting_root "${ROOT}/datasets/Code-DiTing" \
      --codejudgebench_root "${ROOT}/benchmarks/mattymchen___codejudgebench" \
      --include_codejudge_as_intervals \
      --train_lm_exact \
      --limit_per_source "${LIMIT_PER_SOURCE}" \
      --disable_shuffle \
      --output_file "${TRAIN_DATA}"
  fi
  PYTHONDONTWRITEBYTECODE=1 python - <<PY
import json
from collections import Counter
from pathlib import Path

path = Path("${TRAIN_DATA}")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
summary = {
    "path": str(path),
    "items": len(rows),
    "sources": dict(Counter(row.get("source") for row in rows)),
    "label_types": dict(Counter(row.get("label_type") for row in rows)),
    "train_lm": sum(1 for row in rows if row.get("train_lm")),
    "value_only": sum(1 for row in rows if not row.get("train_lm")),
    "pair_items": sum(1 for row in rows if row.get("pair_id")),
}
(Path("${RUN_DIR}") / "train_data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
PY
}

train_model() {
  CURRENT_STAGE="train_model"
  local log_file="${LOG_DIR}/train.log"
  if [[ -f "${OUTPUT_MODEL}/adapter_model.safetensors" && -f "${OUTPUT_MODEL}/value_head.pth" ]]; then
    echo "[stage:${CURRENT_STAGE}] final checkpoint exists, skipping: ${OUTPUT_MODEL}"
    return
  fi
  cd "${ROOT}/model_training/src"
  resume_args=()
  latest_checkpoint="$(find "${OUTPUT_MODEL}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "${latest_checkpoint}" ]]; then
    resume_args=(--resume_from_checkpoint "${latest_checkpoint}")
    echo "[stage:${CURRENT_STAGE}] resuming from ${latest_checkpoint}"
  fi
  echo "[stage:${CURRENT_STAGE}] start=$(date -Is)"
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_DATASETS_CACHE=/tmp/hf-datasets-cache \
  HF_HUB_OFFLINE=1 \
  TRL_EXPERIMENTAL_SILENCE=1 \
  UV_CACHE_DIR=/tmp/uv-cache \
  uv run python -m magicoder.train_multi \
    --task review \
    --model_key Qwen/Qwen3.5-9B \
    --model_name_or_path "${MODEL_PATH}" \
    --datafile_paths "../review_mcts_train_data/$(basename "${TRAIN_DATA}")" \
    --output_dir "${OUTPUT_MODEL}" \
    --max_steps 240 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_training_seq_length "${MAX_TRAINING_SEQ_LENGTH}" \
    --bf16 True \
    --logging_steps 20 \
    --save_strategy steps \
    --save_steps 120 \
    --save_total_limit 2 \
    --report_to none \
    --optim adafactor \
    --learning_rate 3e-5 \
    --lr_scheduler_type linear \
    --warmup_steps 24 \
    --peft lora \
    --value_weight 0.05 \
    --boundary_value_weight 0.02 \
    --pairwise_value_weight 0.05 \
    --pairwise_margin 0.2 \
    --disable_train_shuffle True \
    --train_sampling_strategy sequential \
    --num_proc 1 \
    --seed 20260421 \
    "${resume_args[@]}" >"${log_file}" 2>&1
  echo "[stage:${CURRENT_STAGE}] done=$(date -Is)"
}

eval_one() {
  local tag="$1"
  local cuda_device="$2"
  local skip_value="$3"
  local max_steps="$4"
  local num_candidates="$5"
  local temperature="$6"
  local top_p="$7"
  local seed="$8"
  local eval_dir="${EVAL_ROOT}/${tag}"
  local log_file="${LOG_DIR}/eval_${tag}.log"
  mkdir -p "${eval_dir}"
  if [[ -f "${eval_dir}/summary.json" ]]; then
    echo "[stage:eval_${tag}] summary exists, skipping"
    return
  fi
  (
    set -euo pipefail
    cd "${ROOT}/model_training/src"
    skip_args=()
    if [[ "${skip_value}" == "skip_value" ]]; then
      skip_args=(--skip_value_scoring)
    fi
    CUDA_VISIBLE_DEVICES="${cuda_device}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_DATASETS_CACHE=/tmp/hf-datasets-cache \
    HF_HUB_OFFLINE=1 \
    TRL_EXPERIMENTAL_SILENCE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python -m magicoder.batch_review_evaluator \
      --policy_model_path "${OUTPUT_MODEL}" \
      --value_model_path "${OUTPUT_MODEL}" \
      --share_policy_value_model \
      "${skip_args[@]}" \
      --input_record "${MANIFEST}" \
      --record_indices_file "${INDICES}" \
      --output_dir "${eval_dir}" \
      --dimensions "Correctness Verification" \
      --device cuda \
      --dtype bf16 \
      --max_steps "${max_steps}" \
      --num_candidates "${num_candidates}" \
      --max_new_tokens 192 \
      --final_max_new_tokens 256 \
      --temperature "${temperature}" \
      --top_p "${top_p}" \
      --score_key response_mean_value \
      --seed "${seed}" \
      --final_only_json \
      --format_penalty 1.0 \
      --low_grade_no_evidence_penalty 0.4 \
      --final_temperature 0 \
      --max_final_retries 2
    PYTHONPATH="${ROOT}/model_training/src:${ROOT}/data_collection" \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    uv run python "${ROOT}/data_collection/scripts/summarize_review_eval_outputs.py" \
      --eval_dir "${eval_dir}" \
      --output "${eval_dir}/summary.json"
    echo "[stage:eval_${tag}] done=$(date -Is)"
  ) >"${log_file}" 2>&1
}

write_comparison() {
  CURRENT_STAGE="write_comparison"
  PYTHONDONTWRITEBYTECODE=1 python - <<PY
import json
from pathlib import Path

baseline_path = Path("${CROSS_EVAL_ROOT}") / "comparison.json"
eval_root = Path("${EVAL_ROOT}")
baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
current = {}
for tag in ["loss_aligned_direct_final", "loss_aligned_value_rerank_final"]:
    summary = json.loads((eval_root / tag / "summary.json").read_text(encoding="utf-8"))
    current[tag] = {
        "valid_rate": summary.get("valid_rate"),
        "grade_mae": summary.get("grade_mae"),
        "boundary_acc": summary.get("boundary_acc"),
        "interval_acc": summary.get("interval_acc"),
        "pairwise_acc": summary.get("pairwise_acc"),
        "lenient_grade_mae": summary.get("lenient_grade_mae"),
        "lenient_boundary_acc": summary.get("lenient_boundary_acc"),
        "lenient_interval_acc": summary.get("lenient_interval_acc"),
        "lenient_pairwise_acc": summary.get("lenient_pairwise_acc"),
        "by_source": summary.get("rows") and {},
    }
    by_source = {}
    for row in summary.get("rows") or []:
        source = row.get("source") or "unknown"
        by_source.setdefault(source, {"rows": 0, "valid": 0, "errors": [], "boundary": [], "interval": []})
        slot = by_source[source]
        slot["rows"] += 1
        if row.get("valid"):
            slot["valid"] += 1
        if "abs_grade_error" in row:
            slot["errors"].append(float(row["abs_grade_error"]))
        if "boundary_correct" in row:
            slot["boundary"].append(1.0 if row["boundary_correct"] else 0.0)
        if "interval_correct" in row:
            slot["interval"].append(1.0 if row["interval_correct"] else 0.0)
    compact = {}
    for source, slot in by_source.items():
        compact[source] = {
            "valid_rate": round(slot["valid"] / max(1, slot["rows"]), 6),
            "grade_mae": round(sum(slot["errors"]) / len(slot["errors"]), 6) if slot["errors"] else None,
            "boundary_acc": round(sum(slot["boundary"]) / len(slot["boundary"]), 6) if slot["boundary"] else None,
            "interval_acc": round(sum(slot["interval"]) / len(slot["interval"]), 6) if slot["interval"] else None,
        }
    current[tag]["by_source"] = compact
comparison = {"baseline_cross_eval": baseline, "loss_alignment": current}
(eval_root / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(comparison["loss_alignment"], ensure_ascii=False, indent=2, sort_keys=True))
PY
}

main() {
  echo "[start] run=${RUN_NAME} time=$(date -Is) log=${LOG_FILE}"
  require_inputs
  prepare_data
  if [[ "${STOP_AFTER_DATA:-0}" == "1" ]]; then
    CURRENT_STAGE="finished"
    echo "[finished] STOP_AFTER_DATA=1 data=${TRAIN_DATA}"
    exit 0
  fi
  train_model
  CURRENT_STAGE="eval"
  eval_one "loss_aligned_direct_final" 0 "skip_value" 1 1 0 1.0 202604214 &
  pid_direct=$!
  eval_one "loss_aligned_value_rerank_final" 1 "use_value" 1 2 0.7 0.95 202604215 &
  pid_value=$!
  echo "[stage:${CURRENT_STAGE}] waiting direct=${pid_direct} value=${pid_value}"
  wait "${pid_direct}"
  wait "${pid_value}"
  write_comparison
  CURRENT_STAGE="finished"
  notify "FINISHED"
  echo "[finished] output=${EVAL_ROOT} log=${LOG_FILE}"
}

main "$@"
