#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config (edit if needed)
# -----------------------------
export CUDA_VISIBLE_DEVICES=0,1

BASE_MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
VAL_PATH="data/sft-cs336-assign5-datasets/sft-reason/val.jsonl"

RAW_TRAIN="data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b.jsonl"
FILTERED_TRAIN="data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b_filtered.jsonl"

BATCH_SIZE=4
GRAD_ACCUM=8
LR=5e-5
EPOCHS=5

# Sweep sizes you requested
SIZES=(128 256 512 1024 2048)

# For "full dataset" runs (no max_examples), pick a not-too-frequent eval interval.
# If eval is still slow, increase this (e.g., 500).
FULL_EVAL_EVERY=200

# -----------------------------
# Helpers
# -----------------------------
calc_eval_every_steps () {
  local n="$1"
  # steps_per_epoch = ceil(n / batch_size)
  local steps_per_epoch=$(( (n + BATCH_SIZE - 1) / BATCH_SIZE ))
  # eval 2x per epoch => half an epoch
  local eval_every=$(( steps_per_epoch / 2 ))
  # guardrails
  if (( eval_every < 5 )); then eval_every=5; fi
  echo "${eval_every}"
}

run_one () {
  local train_path="$1"
  local tag="$2"          # e.g. raw or filtered
  local size="$3"         # e.g. 256 or "full"
  local out_dir="$4"
  local eval_every="$5"

  echo "=============================="
  echo "RUN: ${tag} | size=${size} | eval_every_steps=${eval_every}"
  echo "OUT: ${out_dir}"
  echo "TRAIN: ${train_path}"
  echo "=============================="

  mkdir -p "${out_dir}"
  local log_file="${out_dir}/run.log"

  # Build the python command as an array (safe for spaces)
  cmd=(python scripts/sft.py
      --base_model "${BASE_MODEL}"
      --sft_data_path "${train_path}"
      --val_data_path "${VAL_PATH}"
      --output_dir "${out_dir}"
      --batch_size "${BATCH_SIZE}"
      --grad_accum_steps "${GRAD_ACCUM}"
      --lr "${LR}"
      --eval_every_steps "${eval_every}"
      --epochs "${EPOCHS}"
  )

  # If not full, add max_examples
  if [[ "${size}" != "full" ]]; then
    cmd+=(--max_examples "${size}")
  fi

    {
    echo "Command:"
    printf '  %q ' "${cmd[@]}"
    echo
    echo "------------------------------"
    "${cmd[@]}"
  } 2>&1 | tee -a "${log_file}"

}
# -----------------------------
# Main sweep
# -----------------------------
timestamp=$(date +%m%d_%H%M%S)

# 1) Raw dataset sweep
for n in "${SIZES[@]}"; do
  eval_every=$(calc_eval_every_steps "${n}")
  out_dir="./sft_experiments_raw_${n}_${timestamp}"
  run_one "${RAW_TRAIN}" "raw" "${n}" "${out_dir}" "${eval_every}"
done

# Raw full
out_dir="./sft_experiments_raw_full_${timestamp}"
run_one "${RAW_TRAIN}" "raw" "full" "${out_dir}" "${FULL_EVAL_EVERY}"

# 2) Filtered dataset sweep
for n in "${SIZES[@]}"; do
  eval_every=$(calc_eval_every_steps "${n}")
  out_dir="./sft_experiments_filtered_${n}_${timestamp}"
  run_one "${FILTERED_TRAIN}" "filtered" "${n}" "${out_dir}" "${eval_every}"
done

# Filtered full
out_dir="./sft_experiments_filtered_full_${timestamp}"
run_one "${FILTERED_TRAIN}" "filtered" "full" "${out_dir}" "${FULL_EVAL_EVERY}"

echo "All runs submitted/completed."
