#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1

# -----------------------------
# Fixed config
# -----------------------------
BASE_MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
TRAIN_PATH="data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b_filtered.jsonl"
VAL_PATH="data/sft-cs336-assign5-datasets/sft-reason/val.jsonl"

ROOT_OUT="./grpo_offpolicy_sweeps"

ROLLOUT_BS=256
GROUP_SIZE=8
LR="1e-5"
EVAL_EVERY=10
SEED=42

LOSS_TYPE="grpo_clip"
LENGTH_NORM="masked_mean"

# 你当前 argparse 默认 use_std_normalization=False
# 如果想统一打开，把 STD_NORM_FLAG 改成 "--use_std_normalization"
STD_NORM_FLAG="--use_std_normalization"

# 保持显存稳定：micro_batch = train_bs / grad_accum = 4
MICRO_BS=4

# -----------------------------
# Sweep grids
# -----------------------------
# Coarse (<50 steps)
COARSE_STEPS=50
COARSE_EPOCHS=(1 2 4)
COARSE_TRAIN_BS=(64 128 256)

# Fine (200 steps) -- 更聚焦
FINE_STEPS=200
FINE_EPOCHS=(4)
FINE_TRAIN_BS=(256 64 128)

# -----------------------------
# Helpers
# -----------------------------
run_one () {
  local phase="$1"      # "coarse" or "fine"
  local n_steps="$2"
  local epochs="$3"
  local train_bs="$4"

  local grad_accum=$(( train_bs / MICRO_BS ))
  if (( train_bs % MICRO_BS != 0 )); then
    echo "ERROR: train_bs (${train_bs}) must be divisible by MICRO_BS (${MICRO_BS})" >&2
    exit 1
  fi

  local timestamp
  timestamp=$(date +%m%d_%H%M%S)

  local out_dir="${ROOT_OUT}/${phase}/e${epochs}_bs${train_bs}_ga${grad_accum}_steps${n_steps}_${timestamp}"
  mkdir -p "${out_dir}"
  local log_file="${out_dir}/run.log"

  echo "============================================================"
  echo "PHASE: ${phase}"
  echo "RUN: epochs_per_rollout_batch=${epochs} | train_batch_size=${train_bs} | grad_accum_steps=${grad_accum} | n_steps=${n_steps}"
  echo "OUT: ${out_dir}"
  echo "============================================================"

  cmd=(python scripts/grpo.py
    --base_model "${BASE_MODEL}"
    --train_data_path "${TRAIN_PATH}"
    --val_data_path "${VAL_PATH}"
    --output_dir "${out_dir}"
    --n_grpo_steps "${n_steps}"
    --rollout_batch_size "${ROLLOUT_BS}"
    --group_size "${GROUP_SIZE}"
    --train_batch_size "${train_bs}"
    --grad_accum_steps "${grad_accum}"
    --lr "${LR}"
    --epochs_per_rollout_batch "${epochs}"
    --loss_type "${LOSS_TYPE}"
    --length_norm "${LENGTH_NORM}"
    --eval_every_steps "${EVAL_EVERY}"
    --seed "${SEED}"
  )

  # 可选：开 std normalization
  if [[ -n "${STD_NORM_FLAG}" ]]; then
    cmd+=(${STD_NORM_FLAG})
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
# Main
# -----------------------------
mkdir -p "${ROOT_OUT}/coarse" "${ROOT_OUT}/fine"

echo "=============================="
echo "Starting AGGRESSIVE sweep (Epochs=8)..."
echo "=============================="

# We fix Train Batch Size to 256 (best theoretical throughput) 
# and push Epochs to 8 to see if it diverges.
# Running for 50 steps first to check for early crash/divergence.

# AGGRESSIVE_EPOCHS=8
# AGGRESSIVE_BS=256
# AGGRESSIVE_STEPS=50

# run_one "coarse" "${AGGRESSIVE_STEPS}" "${AGGRESSIVE_EPOCHS}" "${AGGRESSIVE_BS}"

# We fix Train Batch Size to 256 (best theoretical throughput) 
# and push Epochs to 8 to see if it diverges.
# Running for 50 steps first to check for early crash/divergence.

# echo "=============================="
# echo "Starting COARSE sweep..."
# echo "=============================="
# for e in "${COARSE_EPOCHS[@]}"; do
#   for bs in "${COARSE_TRAIN_BS[@]}"; do
#     run_one "coarse" "${COARSE_STEPS}" "${e}" "${bs}"
#   done
# done

echo "=============================="
echo "Starting FINE sweep..."
echo "=============================="
for e in "${FINE_EPOCHS[@]}"; do
  for bs in "${FINE_TRAIN_BS[@]}"; do
    run_one "fine" "${FINE_STEPS}" "${e}" "${bs}"
  done
done

echo "All coarse+fine sweep runs completed."
