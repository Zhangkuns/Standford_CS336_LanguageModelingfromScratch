#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config
# -----------------------------
export CUDA_VISIBLE_DEVICES=0,1

BASE_MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
# Note: Expert Iteration needs the raw questions file, not the SFT file
TRAIN_DATA="data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b_filtered.jsonl"
VAL_DATA="data/sft-cs336-assign5-datasets/sft-reason/val.jsonl"

# GPU Training Params (Safe for 48GB GPU)
GPU_BATCH_SIZE=4
GRAD_ACCUM=8
LR=5e-5
MAX_GRAD_NORM=1.0
N_EI_STEPS=5

# -----------------------------
# Helper Function
# -----------------------------
run_ei_one () {
  local exp_bs="$1"      # expert_batch_size (Db)
  local rollouts="$2"    # rollouts_per_question (G)
  local epochs="$3"      # sft_epochs_per_step
  local tag="$4"         # naming tag

  local out_dir="./ei_experiments_${tag}_Db${exp_bs}_G${rollouts}_Ep${epochs}_$(date +%m%d_%H%M)"

  echo "============================================================"
  echo "RUN: ${tag} | Db=${exp_bs} | G=${rollouts} | Ep=${epochs}"
  echo "OUT: ${out_dir}"
  echo "============================================================"

  mkdir -p "${out_dir}"
  local log_file="${out_dir}/run.log"

  cmd=(python scripts/sft_ei.py
      --base_model "${BASE_MODEL}"
      --train_data_path "${TRAIN_DATA}"
      --val_data_path "${VAL_DATA}"
      --output_dir "${out_dir}"
      --n_ei_steps "${N_EI_STEPS}"
      --expert_batch_size "${exp_bs}"
      --rollouts_per_question "${rollouts}"
      --sft_epochs_per_step "${epochs}"
      --batch_size "${GPU_BATCH_SIZE}"
      --grad_accum_steps "${GRAD_ACCUM}"
      --lr "${LR}"
      --max_grad_norm "${MAX_GRAD_NORM}"
      --eval_max_examples 256
  )

  {
    echo "Command:"
    printf '  %q ' "${cmd[@]}"
    echo
    echo "------------------------------------------------------------"
    "${cmd[@]}"
  } 2>&1 | tee -a "${log_file}"
}

# -----------------------------
# 1. Batch Size Sweep
# (Fix G=4, Ep=4, Vary Db)
# -----------------------------
BATCH_SIZES=(512 1024 2048)

for db in "${BATCH_SIZES[@]}"; do
    run_ei_one "${db}" "4" "4" "batch_sweep"
done

# -----------------------------
# 2. Rollout & Epoch Sweep
# (Fix Db=512, Vary G and Ep)
# Note: We already ran Db=512, G=4, Ep=1 above, so we skip that combo here.
# -----------------------------

# Experiment A: High Exploration (More Rollouts)
run_ei_one "512" "1" "4" "rollout_sweep"

run_ei_one "512" "8" "4" "rollout_sweep"

run_ei_one "512" "16" "4" "rollout_sweep"

# Experiment B: Deep Training (More Epochs)
run_ei_one "512" "4" "8" "epoch_sweep"

run_ei_one "512" "4" "16" "combo_sweep"

echo "All Expert Iteration runs submitted/completed."