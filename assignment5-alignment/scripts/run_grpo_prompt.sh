#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1

BASE_MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
TRAIN_PATH="data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b_filtered.jsonl"
VAL_PATH="data/sft-cs336-assign5-datasets/sft-reason/val.jsonl"

ROOT_OUT="./grpo_prompt"
N_STEPS=200
ROLLOUT_BS=256
GROUP_SIZE=8
TRAIN_BS=256
GRAD_ACCUM=64
LR="1e-5"
EPOCHS_PER_ROLLOUT=1
EVAL_EVERY=10
SEED=42

LOSS_TYPE="grpo_clip"
LENGTH_NORM="masked_mean"   

timestamp=$(date +%m%d_%H%M%S)
out_dir="${ROOT_OUT}/prompt_${timestamp}"
mkdir -p "${out_dir}"
log_file="${out_dir}/run.log"

cmd=(python scripts/grpo_prompt.py
  --base_model "${BASE_MODEL}"
  --train_data_path "${TRAIN_PATH}"
  --val_data_path "${VAL_PATH}"
  --output_dir "${out_dir}"
  --n_grpo_steps "${N_STEPS}"
  --rollout_batch_size "${ROLLOUT_BS}"
  --group_size "${GROUP_SIZE}"
  --train_batch_size "${TRAIN_BS}"
  --grad_accum_steps "${GRAD_ACCUM}"
  --lr "${LR}"
  --use_std_normalization
  --epochs_per_rollout_batch "${EPOCHS_PER_ROLLOUT}"
  --loss_type "${LOSS_TYPE}"
  --length_norm "${LENGTH_NORM}"
  --eval_every_steps "${EVAL_EVERY}"
  --seed "${SEED}"
)

{
  echo "Command:"
  printf '  %q ' "${cmd[@]}"
  echo
  echo "------------------------------"
  "${cmd[@]}"
} 2>&1 | tee -a "${log_file}"

echo "Done. Logs: ${log_file}"
