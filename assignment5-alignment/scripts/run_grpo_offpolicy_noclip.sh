#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1

# -----------------------------
# Fixed config (same as your best off-policy run)
# -----------------------------
BASE_MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
TRAIN_PATH="data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b_filtered.jsonl"
VAL_PATH="data/sft-cs336-assign5-datasets/sft-reason/val.jsonl"

ROOT_OUT="./grpo_off_policy_clip_ablation"

# Best off-policy hyperparams you reported:
N_STEPS=200
ROLLOUT_BS=256
GROUP_SIZE=8
TRAIN_BS=256
MICRO_BS=4
GRAD_ACCUM=64         # = TRAIN_BS / MICRO_BS
LR="1e-5"
EPOCHS_PER_ROLLOUT=4  # off-policy
EVAL_EVERY=10
SEED=42

# IMPORTANT: set this to your newly-implemented loss type name
# e.g. "GRPO-No-Clip" or "grpo_no_clip" depending on argparse
LOSS_TYPE="grpo_no_clip"

# keep same as before
LENGTH_NORM="masked_mean"

# If your best run used std normalization, keep it on.
# Otherwise set STD_NORM_FLAG="" to disable.
STD_NORM_FLAG="--use_std_normalization"

# sanity check
if (( TRAIN_BS % MICRO_BS != 0 )); then
  echo "ERROR: TRAIN_BS (${TRAIN_BS}) must be divisible by MICRO_BS (${MICRO_BS})" >&2
  exit 1
fi
if (( GRAD_ACCUM != TRAIN_BS / MICRO_BS )); then
  echo "ERROR: GRAD_ACCUM (${GRAD_ACCUM}) should equal TRAIN_BS/MICRO_BS ($((TRAIN_BS/MICRO_BS)))" >&2
  exit 1
fi

timestamp=$(date +%m%d_%H%M%S)
out_dir="${ROOT_OUT}/no_clip_e${EPOCHS_PER_ROLLOUT}_bs${TRAIN_BS}_ga${GRAD_ACCUM}_steps${N_STEPS}_${timestamp}"
mkdir -p "${out_dir}"
log_file="${out_dir}/run.log"

cmd=(python scripts/grpo.py
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
  --epochs_per_rollout_batch "${EPOCHS_PER_ROLLOUT}"
  --loss_type "${LOSS_TYPE}"
  --length_norm "${LENGTH_NORM}"
  --eval_every_steps "${EVAL_EVERY}"
  --seed "${SEED}"
)

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

echo "Done. Logs: ${log_file}"
