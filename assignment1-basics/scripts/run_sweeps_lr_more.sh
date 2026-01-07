#!/usr/bin/env bash
set -euo pipefail

TRAIN=/workspace/CS336/LAB/assignment1-basics/result/tinystories_train.npy
VAL=/workspace/CS336/LAB/assignment1-basics/result/tinystories_val.npy

VOCAB=10000
CTX=256
D_MODEL=512
N_LAYERS=4
N_HEADS=16
D_FF=1344
ROPE_THETA=10000.0
MAX_ITERS=40000
COS_ITERS=40000
WARMUP=1000
WD=0.1

GPU=0

LRS=(1e-1 5e-2)
FIXED_BS_FOR_LR=32

run_one () {
  local outdir="$1"
  local lr="$2"
  local bs="$3"
  local project="$4"
  local runname="$5"

  mkdir -p "$outdir"
  local logfile="$outdir/train_$(date '+%Y-%m-%d_%H-%M-%S').log"

  HIP_VISIBLE_DEVICES="$GPU" stdbuf -oL -eL \
  python -u cs336_basics/train.py \
    --train_data "$TRAIN" \
    --val_data "$VAL" \
    --output_dir "$outdir" \
    --vocab_size "$VOCAB" \
    --context_length "$CTX" \
    --d_model "$D_MODEL" \
    --num_layers "$N_LAYERS" \
    --num_heads "$N_HEADS" \
    --d_ff "$D_FF" \
    --rope_theta "$ROPE_THETA" \
    --batch_size "$bs" \
    --max_iters "$MAX_ITERS" \
    --cosine_cycle_iters "$COS_ITERS" \
    --warmup_iters "$WARMUP" \
    --lr "$lr" \
    --weight_decay "$WD" \
    --wandb \
    --wandb_project "$project" \
    --wandb_run_name "$runname" \
    2>&1 | ts '[%Y-%m-%d %H:%M:%S]' | tee "$logfile"

  local rc=${PIPESTATUS[0]}
  return $rc
}

for lr in "${LRS[@]}"; do
  out="result/lr_${lr}"
  proj="cs336-lab1-lr"
  name="lr_${lr}_bs_${FIXED_BS_FOR_LR}"
  run_one "$out" "$lr" "$FIXED_BS_FOR_LR" "$proj" "$name"
done

echo "GPU0 LR SWEEP FINISHED"
