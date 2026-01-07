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

LRS=(1e-3 1e-4 1e-2 1e-5)
BATCHES=(16 32 64)

FIXED_BS_FOR_LR=32
FIXED_LR_FOR_BS=1e-4

run_one () {
  local gpu="$1"
  local outdir="$2"
  local lr="$3"
  local bs="$4"
  local project="$5"
  local runname="$6"

  mkdir -p "$outdir"
  local logfile="$outdir/train_$(date '+%Y-%m-%d_%H-%M-%S').log"

  HIP_VISIBLE_DEVICES="$gpu" stdbuf -oL -eL \
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

# GPU0
(
  for lr in "${LRS[@]}"; do
    out="result/lr_${lr}"
    proj="cs336-lab1-lr"
    name="lr_${lr}_bs_${FIXED_BS_FOR_LR}"
    run_one 0 "$out" "$lr" "$FIXED_BS_FOR_LR" "$proj" "$name"
  done
) 2>&1 | sed -u 's/^/[GPU0] /' &
pid0=$!

# GPU1
(
  for bs in "${BATCHES[@]}"; do
    out="result/bs_${bs}"
    proj="cs336-lab1-bs"
    name="lr_${FIXED_LR_FOR_BS}_bs_${bs}"
    run_one 1 "$out" "$FIXED_LR_FOR_BS" "$bs" "$proj" "$name"
  done
) 2>&1 | sed -u 's/^/[GPU1] /' &
pid1=$!

wait "$pid0"
wait "$pid1"
echo "ALL SWEEPS FINISHED"

