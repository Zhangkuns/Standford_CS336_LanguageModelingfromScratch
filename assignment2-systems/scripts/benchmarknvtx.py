import torch
import timeit
import argparse
import sys
import numpy as np
from cs336_basics.module import TransformerLM
from cs336_basics.loss import cross_entropy_loss
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext

def get_args():
    parser = argparse.ArgumentParser(description="Benchmark Transformer Performance")
    # Model config
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--rope_theta', type=float, default=10000.0)

    # Benchmark config
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--measure_steps', type=int, default=50)
    parser.add_argument('--mode', type=str, default='fwd_bwd', choices=['fwd', 'fwd_bwd'])
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'bf16'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def benchmark():
    args = get_args()
    device = torch.device(args.device)

    print(f"--- Benchmarking: {args.mode} ---")
    print(f"Device: {device}, Precision: {args.precision}")
    print(f"Batch: {args.batch_size}, Seq: {args.context_length}")

    # 1. Init Model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    ).to(device)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (device.type == "cuda" and args.precision == "bf16")
        else nullcontext()
    )

    # 2. Data & Optimizer
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size * args.context_length,), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 3. Warmup
    with nvtx.range("warmup"):
        print("Warming up {} steps...".format(args.warmup_steps))
    for _ in range(args.warmup_steps):
        if args.mode == 'fwd_bwd':
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                out = model(x)
                loss = cross_entropy_loss(out.view(-1, args.vocab_size), y)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                with autocast_ctx:
                    model(x)

    if device.type == 'cuda': torch.cuda.synchronize()
    # 4. Measurement
    print(f"Measuring {args.measure_steps} steps...")

    for _ in range(args.measure_steps):
        optimizer.zero_grad(set_to_none=True)

    if args.mode == 'fwd_bwd':
        with nvtx.range("forward_pass"):
            with autocast_ctx:
                logits = model(x)
        with nvtx.range("loss_computation"):
            with autocast_ctx:
                loss = cross_entropy_loss(logits.view(-1, args.vocab_size), y)
    else:
        with torch.no_grad():
            with nvtx.range("forward_pass"):
                with autocast_ctx:
                    logits = model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    if args.mode == 'fwd_bwd':
        with nvtx.range("backward_pass"):
            loss.backward()
        with nvtx.range("optimizer_step"):
            optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()


if __name__ == "__main__":
    benchmark()