import torch
import timeit
import argparse
import sys
import numpy as np
from cs336_basics.module import TransformerLM
from cs336_basics.loss import cross_entropy_loss

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
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
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

    if args.precision == 'fp16': model = model.half()
    elif args.precision == 'bf16': model = model.bfloat16()

    # 2. Data & Optimizer
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size * args.context_length,), device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 3. Warmup
    print("Warming up {} steps...".format(args.warmup_steps))
    for _ in range(args.warmup_steps):
        if args.mode == 'fwd_bwd':
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = cross_entropy_loss(out.view(-1, args.vocab_size), y)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model(x)

    if device.type == 'cuda': torch.cuda.synchronize()

    # 4. Measurement
    print(f"Measuring {args.measure_steps} steps...")

    fwd_times = []
    bwd_times = []

    for _ in range(args.measure_steps):
        optimizer.zero_grad(set_to_none=True)

        # --- Measure Forward ---
        t0 = timeit.default_timer()

        # Depending on mode, we either track gradients or not
        if args.mode == 'fwd_bwd':
            logits = model(x)
            loss = cross_entropy_loss(logits.view(-1, args.vocab_size), y)
        else:
            with torch.no_grad():
                logits = model(x)

        if device.type == 'cuda': torch.cuda.synchronize()
        t1 = timeit.default_timer()
        fwd_times.append(t1 - t0)

        # --- Measure Backward (if applicable) ---
        if args.mode == 'fwd_bwd':
            t2 = timeit.default_timer()
            loss.backward()
            optimizer.step()

            if device.type == 'cuda': torch.cuda.synchronize()
            t3 = timeit.default_timer()
            bwd_times.append(t3 - t2)

    # 5. Results
    avg_fwd = np.mean(fwd_times)
    print(f"\nResults:")
    print(f"Forward Pass:  {avg_fwd*1000:.2f} ms")

    total_time = avg_fwd

    if args.mode == 'fwd_bwd':
        avg_bwd = np.mean(bwd_times)
        total_time += avg_bwd
        print(f"Backward Pass: {avg_bwd*1000:.2f} ms")

    print("-" * 20)
    print(f"Total Step:    {total_time*1000:.2f} ms")
    print(f"Throughput:    {(args.batch_size * args.context_length) / total_time:.2f} tokens/sec")

if __name__ == "__main__":
    benchmark()