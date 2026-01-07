import os
import socket
import argparse
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

# Import Model
from cs336_basics.module import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.loss import cross_entropy_loss
# Import your DDP class
from cs336_systems.ddp_bucket import DDPBucketed

def find_free_port() -> int:
    try:
        s = socket.socket(); s.bind(("", 0)); port = s.getsockname()[1]; s.close()
        return port
    except: return 29500

def setup(rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def worker(rank: int, world_size: int, bucket_size_mb: float, args, global_inputs, global_targets, return_dict):
    """
    Runs ONE benchmark for ONE bucket size, then exits to free memory.
    """
    try:
        setup(rank, world_size, args.master_addr, args.master_port)
        device = torch.device(f"cuda:{rank}")

        # Data Slice
        start_idx = rank * args.batch_size
        end_idx = (rank + 1) * args.batch_size
        x = global_inputs[start_idx:end_idx].to(device)
        y = global_targets[start_idx:end_idx].to(device)

        # Model Config (GPT-2 XL)
        config = {
            "vocab_size": 50257, "context_length": 256, "d_model": 1600,
            "num_layers": 48, "num_heads": 25, "d_ff": 6400, "rope_theta": 10000.0,
        }
        if args.use_small_model:
            config.update({"d_model": 512, "num_layers": 4, "num_heads": 8, "d_ff": 2048})

        # Initialize Model (BF16 to save memory)
        model_base = TransformerLM(**config, device=device, dtype=torch.bfloat16)

        # Wrap with DDP Bucketed
        model = DDPBucketed(model_base, bucket_size_mb=bucket_size_mb)

        # Use SGD to save optimizer state memory
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

        # Warmup
        if rank == 0: print(f"  > Warmup...")
        for _ in range(args.warmup_steps):
            optimizer.zero_grad()
            logits = model(x)
            loss = cross_entropy_loss(logits.view(-1, config['vocab_size']), y.view(-1))
            loss.backward()
            model.finish_gradient_synchronization()
            optimizer.step()

        torch.cuda.synchronize()
        if world_size > 1: dist.barrier()

        # Measure
        if rank == 0: print(f"  > Measuring...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        optimizer.zero_grad()
        start_event.record()

        for _ in range(args.steps):
            logits = model(x)
            loss = cross_entropy_loss(logits.view(-1, config['vocab_size']), y.view(-1))
            loss.backward()
            model.finish_gradient_synchronization()
            optimizer.step()
            optimizer.zero_grad()

        end_event.record()
        torch.cuda.synchronize()

        total_time = start_event.elapsed_time(end_event)
        avg_time = total_time / args.steps

        # Store result in shared dict
        if rank == 0:
            print(f"  > Done. Time: {avg_time:.2f} ms")
            return_dict[bucket_size_mb] = avg_time

    finally:
        cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--use_small_model", action="store_true")
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("Need 2 GPUs.")
        exit(1)

    world_size = num_gpus
    vocab_size = 50257

    # Create Data on CPU
    global_inputs = torch.randint(0, vocab_size, (world_size * args.batch_size, 256))
    global_targets = torch.randint(0, vocab_size, (world_size * args.batch_size, 256))

    bucket_sizes = [1, 10, 100, 1000]
    final_results = {}

    print(f"Benchmarking DDP Bucketed (GPUs={world_size})")
    print(f"Model: {'Small' if args.use_small_model else 'GPT-2 XL (BF16)'}")
    print("-" * 40)

    # --- Loop in Main Process ---
    # We spawn fresh processes for EACH bucket size.
    # This guarantees 100% memory cleanup between runs.

    for mb in bucket_sizes:
        print(f"\n--- Testing Bucket Size: {mb} MB ---")

        # Get a fresh port to avoid "Address already in use" between quick restarts
        args.master_addr = "127.0.0.1"
        args.master_port = find_free_port()

        manager = mp.Manager()
        return_dict = manager.dict()

        mp.spawn(
            worker,
            args=(world_size, float(mb), args, global_inputs, global_targets, return_dict),
            nprocs=world_size,
            join=True
        )

        # Collect result
        if mb in return_dict:
            final_results[mb] = return_dict[mb]

    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("-" * 40)
    for mb in bucket_sizes:
        ms = final_results.get(mb, float('nan'))
        print(f"Bucket {mb:<4} MB: {ms:.2f} ms")