import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import time

# Imports
from cs336_basics.module import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.loss import cross_entropy_loss
from cs336_systems.ddp_bucket import DDPBucketed # Assuming you saved DDPBucketed here
from cs336_systems.optimizer_state_sharding import ShardedOptimizer

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29505'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def worker(rank, world_size, args, return_dict):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # 1. Config (GPT-2 XL)
    config = {
        "vocab_size": 50257, "context_length": 256,
        "d_model": 1600, "num_layers": 48, "num_heads": 25, "d_ff": 6400,
        "rope_theta": 10000.0,
    }

    # 2. Model (BF16)
    # We use DDPBucketed for both to ensure gradient sync time is constant
    model_base = TransformerLM(**config, device=device, dtype=torch.bfloat16)
    model = DDPBucketed(model_base, bucket_size_mb=10)

    # 3. Optimizer selection
    if args.use_sharding:
        # Wraps AdamW, handles sharding + broadcast
        optimizer = ShardedOptimizer(model.parameters(), AdamW, lr=1e-4)
    else:
        # Standard AdamW
        optimizer = AdamW(model.parameters(), lr=1e-4)

    # 4. Data
    x = torch.randint(0, 50257, (args.batch_size, 256), device=device)
    y = torch.randint(0, 50257, (args.batch_size, 256), device=device)

    # 5. Warmup
    if rank == 0: print("  Warmup...")
    for _ in range(args.warmup_steps):
        optimizer.zero_grad()
        loss = cross_entropy_loss(model(x).view(-1, 50257), y.view(-1))
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()

    torch.cuda.synchronize()
    dist.barrier()

    # 6. Measure
    if rank == 0: print("  Measuring...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(args.steps):
        optimizer.zero_grad()
        # Forward
        loss = cross_entropy_loss(model(x).view(-1, 50257), y.view(-1))
        # Backward + Grad Sync
        loss.backward()
        model.finish_gradient_synchronization()
        # Optimizer Step + (Optional Weight Sync)
        optimizer.step()

    end_event.record()
    torch.cuda.synchronize()

    avg_time = start_event.elapsed_time(end_event) / args.steps

    if rank == 0:
        mode = "Sharded" if args.use_sharding else "Standard"
        print(f"  {mode} Time: {avg_time:.2f} ms")
        return_dict[mode] = avg_time

    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=5)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Requires 2 GPUs for comparison.")
        return

    manager = mp.Manager()
    return_dict = manager.dict()

    print(">>> Benchmarking Standard AdamW...")
    args.use_sharding = False
    mp.spawn(worker, args=(world_size, args, return_dict), nprocs=world_size, join=True)

    print("\n>>> Benchmarking Sharded AdamW...")
    args.use_sharding = True
    mp.spawn(worker, args=(world_size, args, return_dict), nprocs=world_size, join=True)

    print("\n=== FINAL COMPARISON ===")
    std = return_dict['Standard']
    sharded = return_dict['Sharded']
    print(f"Standard: {std:.2f} ms")
    print(f"Sharded:  {sharded:.2f} ms")
    print(f"Difference: {sharded - std:.2f} ms ({((sharded - std)/std)*100:.1f}%)")

if __name__ == "__main__":
    main()