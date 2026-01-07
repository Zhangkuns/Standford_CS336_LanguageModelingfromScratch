import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from cs336_basics.module import TransformerLM
from cs336_basics.loss import cross_entropy_loss
from cs336_systems.optimizer_state_sharding import ShardedOptimizer

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def get_mem_gb():
    torch.cuda.synchronize()
    # We use memory_allocated() to see the persistent state size (Weights, Grads, Opt States)
    # max_memory_allocated() would include transient activation spikes which confuses the
    # optimizer comparison.
    return torch.cuda.memory_allocated() / (1024 ** 3)

def worker(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # 1. Configure Model (GPT-2 XL)
    # Using BF16 for weights to ensure it fits in 24GB GPUs along with Optimizer
    config = {
        "vocab_size": 50257, "context_length": 256,
        "d_model": 1600, "num_layers": 48, "num_heads": 25, "d_ff": 6400,
        "rope_theta": 10000.0,
    }

    if rank == 0:
        print(f"\n--- Benchmarking: {'Sharded' if args.use_sharding else 'Standard'} Optimizer ---")

    # --- Phase 1: Initialization ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = TransformerLM(**config, device=device, dtype=torch.bfloat16)

    # Broadcast weights
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    mem_init = get_mem_gb()

    # Initialize Optimizer
    # Note: AdamW initializes states lazily (on first step), so memory won't jump yet
    if args.use_sharding:
        optimizer = ShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # --- Phase 2: Before Step (Forward + Backward) ---
    x = torch.randint(0, 50257, (1, 256), device=device)
    y = torch.randint(0, 50257, (1, 256), device=device)

    logits = model(x)
    loss = cross_entropy_loss(logits.view(-1, 50257), y.view(-1))
    loss.backward()

    mem_before_step = get_mem_gb()

    # --- Phase 3: After Step ---
    optimizer.step()

    mem_after_step = get_mem_gb()

    # Report Results (Rank 0 only)
    if rank == 0:
        print(f"1. After Model Init:   {mem_init:.2f} GB")
        print(f"2. Before Opt Step:    {mem_before_step:.2f} GB (Weights + Grads)")
        print(f"3. After Opt Step:     {mem_after_step:.2f} GB (Weights + Grads + Opt States)")

        opt_state_size = mem_after_step - mem_before_step
        print(f"-> Optimizer State Size: {opt_state_size:.2f} GB")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_sharding", action="store_true")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(worker, args=(world_size, args), nprocs=world_size, join=True)