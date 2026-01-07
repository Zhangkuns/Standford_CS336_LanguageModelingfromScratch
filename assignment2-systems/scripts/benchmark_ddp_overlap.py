import os
import socket
import argparse
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

# Import your components
from cs336_basics.module import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.loss import cross_entropy_loss
from cs336_systems.ddp import DDPOverlapIndividual
import torch.cuda.nvtx as nvtx
# ==========================================
# 2. Benchmarking Logic
# ==========================================

def find_free_port() -> int:
    try:
        s = socket.socket()
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port
    except:
        return 29500

def setup(rank: int, world_size: int, master_addr: str, master_port: int):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def worker(rank: int, world_size: int, args, global_inputs, global_targets):
    setup(rank, world_size, args.master_addr, args.master_port)
    device = torch.device(f"cuda:{rank}")

    # --- Data Slicing ---
    start_idx = rank * args.batch_size
    end_idx = (rank + 1) * args.batch_size
    x = global_inputs[start_idx:end_idx].to(device)
    y = global_targets[start_idx:end_idx].to(device)

    # --- Model Config (GPT-2 XL) ---
    # d_model=1600, n_layers=48
    config = {
        "vocab_size": 50257,
        "context_length": 1024, # Reduced to 256 to prevent OOM
        "d_model": 1600,
        "num_layers": 48,
        "num_heads": 25,
        "d_ff": 6400,
        "rope_theta": 10000.0,
    }

    if args.use_small_model:
        config.update({"d_model": 512, "num_layers": 4, "num_heads": 8, "d_ff": 2048})

    # Initialize Model in BF16
    model = TransformerLM(**config, device=device, dtype=torch.bfloat16)

    # Wrap with our Overlap DDP
    # (Only wrap if we are actually distributed, else it's a no-op wrapper logic)
    if world_size > 1:
        model = DDPOverlapIndividual(model)

    # Use SGD to save memory
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    # --- Warmup ---
    if rank == 0:
        print(f"Starting warmup ({args.warmup_steps} steps)...")

    for _ in range(args.warmup_steps):
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy_loss(logits.view(-1, config['vocab_size']), y.view(-1))
        loss.backward()
        if world_size > 1: model.finish_gradient_synchronization()
        optimizer.step()

    torch.cuda.synchronize()

    if world_size > 1:
        dist.barrier()

    # --- Benchmark ---
    if rank == 0: print(f"Starting benchmark ({args.steps} steps)...")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    step_times = []

    # NVTX Range for the whole benchmark loop
    with nvtx.range("Benchmark Loop"):
        for i in range(args.steps):
            # Annotate individual steps
            with nvtx.range(f"Step {i}"):
                optimizer.zero_grad()
                start_event.record()

                # A. Computation
                with nvtx.range("Forward Pass"):
                    logits = model(x)
                    loss = cross_entropy_loss(logits.view(-1, config['vocab_size']), y.view(-1))

                with nvtx.range("Backward Pass"):
                    loss.backward()

                if world_size > 1:
                  model.finish_gradient_synchronization()

                # C. Update
                with nvtx.range("Optimizer Step"):
                    optimizer.step()

                end_event.record()
                torch.cuda.synchronize()

                step_times.append(start_event.elapsed_time(end_event))

            if rank == 0 and i % 5 == 0:
                print(f"Step {i}: {step_times[-1]:.2f} ms")

    # --- Reporting ---
    if rank == 0:
        avg_time = sum(step_times) / len(step_times)
        print("\n" + "="*50)
        print(f"DDP OVERLAP INDIVIDUAL RESULTS")
        print(f"GPUs: {world_size} | Model: {'Small' if args.use_small_model else 'GPT-2 XL'}")
        print("="*50)
        print(f"Avg Time Per Iteration: {avg_time:.2f} ms")
        print("="*50)

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--use_small_model", action="store_true")
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    world_size = num_gpus

    if world_size < 2:
        print("Warning: Running on 1 GPU. Communication overhead will be 0.")
        world_size = 1

    # Create Dummy Data
    vocab_size = 50257
    context_length = 1024 # Matches config above
    global_inputs = torch.randint(0, vocab_size, (world_size * args.batch_size, context_length))
    global_targets = torch.randint(0, vocab_size, (world_size * args.batch_size, context_length))

    args.master_addr = "127.0.0.1"
    args.master_port = find_free_port()

    print(f"Spawning {world_size} workers...")
    mp.spawn(worker, args=(world_size, args, global_inputs, global_targets), nprocs=world_size, join=True)