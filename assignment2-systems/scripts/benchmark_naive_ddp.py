import os
import socket
import argparse
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import torch.cuda.nvtx as nvtx
# Import your components
from cs336_basics.module import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.loss import cross_entropy_loss

def nvtx_push(name):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)

def nvtx_pop():
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()

def find_free_port() -> int:
    try:
        s = socket.socket()
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port
    except:
        return 29500 # Fallback default

def setup(rank: int, world_size: int, master_addr: str, master_port: int):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    # Initialize Process Group
    # If world_size is 1, we still init to allow the code to run uniformly
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def naive_allreduce_grads(model: nn.Module, world_size: int):
    """
    Naïve DDP: Individually all-reduce every parameter's gradient.
    """
    # If only 1 GPU, communication is effectively zero, but we simulate the loop overhead
    if world_size == 1:
        return

    for p in model.parameters():
        if p.grad is not None:
            # 1. Communication: Sum gradients from all GPUs
            # This is the "Expensive" part we want to measure
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

            # 2. Average them
            p.grad.div_(world_size)

def worker(rank: int, world_size: int, args, global_inputs, global_targets):
    """
    Worker process for each GPU.
    """
    setup(rank, world_size, args.master_addr, args.master_port)
    device = torch.device(f"cuda:{rank}")

    # --- 1. Data Slicing (Simulating DDP Sharding) ---
    # Global Batch Size = args.batch_size * world_size
    # Each GPU takes a slice of size args.batch_size

    start_idx = rank * args.batch_size
    end_idx = (rank + 1) * args.batch_size

    # Move only my slice to my GPU
    x = global_inputs[start_idx:end_idx].to(device)
    y = global_targets[start_idx:end_idx].to(device)

    if rank == 0:
        print(f"Rank 0 Input Shape: {x.shape} (Slice of Global Batch)")

    # --- 2. Model Configuration (GPT-2 XL) ---
    config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "d_model": 1600,
        "num_layers": 48,
        "num_heads": 25,
        "d_ff": 6400,
        "rope_theta": 10000.0,
    }

    # Initialize Model
    # For benchmarking, we don't need real weights, just the architecture size.
    # Note: GPT-2 XL is HUGE. If you OOM on 1 GPU, use --config_small for debugging.
    if args.use_small_model:
        config.update({"d_model": 512, "num_layers": 4, "num_heads": 8, "d_ff": 2048})
        if rank == 0: print("(!) Using Small Model for debugging")

    model = TransformerLM(**config, device=device, dtype=torch.bfloat16)

    # Broadcast initial weights (Standard DDP practice, though not timed here)
    if world_size > 1:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    # --- 3. Warmup ---
    if rank == 0:
        print(f"Starting warmup ({args.warmup_steps} steps)...")

    for _ in range(args.warmup_steps):
        optimizer.zero_grad()
        logits = model(x)
        # Flatten logic matches your train.py
        loss = cross_entropy_loss(logits.view(-1, config['vocab_size']), y.view(-1))
        loss.backward()
        naive_allreduce_grads(model, world_size)
        optimizer.step()

    torch.cuda.synchronize()

    # Sync all ranks before starting timer
    if world_size > 1:
        dist.barrier()

    # --- 4. Benchmarking ---
    if rank == 0:
        print(f"Starting benchmark ({args.steps} steps)...")

    # CUDA Events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Separate timer for communication
    comm_start = torch.cuda.Event(enable_timing=True)
    comm_end = torch.cuda.Event(enable_timing=True)

    step_times = []
    comm_times = []

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

                # B. Communication
                comm_start.record()
                naive_allreduce_grads(model, world_size) # NVTX inside this function
                comm_end.record()

                # C. Update
                with nvtx.range("Optimizer Step"):
                    optimizer.step()

                end_event.record()
                torch.cuda.synchronize()

                step_times.append(start_event.elapsed_time(end_event))
                comm_times.append(comm_start.elapsed_time(comm_end))

            if rank == 0 and i % 5 == 0:
                print(f"Step {i}: Total {step_times[-1]:.1f}ms | Comm {comm_times[-1]:.1f}ms")
    # --- 5. Reporting (Rank 0 only) ---
    if rank == 0:
        avg_total = sum(step_times) / len(step_times)
        avg_comm = sum(comm_times) / len(comm_times)
        avg_comp = avg_total - avg_comm
        overhead = (avg_comm / avg_total) * 100 if avg_total > 0 else 0

        print("\n" + "="*50)
        print(f"BENCHMARK RESULTS")
        print(f"Model: {'Small' if args.use_small_model else 'GPT-2 XL'}")
        print(f"GPUs: {world_size} | Local Batch: {args.batch_size} | Global Batch: {args.batch_size * world_size}")
        print("="*50)
        print(f"Avg Total Time/Step:  {avg_total:.2f} ms")
        print(f"Avg Computation Time: {avg_comp:.2f} ms")
        print(f"Avg Comm Time:        {avg_comm:.2f} ms")
        print(f"Communication Overhead: {overhead:.2f}%")
        print("="*50)

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to measure")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps")
    parser.add_argument("--use_small_model", action="store_true", help="Use small model for debugging on small GPUs")
    args = parser.parse_args()

    # 1. Detect Devices
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found. Exiting.")
        exit(1)

    # Allow running on 1 GPU if that's all we have
    world_size = num_gpus
    print(f"Found {world_size} GPUs. Preparing data...")

    # 2. Create Global Batch (On CPU to save GPU memory during spawn)
    # Global Batch Size = Local Batch * World Size
    vocab_size = 50257
    context_length = 1024

    global_inputs = torch.randint(0, vocab_size, (world_size * args.batch_size, context_length))
    global_targets = torch.randint(0, vocab_size, (world_size * args.batch_size, context_length))

    # 3. Network Config
    args.master_addr = "127.0.0.1"
    args.master_port = find_free_port()

    # 4. Spawn Workers
    print(f"Spawning {world_size} workers...")
    mp.spawn(
        worker,
        args=(world_size, args, global_inputs, global_targets),
        nprocs=world_size,
        join=True,
    )