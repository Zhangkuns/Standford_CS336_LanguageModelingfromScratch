import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# --- Configuration ---
BACKENDS = [("gloo", "cpu"), ("nccl", "cuda")]
# Sizes in Bytes: 1MB, 10MB, 100MB, 1GB
SIZES_MB = [1, 10, 100, 1000]
SIZES_BYTES = [s * 1024 * 1024 for s in SIZES_MB]
WORLD_SIZES = [2, 4, 6]
WARMUP_ITERS = 5
MEASURE_ITERS = 10
MASTER_ADDR = '127.0.0.1'
MASTER_PORT = '29500'

def run_benchmark(rank, world_size, backend, device_type, results_dict):
    """
    Worker function for multiprocessing.
    """
    # 1. Setup Environment
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT

    # Map rank to GPU if using NCCL
    if device_type == "cuda":
        # Ensure we don't index out of bounds if user has fewer GPUs than requested
        # (Though the main loop checks this, it's safe practice)
        if rank < torch.cuda.device_count():
            torch.cuda.set_device(rank)
        else:
            print(f"Rank {rank} cannot set CUDA device: only {torch.cuda.device_count()} GPUs available.")

    # 2. Init Process Group
    try:
        dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timedelta(seconds=60))
    except Exception as e:
        print(f"Rank {rank} failed to init process group: {e}")
        return

    # 3. Iterate over Data Sizes
    for size_bytes, size_mb in zip(SIZES_BYTES, SIZES_MB):
        # Create Data
        # float32 = 4 bytes. Num elements = Bytes / 4
        num_elements = int(size_bytes / 4)

        try:
            # Allocate tensor
            tensor = torch.randn(num_elements, dtype=torch.float32)
            if device_type == "cuda":
                tensor = tensor.cuda()

            # --- Warmup ---
            for _ in range(WARMUP_ITERS):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

            # Sync before timing
            if device_type == "cuda":
                torch.cuda.synchronize()
            else:
                dist.barrier()

            # --- Measurement ---
            start_time = time.time()

            for _ in range(MEASURE_ITERS):
                # async_op=False is default, but for CUDA we need explicit sync for timing
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

            if device_type == "cuda":
                torch.cuda.synchronize()
            else:
                # Gloo is CPU synchronous, but barrier ensures all ranks finished
                dist.barrier()

            end_time = time.time()

            # Average time per iteration (in ms)
            avg_time_sec = (end_time - start_time) / MEASURE_ITERS
            avg_time_ms = avg_time_sec * 1000

            # Calculate effective Bandwidth (Algorithmic Bandwidth)
            # Formula for Ring AllReduce: 2 * (N-1)/N * Size / Time
            # Here we just report raw throughput: Size / Time
            throughput_gbps = (size_bytes / 1e9) / avg_time_sec

            # Gather results to Rank 0
            # We use all_gather_object to collect python dicts
            local_result = {
                "rank": rank,
                "time_ms": avg_time_ms
            }
            gather_list = [None for _ in range(world_size)]
            dist.all_gather_object(gather_list, local_result)

            # Report from Rank 0
            if rank == 0:
                # Average time across all ranks
                total_avg_ms = sum(r["time_ms"] for r in gather_list) / world_size

                results_dict.append({
                    "Backend": f"{backend.upper()}/{device_type.upper()}",
                    "Processes": world_size,
                    "Size_MB": size_mb,
                    "Latency_ms": total_avg_ms,
                    "Throughput_GB_s": (size_bytes / 1e9) / (total_avg_ms / 1000)
                })

        except RuntimeError as e:
            if rank == 0:
                print(f"OOM or Error on {size_mb}MB: {e}")
            break

    # 4. Cleanup
    dist.destroy_process_group()

def main():
    results = []
    # Using a Manager list to retrieve data from spawned processes
    manager = mp.Manager()
    shared_results = manager.list()

    print(f"{'Backend':<15} {'World Size':<12} {'Size (MB)':<12} {'Time (ms)':<15} {'GB/s':<10}")
    print("-" * 65)

    for backend, device in BACKENDS:
        for ws in WORLD_SIZES:
            # Check resource availability
            if device == "cuda" and torch.cuda.device_count() < ws:
                print(f"Skipping {backend}/{device} with {ws} processes (Only {torch.cuda.device_count()} GPUs found)")
                continue

            # print(f"Running {backend} on {device} with {ws} processes...")

            # Spawn processes
            mp.spawn(
                run_benchmark,
                args=(ws, backend, device, shared_results),
                nprocs=ws,
                join=True
            )

            # Print latest results added to shared list
            # We filter for the current config to print them cleanly
            current_config_results = [r for r in shared_results
                                      if r["Backend"] == f"{backend.upper()}/{device.upper()}"
                                      and r["Processes"] == ws]

            # Clear shared list for next batch to avoid reprocessing (optional, but cleaner logic here)
            # Actually, let's just dump the new ones to our local list and print
            for r in current_config_results:
                if r not in results:
                    print(f"{r['Backend']:<15} {r['Processes']:<12} {r['Size_MB']:<12} {r['Latency_ms']:<15.4f} {r['Throughput_GB_s']:<10.4f}")
                    results.append(r)

    # --- Generate Table/Plots ---
    df = pd.DataFrame(results)
    print("\n--- Final Summary Table ---")
    print(df.to_string(index=False))

    # Save to CSV
    df.to_csv("benchmark_dist_results.csv", index=False)
    print("\nResults saved to 'benchmark_dist_results.csv'")

    # Optional: Plotting
    try:
        plot_results(df)
        print("Plots saved to 'benchmark_dist_plot.png'")
    except Exception as e:
        print(f"Skipping plotting: {e}")

def plot_results(df):
    # Plotting Latency vs Size for different configs
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Log-Log Plot of Latency vs Size
    ax = axes[0]
    for backend in df['Backend'].unique():
        subset = df[df['Backend'] == backend]
        for ws in subset['Processes'].unique():
            data = subset[subset['Processes'] == ws]
            ax.plot(data['Size_MB'], data['Latency_ms'], marker='o', label=f"{backend} (N={ws})")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Data Size (MB)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("All-Reduce Latency (Log-Log)")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)

    # 2. Throughput vs Size
    ax = axes[1]
    for backend in df['Backend'].unique():
        subset = df[df['Backend'] == backend]
        for ws in subset['Processes'].unique():
            data = subset[subset['Processes'] == ws]
            ax.plot(data['Size_MB'], data['Throughput_GB_s'], marker='x', label=f"{backend} (N={ws})")

    ax.set_xscale('log')
    ax.set_xlabel("Data Size (MB)")
    ax.set_ylabel("Throughput (GB/s)")
    ax.set_title("Effective Throughput")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig("benchmark_dist_plot.png")

if __name__ == "__main__":
    main()