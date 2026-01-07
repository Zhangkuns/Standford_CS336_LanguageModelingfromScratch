import torch
import time
import pandas as pd
import math
from cs336_basics.module import MultiHeadSelfAttention

# --- Configurations ---
BATCH_SIZE = 8
NUM_HEADS = 8  # We fix heads to 8 so we can vary d_model based on head_dim
HEAD_DIMS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def benchmark_config(head_dim, seq_len):
    if DEVICE == "cpu": return "N/A", "N/A", "N/A"

    d_model = NUM_HEADS * head_dim

    try:
        # 1. Initialize & Compile
        model = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=NUM_HEADS,
            max_seq_len=max(seq_len, 2048),
            rope_theta=10000.0,
            use_rope=True,
            device=DEVICE
        ).to(DEVICE)

        # --- COMPILE HERE ---
        # Using default backend is fine, or mode="reduce-overhead" for benchmarks
        # model = torch.compile(model)

        # 2. Inputs
        x = torch.randn(BATCH_SIZE, seq_len, d_model, device=DEVICE, requires_grad=True)
        grad_output = torch.randn(BATCH_SIZE, seq_len, d_model, device=DEVICE)

        # --- Warmup ---
        # Note: We do NOT use retain_graph=True here either.
        # We run the full cycle (Forward -> Backward) every time.
        for _ in range(10):
            # Zero grad
            for p in model.parameters(): p.grad = None
            x.grad = None

            # Forward + Backward
            out = model(x)
            out.backward(grad_output)

        torch.cuda.synchronize()

        # --- Measure Forward Only ---
        # We disable gradients for pure forward timing
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(50):
                _ = model(x)
        end_event.record()
        torch.cuda.synchronize()

        fwd_ms = start_event.elapsed_time(end_event) / 50.0

        # --- Measure Memory ---
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # One forward pass with gradients enabled to measure activation memory
        out = model(x)
        mem_bytes = torch.cuda.max_memory_allocated()
        mem_mb = mem_bytes / (1024 ** 2)

        # Clean up this specific graph before starting backward benchmark
        del out
        torch.cuda.empty_cache()

        # --- Measure Backward Time ---
        # We measure the cost of ".backward()".
        # Since we can't use retain_graph=True with compile easily,
        # we measure (Forward + Backward) and subtract (Forward).
        # This is the standard way to benchmark backward passes in complex scenarios.

        start_event.record()
        for _ in range(50):
            # We must run forward again to build a fresh graph
            out = model(x)

            # Backward
            out.backward(grad_output)

            # Zero grads
            for p in model.parameters(): p.grad = None
            x.grad = None
        end_event.record()
        torch.cuda.synchronize()

        total_ms = start_event.elapsed_time(end_event) / 50.0

        # To get "Backward Only" time, we subtract the forward time.
        # Note: The forward time here includes autograd overhead, so it's slightly
        # slower than the "no_grad" forward above, but it's a close approximation.
        # Ideally, you'd measure the "Forward with grad" time separately if you want exactness.

        # Let's measure Forward-With-Grad quickly to be precise
        start_event.record()
        for _ in range(20):
            _ = model(x)
        end_event.record()
        torch.cuda.synchronize()
        fwd_with_grad_ms = start_event.elapsed_time(end_event) / 20.0

        bwd_ms = total_ms - fwd_with_grad_ms

        return fwd_ms, bwd_ms, mem_mb

    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return "OOM", "OOM", "OOM"
        else:
            raise e

def main():
    print(f"Benchmarking MultiHeadSelfAttention on {torch.cuda.get_device_name(0)}")
    print(f"Batch Size: {BATCH_SIZE}, Num Heads: {NUM_HEADS}")
    print("-" * 65)
    print(f"{'d_head':<8} {'d_model':<8} {'seq_len':<10} {'Fwd(ms)':<10} {'Bwd(ms)':<10} {'Mem(MB)':<10}")
    print("-" * 65)

    results = []

    for head_dim in HEAD_DIMS:
        for seq_len in SEQ_LENS:
            d_model = NUM_HEADS * head_dim

            fwd, bwd, mem = benchmark_config(head_dim, seq_len)

            # Formatting
            fwd_s = f"{fwd:.2f}" if isinstance(fwd, float) else fwd
            bwd_s = f"{bwd:.2f}" if isinstance(bwd, float) else bwd
            mem_s = f"{mem:.0f}" if isinstance(mem, float) else mem

            print(f"{head_dim:<8} {d_model:<8} {seq_len:<10} {fwd_s:<10} {bwd_s:<10} {mem_s:<10}")

            results.append({
                "head_dim": head_dim,
                "d_model": d_model,
                "seq_len": seq_len,
                "fwd_ms": fwd,
                "bwd_ms": bwd,
                "mem_mb": mem
            })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("mha_benchmark_results.csv", index=False)
    print("\nResults saved to mha_benchmark_results.csv")

if __name__ == "__main__":
    main()