import torch
import triton
import triton.testing
import math
import pandas as pd
import sys

# Import your Triton implementation
from cs336_systems.flashattention_triton import FlashAttentionTritonFunc

# --- 1. Naive PyTorch Implementation ---
def pytorch_naive_attention(q, k, v, is_causal=True):
    """
    Standard PyTorch implementation using matmul + softmax.
    """
    d_head = q.shape[-1]
    scale = 1.0 / math.sqrt(d_head)

    # 1. Scores: (B, H, S, S)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # 2. Causal Mask
    if is_causal:
        seq_len = q.shape[2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool)).view(1, 1, seq_len, seq_len)
        scores = scores.masked_fill(~mask, float('-inf'))

    # 3. Softmax
    probs = torch.softmax(scores, dim=-1)

    # 4. Output
    output = torch.matmul(probs, v)
    return output

# --- 2. Triton Wrapper (THE FIX) ---
def triton_flash_wrapper(q, k, v, is_causal=True):
    """
    Wraps the autograd.Function.apply to handle arguments correctly.
    'apply' often requires positional args only.
    """
    return FlashAttentionTritonFunc.apply(q, k, v, is_causal)

# --- 3. Benchmarking Helper ---
def benchmark_op(op_name, func_factory, q, k, v, dout):
    # Clear cache before starting
    torch.cuda.empty_cache()

    try:
        # Define Forward Closure
        def forward_fn():
            return func_factory(q, k, v, is_causal=True)

        # Measure Forward
        fwd_ms = triton.testing.do_bench(forward_fn, return_mode="median")

        # Define Forward + Backward Closure
        def fwd_bwd_fn():
            if q.grad is not None: q.grad = None
            if k.grad is not None: k.grad = None
            if v.grad is not None: v.grad = None

            out = func_factory(q, k, v, is_causal=True)
            out.backward(dout)

        # Measure Total
        total_ms = triton.testing.do_bench(fwd_bwd_fn, return_mode="median")

        # Derived Backward time
        bwd_ms = total_ms - fwd_ms

        return fwd_ms, bwd_ms, total_ms

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "OOM", "OOM", "OOM"
    except Exception as e:
        return "ERR", "ERR", str(e)[:20]

# --- 4. Main Sweep Loop ---
def main():
    device = torch.device("cuda")

    # Sweep configurations
    BATCH_SIZE = 1
    # Powers of 2 from 128 to 65536
    SEQ_LENS = [2**i for i in range(7, 17)]
    # Powers of 2 from 16 to 128
    HEAD_DIMS = [16, 32, 64, 128]
    # Precisions
    DTYPES = [torch.float32, torch.bfloat16]

    results = []

    print(f"{'Impl':<10} | {'Dtype':<10} | {'Dim':<5} | {'Seq':<6} | {'Fwd(ms)':<10} | {'Bwd(ms)':<10} | {'Tot(ms)':<10}")
    print("-" * 80)

    for dtype in DTYPES:
        dtype_str = "fp32" if dtype == torch.float32 else "bf16"

        for d in HEAD_DIMS:
            for seq in SEQ_LENS:
                # Setup Inputs
                # shape: (B, H, S, D) - Using Batch=1, Heads=1 for atomic benchmark
                shape = (BATCH_SIZE, 1, seq, d)

                try:
                    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                    dout = torch.randn(shape, device=device, dtype=dtype)
                except torch.cuda.OutOfMemoryError:
                    # If we can't even alloc inputs, skip
                    print(f"Skipping {d}x{seq} due to input alloc OOM")
                    continue

                # --- Run Naive PyTorch ---
                # Heuristic: Skip PyTorch if seq len is huge to save time on OOMs
                if seq <= 8192:
                    py_fwd, py_bwd, py_tot = benchmark_op(
                        "PyTorch", pytorch_naive_attention, q, k, v, dout
                    )
                else:
                    py_fwd, py_bwd, py_tot = "OOM", "OOM", "OOM"

                res_py = {
                    "Implementation": "PyTorch",
                    "Precision": dtype_str,
                    "Head Dim": d,
                    "Seq Len": seq,
                    "Fwd Latency": py_fwd,
                    "Bwd Latency": py_bwd,
                    "Total Latency": py_tot
                }
                results.append(res_py)
                print(f"PyTorch    | {dtype_str:<10} | {d:<5} | {seq:<6} | {str(py_fwd):<10} | {str(py_bwd):<10} | {str(py_tot):<10}")

                # --- Run Triton FlashAttention ---
                # Use the Wrapper function here!
                q.grad = None; k.grad = None; v.grad = None

                tri_fwd, tri_bwd, tri_tot = benchmark_op(
                    "Triton", triton_flash_wrapper, q, k, v, dout
                )

                res_tri = {
                    "Implementation": "Triton-FA2",
                    "Precision": dtype_str,
                    "Head Dim": d,
                    "Seq Len": seq,
                    "Fwd Latency": tri_fwd,
                    "Bwd Latency": tri_bwd,
                    "Total Latency": tri_tot
                }
                results.append(res_tri)
                print(f"Triton     | {dtype_str:<10} | {d:<5} | {seq:<6} | {str(tri_fwd):<10} | {str(tri_bwd):<10} | {str(tri_tot):<10}")

    # Save
    df = pd.DataFrame(results)
    df.to_csv("flash_attention_benchmark.csv", index=False)
    print("\nBenchmark saved to flash_attention_benchmark.csv")

if __name__ == "__main__":
    main()