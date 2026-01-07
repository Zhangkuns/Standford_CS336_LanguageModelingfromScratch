import torch
import argparse
from contextlib import nullcontext
from cs336_basics.module import TransformerLM
from cs336_basics.loss import cross_entropy_loss
import torch.cuda.nvtx as nvtx

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

    # Memory profiling config
    parser.add_argument('--memory_profile', action='store_true',
                        help="Enable CUDA memory history + dump snapshot for memory_viz")
    parser.add_argument('--memory_out', type=str, default='mem_snapshot.pickle',
                        help="Output path for memory snapshot pickle")

    return parser.parse_args()

def benchmark():
    args = get_args()
    device = torch.device(args.device)

    print(f"--- Benchmarking: {args.mode} ---")
    print(f"Device: {device}, Precision: {args.precision}")
    print(f"Batch: {args.batch_size}, Seq: {args.context_length}")

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

    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size * args.context_length,), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # ---------------- Warmup ----------------
    print(f"Warming up {args.warmup_steps} steps...")
    with nvtx.range("warmup"):
        for _ in range(args.warmup_steps):
            optimizer.zero_grad(set_to_none=True)
            if args.mode == 'fwd_bwd':
                with autocast_ctx:
                    out = model(x)
                    loss = cross_entropy_loss(out.view(-1, args.vocab_size), y)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    with autocast_ctx:
                        _ = model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # ------------- Memory profiler setup -------------
    # 关键：只在 measurement 前清零峰值，避免 warmup 污染
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    if args.memory_profile and device.type == "cuda":
        # 记录 CUDA allocator 的分配/释放历史（供 memory_viz 做 timeline）
        torch.cuda.memory._record_memory_history(
            # max_entries 适当大一点，否则会截断；太大也会慢
            max_entries=200000,
        )

    # ---------------- Measurement ----------------
    print(f"Measuring {args.measure_steps} steps...")
    with nvtx.range("measurement"):
        for _ in range(args.measure_steps):
            optimizer.zero_grad(set_to_none=True)

            if args.mode == 'fwd_bwd':
                with nvtx.range("forward_pass"):
                    with autocast_ctx:
                        logits = model(x)

                with nvtx.range("loss_computation"):
                    with autocast_ctx:
                        loss = cross_entropy_loss(logits.view(-1, args.vocab_size), y)

                with nvtx.range("backward_pass"):
                    loss.backward()

                with nvtx.range("optimizer_step"):
                    optimizer.step()

            else:
                with torch.no_grad():
                    with nvtx.range("forward_pass"):
                        with autocast_ctx:
                            _ = model(x)

            if device.type == 'cuda':
                torch.cuda.synchronize()

    # ---------------- Report peak memory ----------------
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
        print(f"\n[CUDA peak] max_memory_allocated: {peak_alloc:.2f} MB")
        print(f"[CUDA peak] max_memory_reserved:  {peak_reserved:.2f} MB")

    # ---------------- Dump snapshot for memory_viz ----------------
    if args.memory_profile and device.type == "cuda":
        # snapshot：给 memory_viz 用（Active Memory Timeline 就靠这个）
        # 获取并保存CUDA内存快照
        torch.cuda.memory._dump_snapshot(args.memory_out)
        # 关掉历史记录，避免影响后续运行
        torch.cuda.memory._record_memory_history(enabled=False)

        print(f"\nSaved memory snapshot to: {args.memory_out}")
        print("Open https://pytorch.org/memory_viz and upload this .pickle file.")

if __name__ == "__main__":
    benchmark()
