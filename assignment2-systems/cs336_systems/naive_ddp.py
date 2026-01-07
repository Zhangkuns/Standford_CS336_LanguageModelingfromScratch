import os
import socket
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn


def find_free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def setup(rank: int, world_size: int, backend: str, master_addr: str, master_port: int):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def broadcast_model_params(model: nn.Module, src: int = 0):
    # 所有 rank 都必须调用
    for p in model.parameters():
        dist.broadcast(p.data, src=src)
    for b in model.buffers():
        dist.broadcast(b.data, src=src)


def allreduce_grads(model: nn.Module, world_size: int):
    # naïve DDP：对每个参数梯度 all-reduce
    for p in model.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)  # 平均梯度


class ToyModel(nn.Module):
    def __init__(self, in_dim=16, hidden=32, out_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_toy_batches(steps: int, batch_size: int, in_dim: int, out_dim: int, seed: int):
    """
    生成固定的“全局 batch 序列”，保证 single-process 与 distributed 使用完全相同的 batch。
    """
    g = torch.Generator().manual_seed(seed)
    xb = torch.randn(steps, batch_size, in_dim, generator=g)
    # 用一个固定的“真值线性映射”生成目标（回归），确保有学习信号
    w_true = torch.randn(in_dim, out_dim, generator=g)
    yb = xb @ w_true + 0.01 * torch.randn(steps, batch_size, out_dim, generator=g)
    return xb, yb


def run_reference_single_process(init_state, steps, batch_size, in_dim, out_dim, lr, seed, device):
    """
    单进程“等效大 batch”训练：每步用完整 batch_size 的数据。
    """
    model = ToyModel(in_dim=in_dim, hidden=32, out_dim=out_dim).to(device)
    model.load_state_dict(init_state, strict=True)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    xb, yb = make_toy_batches(steps, batch_size, in_dim, out_dim, seed=seed)

    for t in range(steps):
        x_full = xb[t].to(device)
        y_full = yb[t].to(device)

        opt.zero_grad(set_to_none=True)
        pred = model(x_full)
        loss = F.mse_loss(pred, y_full, reduction="mean")
        loss.backward()
        opt.step()

    return model.state_dict()


def worker(rank: int, world_size: int, backend: str, device: str, master_addr: str, master_port: int):
    setup(rank, world_size, backend, master_addr, master_port)

    # 设备设置
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda 但 torch.cuda.is_available() 为 False")
        # 常见做法：每个 rank 绑定不同 GPU；GPU 不够就取模（仅用于跑通，不建议拿它做性能结论）
        torch.cuda.set_device(rank % max(torch.cuda.device_count(), 1))
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    # 训练超参（你也可以改成 argparse）
    steps = 20
    batch_size = 64
    in_dim = 16
    out_dim = 8
    lr = 1e-1
    data_seed = 1234

    assert batch_size % world_size == 0, "batch_size 必须能整除 world_size"
    local_bs = batch_size // world_size

    # 初始化模型（每个 rank 随机初始化可能不同）
    model = ToyModel(in_dim=in_dim, hidden=32, out_dim=out_dim).to(dev)

    # 保存 rank0 的初始权重作为“真起点”，并广播给所有 rank
    if rank == 0:
        init_state = deepcopy(model.state_dict())
    else:
        init_state = None

    # 用 broadcast_object_list 让 rank0 的 init_state 发给其它 rank（这样 reference 可用同一份起点）
    obj_list = [init_state]
    dist.broadcast_object_list(obj_list, src=0)
    init_state = obj_list[0]

    model.load_state_dict(init_state, strict=True)
    broadcast_model_params(model, src=0)

    # 注意：optimizer 最好在 broadcast 之后创建
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    # 每个 rank 生成同一份全局 batch 序列（同 seed）
    xb, yb = make_toy_batches(steps, batch_size, in_dim, out_dim, seed=data_seed)

    # ---- Distributed training (naïve DDP) ----
    for t in range(steps):
        x_full = xb[t].to(dev)
        y_full = yb[t].to(dev)

        # shard：每个 rank 拿 disjoint 的 n/d 样本
        start = rank * local_bs
        end = (rank + 1) * local_bs
        x = x_full[start:end]
        y = y_full[start:end]

        opt.zero_grad(set_to_none=True)
        pred = model(x)
        # reduction=mean 很关键：这样 all-reduce 平均后等价于全局 batch 的 mean loss 梯度
        loss = F.mse_loss(pred, y, reduction="mean")
        loss.backward()

        allreduce_grads(model, world_size)
        opt.step()

    # ---- Verification on rank0: compare to single-process ----
    # 分布式训练完成后，每个 rank 的参数应一致，拿 rank0 来比对就行
    dist.barrier()
    if rank == 0:
        ref_state = run_reference_single_process(
            init_state=init_state,
            steps=steps,
            batch_size=batch_size,
            in_dim=in_dim,
            out_dim=out_dim,
            lr=lr,
            seed=data_seed,
            device=dev,
        )

        # 计算最大绝对误差
        max_abs = 0.0
        dist_state = model.state_dict()
        for k in dist_state.keys():
            a = dist_state[k].detach().cpu()
            b = ref_state[k].detach().cpu()
            max_abs = max(max_abs, (a - b).abs().max().item())

        print(f"[rank0] max |DDP - single| = {max_abs:.6e}")
        if max_abs < 1e-6:
            print("[rank0] ✅ Match (within tolerance)")
        else:
            print("[rank0] ❌ Mismatch (check scaling / sharding / seeds)")

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    backend = "gloo"   # GPU/NCCL(含ROCm/RCCL)用 "nccl"
    device = "cpu"     # GPU用 "cuda"

    master_addr = "127.0.0.1"
    master_port = find_free_port()  # 避免 29500 被占用导致 EADDRINUSE

    mp.spawn(
        worker,
        args=(world_size, backend, device, master_addr, master_port),
        nprocs=world_size,
        join=True,
    )
