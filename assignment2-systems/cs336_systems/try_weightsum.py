import torch
from cs336_systems.weight_sum import WeightedSumFunc

f_weightedsum = WeightedSumFunc.apply

# 设置随机种子，方便复现
torch.manual_seed(0)

# 构造一个很小的输入
x = torch.randn(4, 128, device="cuda", requires_grad=True)
w = torch.randn(128, device="cuda", requires_grad=True)

# 调用你写的自定义函数
y = f_weightedsum(x, w)

print("y:", y)
print("y.shape:", y.shape)
print("y.grad_fn:", y.grad_fn)

# 随便构造一个 loss
loss = y.sum()

# 反向传播
loss.backward()

print("\nx.grad:")
print(x.grad)

print("\nw.grad:")
print(w.grad)
