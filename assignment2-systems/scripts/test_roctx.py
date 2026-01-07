import ctypes
import torch
import time

# 加载 libroctx64.so 库
roctx = ctypes.CDLL('/opt/rocm/lib/libroctx64.so')

# 定义函数的类型
roctx.roctxMarkA.argtypes = [ctypes.c_char_p]
roctx.roctxMarkA.restype = None

roctx.roctxRangePushA.argtypes = [ctypes.c_char_p]
roctx.roctxRangePushA.restype = None

roctx.roctxRangePop.argtypes = []  # 使用 roctxRangePop 而不是 roctxRangePopA
roctx.roctxRangePop.restype = None

# 调用 roctx API 标记程序开始
roctx.roctxMarkA(b"Start of program")

# 开始性能范围
roctx.roctxRangePushA(b"Work Range")

# 在这里进行深度学习计算

# 假设我们在做一个简单的矩阵乘法，模拟一些计算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建一个简单的深度学习模型
model = torch.nn.Linear(1024, 512).to(device)

# 随机生成输入数据
input_data = torch.randn(64, 1024).to(device)

# 前向传播
start_time = time.time()
output = model(input_data)
end_time = time.time()

# 输出计算时间
print(f"Time taken for forward pass: {end_time - start_time:.4f} seconds")

# 结束性能范围
roctx.roctxRangePop()

# 调用 roctx API 标记程序结束
roctx.roctxMarkA(b"End of program")
