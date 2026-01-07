import sys
import torch

print("=== Python & Environment ===")
print("Python executable:", sys.executable)
print("Python version:", sys.version)

print("\n=== PyTorch / ROCm ===")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("HIP version:", torch.version.hip)
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))

print("\n=== Core Dependencies ===")
import numpy
import einops
import einx
import jaxtyping
import regex
import submitit
import tiktoken
import tqdm
import wandb
import ty
import pytest

print("All core dependencies imported successfully!")

print("\n=== Simple GPU Tensor Test ===")
if torch.cuda.is_available():
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.randn(1024, 1024, device="cuda")
    z = x @ y
    print("GPU matmul OK, result mean:", z.mean().item())
else:
    print("CUDA not available, skipped GPU test")

print("\n=== Sanity check PASSED ===")

# Part (a) & (b): Inspecting chr(0)
null_char = chr(0)

# This shows the string representation (what you see in the interactive shell)
print(f"String representation (__repr__): {repr(null_char)}")

# This shows the printed representation
print("Printed representation: ")
print(null_char) 
print("(Note: You likely see nothing above this line because it is invisible)")

print("-" * 30)

# Part (c): Using it in text
# Concatenating the null character between two strings
combined_string = "this is a test" + chr(0) + "string"

# Inspecting the result
print(f"Combined string repr: {repr(combined_string)}")
print("Printed combined string:")
print(combined_string)

# this is the text from the local
# this is the text from the remote