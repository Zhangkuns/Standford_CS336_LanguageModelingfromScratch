from datasets import load_dataset

ds = load_dataset("miromind-ai/MiroMind-M1-SFT-719K", split="train")  # 这时 ds 是 Dataset
print(type(ds))
print("columns:", ds.column_names)
print("features:", ds.features)

print("\n=== example 0 ===")
print(ds[0])

for i in range(3):
    ex = ds[i]
    print(f"\n=== example {i} ===")
    print("id:", ex["id"])
    print("question:", ex["question"])
    print("response:", ex["response"])
