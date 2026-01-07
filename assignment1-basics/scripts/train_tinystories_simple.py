import time
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import tracemalloc
import cProfile
import pstats
from cs336_basics_old.bpe import train_bpe

# --- CONFIGURATION ---
# Make sure this points to your actual TinyStories file location
# It might be in /data/TinyStoriesV2-GPT4-train.txt or similar
# dataset_path = "/workspace/CS336/LAB/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "..", "data", "TinyStoriesV2-GPT4-train.txt")
print(dataset_path)

if not os.path.exists(dataset_path):
    # Fallback for checking standard locations or asking user
    print(f"Warning: {dataset_path} not found. Please edit the script with the correct path.")
    # Example: dataset_path = "/data/TinyStoriesV2-GPT4-train.txt"

vocab_size = 10000
special_tokens = ["<|endoftext|>"]

def make_out_paths():
    # 当前工作目录下的 result/
    out_dir = os.path.join(os.getcwd(), "result")
    os.makedirs(out_dir, exist_ok=True)

    # py 文件名（不带 .py）
    script_name = os.path.splitext(os.path.basename(__file__))[0] if "__file__" in globals() else "interactive"
    ts = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")

    vocab_path = os.path.join(out_dir, f"{script_name}_{ts}_vocab.json")
    merges_path = os.path.join(out_dir, f"{script_name}_{ts}_merges.txt")
    return vocab_path, merges_path

def save_results(vocab, merges):
    """Serializes vocab and merges to disk."""
    print("Saving results...")

    vocab_path, merges_path = make_out_paths()
    print("Save to:", vocab_path)
    print("Save to:", merges_path)

    # Save Vocab (Convert bytes to string representation for JSON)
    # We use latin-1 to safely preserve any byte value, or just repr()
    vocab_str = {str(k): v.decode("latin-1") for k, v in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
       json.dump(vocab_str, f, indent=2, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            f.write(f"{repr(p1)} {repr(p2)}\n")

def analyze_vocab(vocab):
    """Finds the longest token."""
    longest_id = max(vocab, key=lambda k: len(vocab[k]))
    longest_token = vocab[longest_id]

    print("-" * 30)
    print(f"Longest Token ID: {longest_id}")
    print(f"Length: {len(longest_token)} bytes")
    print(f"Content (repr): {longest_token}")
    try:
        print(f"Content (decoded): {longest_token.decode('utf-8')}")
    except:
        print("Content cannot be fully decoded as UTF-8")
    print("-" * 30)

def main():
    print(f"Starting training on {dataset_path}...")

    # --- RUN TRAINING ---
    vocab, merges = train_bpe(dataset_path, vocab_size, special_tokens,num_multiple=8)

    print(f"\nTraining Complete!")

    save_results(vocab, merges)
    analyze_vocab(vocab)

if __name__ == "__main__":
    main()