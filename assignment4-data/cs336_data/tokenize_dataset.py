import argparse
import os
import json
import numpy as np
import multiprocessing
from tqdm import tqdm
from transformers import AutoTokenizer

# Global tokenizer for worker processes
tokenizer = None

def init_worker():
    """Initialize tokenizer in each worker to avoid pickling overhead."""
    global tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

def process_jsonl_file(file_path):
    """
    Reads a .jsonl file, extracts 'text', tokenizes it, and returns a list of token IDs.
    """
    token_list = []
    # GPT-2 EOS token ID is 50256
    eos_id = tokenizer.eos_token_id

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        # Encode
                        tokens = tokenizer.encode(text)
                        token_list.extend(tokens)
                        # Append EOS after every document
                        token_list.append(eos_id)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    return token_list

def main():
    parser = argparse.ArgumentParser(description="Tokenize JSONL dataset (Streaming)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing deduplicated .jsonl files")
    parser.add_argument("--output_file", type=str, required=True, help="Output .bin file path")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    # 1. Gather Files
    input_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".jsonl"):
                input_files.append(os.path.join(root, file))

    input_files.sort()
    print(f"Found {len(input_files)} files to tokenize.")

    # 2. Tokenize and Write Stream
    print(f"Tokenizing to {args.output_file}...")

    total_tokens = 0

    # Open output file in Binary Write mode
    with open(args.output_file, "wb") as f_out:

        with multiprocessing.Pool(args.workers, initializer=init_worker) as pool:
            # imap returns results as they finish (preserving order)
            for file_tokens in tqdm(pool.imap(process_jsonl_file, input_files), total=len(input_files)):

                if not file_tokens:
                    continue

                # Convert chunk to uint16 numpy array
                # GPT-2 vocab is ~50k, which fits in uint16 (max 65535)
                chunk_arr = np.array(file_tokens, dtype=np.uint16)

                # Write raw bytes directly to disk
                f_out.write(chunk_arr.tobytes())

                # Update count
                total_tokens += len(file_tokens)

    print("-" * 30)
    print(f"Done! Saved to {args.output_file}")
    print(f"Total Tokens: {total_tokens}")
    print("-" * 30)

if __name__ == "__main__":
    main()