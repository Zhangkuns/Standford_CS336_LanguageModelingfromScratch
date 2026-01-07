import argparse
import os
import numpy as np
import multiprocessing
from typing import List
from cs336_basics_old.tokenizer import Tokenizer
from cs336_basics_old.bpe import find_chunk_boundaries

def get_args():
    parser = argparse.ArgumentParser(description="Encode a dataset using a trained BPE tokenizer.")
    parser.add_argument("--input", type=str, required=True, help="Path to input text file or directory")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocab.json")
    parser.add_argument("--merges", type=str, required=True, help="Path to merges.txt")
    parser.add_argument("--output", type=str, required=True, help="Path to output .npy file")
    parser.add_argument("--num_threads", type=int, default=os.cpu_count(), help="Number of parallel processes")
    parser.add_argument("--special_token", type=str, default="<|endoftext|>", help="Special token used for splitting")
    return parser.parse_args()

def _worker_encode_chunk(args):
    """
    Worker function to read a specific chunk of the file and tokenize it.
    """
    file_path, start, end, vocab_path, merges_path, special_token = args

    # Re-initialize tokenizer inside worker to avoid pickling issues with Regex objects
    # Pass None for special_tokens list in constructor if we handle them via splitting logic,
    # OR pass them if the Tokenizer handles them.
    # Your Tokenizer class handles special tokens via regex splitting internally.
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=[special_token])

    with open(file_path, 'rb') as f:
        f.seek(start)
        # Read text chunk
        text_bytes = f.read(end - start)
        text = text_bytes.decode("utf-8", errors="ignore")

    # Encode
    ids = tokenizer.encode(text)
    return ids

def main():
    args = get_args()

    print(f"Loading Tokenizer info from {args.vocab}...")
    # Verify tokenizer paths exist
    if not os.path.exists(args.vocab) or not os.path.exists(args.merges):
        raise FileNotFoundError("Vocab or Merges file not found.")

    # Input handling (Directory or Single File)
    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith('.txt') or f.endswith('.md')]
        files.sort()
    else:
        files = [args.input]

    all_ids = []

    # Process files
    special_token_bytes = args.special_token.encode("utf-8")

    print(f"Starting encoding with {args.num_threads} threads...")

    for file_path in files:
        print(f"Processing {file_path}...")
        file_size = os.path.getsize(file_path)

        # 1. Calculate Chunks (Reuse logic from BPE training for safety)
        with open(file_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, args.num_threads, special_token_bytes)

        # 2. Prepare Tasks
        tasks = []
        for i in range(len(boundaries) - 1):
            tasks.append((
                file_path,
                boundaries[i],
                boundaries[i+1],
                args.vocab,
                args.merges,
                args.special_token
            ))

        # 3. Run Parallel Tokenization
        # We assume the file fits in RAM as tokens (uint16 is small).
        # For MASSIVE datasets (TB+), you would write to disk incrementally.
        with multiprocessing.Pool(args.num_threads) as pool:
            # map maintains order of chunks
            results = pool.map(_worker_encode_chunk, tasks)

        # 4. Aggregate results for this file
        for res in results:
            all_ids.extend(res)

    print(f"Encoding complete. Total tokens: {len(all_ids)}")

    # --- Serialize to uint16 NumPy array ---
    print(f"Converting to uint16 numpy array...")
    arr = np.array(all_ids, dtype=np.uint16)

    print(f"Saving to {args.output}...")
    np.save(args.output, arr)

    print("Done.")

if __name__ == "__main__":
    main()