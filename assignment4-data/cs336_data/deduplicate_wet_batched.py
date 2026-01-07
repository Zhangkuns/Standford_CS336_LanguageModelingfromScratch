import os
import json
import argparse
import time
import shutil
import random
import linecache
import itertools
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from cs336_data.deduplication import (
    normalize_text,
    get_ngrams,
    get_minhash_signature,
    compute_jaccard,
    UnionFind,
    MERSENNE_PRIME
)

# --- WORKER: STEP 1 (Compute Signatures) ---
def compute_file_signatures(args):
    file_path, num_hashes, ngram_size, perm_a, perm_b = args
    results = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    doc_id = f"{file_path}|{line_idx}"
                    clean_text = normalize_text(text)
                    ngrams = get_ngrams(clean_text, ngram_size)
                    if not ngrams: continue
                    sig = get_minhash_signature(ngrams, num_hashes, perm_a, perm_b)
                    results[doc_id] = sig
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return results

# --- WORKER: STEP 2 (Verify Candidates) ---
def load_doc_ngrams_worker(doc_id, ngram_size):
    """Read specific line from disk without loading whole file."""
    try:
        filepath, line_idx_str = doc_id.split("|")
        line_idx = int(line_idx_str)
        # Using manual open/seek is more robust than linecache for massive random access across many files
        # but linecache is easier. Let's stick to manual for memory safety.
        with open(filepath, 'r', encoding='utf-8') as f:
            # We can't easily seek to a line number without an index.
            # But iterating is slow.
            # OPTIMIZATION: Use linecache here, it handles caching per process.
            line = linecache.getline(filepath, line_idx + 1)

        if not line: return set()
        data = json.loads(line)
        clean_text = normalize_text(data["text"])
        return get_ngrams(clean_text, ngram_size)
    except Exception:
        return set()

def verify_bucket_chunk(args):
    """
    Worker function to process a list of buckets.
    Loads text from disk, checks Jaccard, returns pairs of confirmed duplicates.
    """
    chunk_of_buckets, ngram_size, threshold = args
    confirmed_pairs = []

    # Check already seen pairs within this chunk to avoid redundant I/O
    local_checked = set()

    for doc_list in chunk_of_buckets:
        if len(doc_list) < 2: continue

        # Sort to ensure stable pairing order
        doc_list.sort()

        # For every pair in this bucket
        for i in range(len(doc_list)):
            for j in range(i + 1, len(doc_list)):
                doc_a = doc_list[i]
                doc_b = doc_list[j]

                if (doc_a, doc_b) in local_checked:
                    continue
                local_checked.add((doc_a, doc_b))

                # Load Text from Disk (IO Bound)
                ngrams_a = load_doc_ngrams_worker(doc_a, ngram_size)
                ngrams_b = load_doc_ngrams_worker(doc_b, ngram_size)

                if not ngrams_a or not ngrams_b:
                    continue

                # Check exact Jaccard
                sim = compute_jaccard(ngrams_a, ngrams_b)
                if sim >= threshold:
                    confirmed_pairs.append((doc_a, doc_b))

    # Clear cache to free memory
    linecache.clearcache()
    return confirmed_pairs

# --- WORKER: STEP 3 (Rewrite Files) ---
def rewrite_file(args):
    input_path, output_dir, duplicate_doc_ids = args
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)
    kept_count = 0
    removed_count = 0
    try:
        with open(input_path, 'r', encoding='utf-8') as fin, \
                open(output_path, 'w', encoding='utf-8') as fout:
            for line_idx, line in enumerate(fin):
                doc_id = f"{input_path}|{line_idx}"
                if doc_id in duplicate_doc_ids:
                    removed_count += 1
                    continue
                fout.write(line)
                kept_count += 1
    except Exception as e:
        print(f"Error rewriting {input_path}: {e}")
    return kept_count, removed_count

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_hashes", type=int, default=100)
    parser.add_argument("--num_bands", type=int, default=20)
    parser.add_argument("--ngram_size", type=int, default=5)
    parser.add_argument("--jaccard_threshold", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    input_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".jsonl"):
                input_files.append(os.path.join(root, file))
    input_files.sort()

    print(f"Found {len(input_files)} files. Running PARALLEL Verification mode.")

    random.seed(42)
    perm_a = [random.randint(1, MERSENNE_PRIME - 1) for _ in range(args.num_hashes)]
    perm_b = [random.randint(0, MERSENNE_PRIME - 1) for _ in range(args.num_hashes)]

    # --- STEP 1: COMPUTE SIGNATURES (Parallel) ---
    print("\n[Step 1/3] Computing Signatures...")
    t0 = time.time()
    global_sigs = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(compute_file_signatures, (f, args.num_hashes, args.ngram_size, perm_a, perm_b)) for f in input_files]
        for future in tqdm(as_completed(futures), total=len(futures)):
            global_sigs.update(future.result())

    print(f"Loaded {len(global_sigs)} signatures in {time.time()-t0:.2f}s")

    # --- STEP 2: LSH BUCKETING ---
    print("\n[Step 2/3 Part A] LSH Bucketing...")
    rows_per_band = args.num_hashes // args.num_bands
    buckets = defaultdict(list)

    for doc_id, sig in tqdm(global_sigs.items(), desc="Bucketing"):
        for band_idx in range(args.num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band_tuple = tuple(sig[start:end])
            buckets[(band_idx, band_tuple)].append(doc_id)

    # Clean up signatures to free RAM for verification
    del global_sigs

    # Filter out buckets with only 1 item (no duplicates possible)
    # This reduces the list from 14M to something smaller
    active_buckets = [docs for docs in buckets.values() if len(docs) > 1]
    del buckets

    print(f"Found {len(active_buckets)} buckets with potential duplicates.")

    # --- STEP 2 Part B: PARALLEL VERIFICATION ---
    print(f"\n[Step 2/3 Part B] Verifying Candidates in Parallel ({args.workers} workers)...")
    t1 = time.time()

    uf = UnionFind()
    # Note: We can't pre-fill UF with all docs since we deleted global_sigs.
    # We will just add confirmed duplicates to UF.

    # Chunk the buckets for workers
    CHUNK_SIZE = 1000
    bucket_chunks = [active_buckets[i:i + CHUNK_SIZE] for i in range(0, len(active_buckets), CHUNK_SIZE)]

    confirmed_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit tasks
        futures = [
            executor.submit(verify_bucket_chunk, (chunk, args.ngram_size, args.jaccard_threshold))
            for chunk in bucket_chunks
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Verifying"):
            pairs = future.result()
            for doc_a, doc_b in pairs:
                uf.union(doc_a, doc_b)
                confirmed_count += 1

    print(f"Verification done in {time.time()-t1:.2f}s. Confirmed duplicates: {confirmed_count}")

    # --- STEP 3: IDENTIFY & REWRITE ---
    print("\n[Step 3/3] Rewriting Files...")
    t2 = time.time()

    # Find all docs that are NOT the root of their cluster
    # We need to scan all input files again to know "all docs",
    # OR we assume if it's in UF and not root, drop it.
    # NOTE: UF only contains docs that were part of a confirmed duplicate pair.
    # Docs that were unique never entered UF. They are implicitly kept.

    duplicate_doc_ids = set()
    for doc in uf.parent: # Iterate all docs tracked in UF
        if uf.find(doc) != doc:
            duplicate_doc_ids.add(doc)

    total_kept = 0
    total_removed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(rewrite_file, (f, args.output_dir, duplicate_doc_ids)) for f in input_files]
        for future in tqdm(as_completed(futures), total=len(futures)):
            k, r = future.result()
            total_kept += k
            total_removed += r

    print(f"\nDone! Kept: {total_kept}, Removed: {total_removed}")

if __name__ == "__main__":
    main()