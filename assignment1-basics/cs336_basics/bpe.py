import os
import regex as re
import multiprocessing
from collections import Counter
from typing import List, Dict, Tuple, BinaryIO, Set
import base64, json

# The GPT-2 Regex Pattern
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# ======================
# 1. Chunking Utilities
# ======================

def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> List[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1:
        return [0, file_size]

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


# ======================
# 2. Pre-tokenization Worker
# ======================

def _worker_pretokenize(args):
    path, start, end, special_tokens = args
    regex_compiler = re.compile(GPT2_SPLIT_PATTERN)

    with open(path, 'rb') as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    if special_tokens:
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        # escaped = [re.escape(t) for t in sorted_specials]
        # # Use capture group () to keep the delimiter in the result list
        # pattern = "(" + "|".join(escaped) + ")"
        # chunks = re.split(pattern, text)
        escaped = [re.escape(t) for t in special_tokens]
        split_pat = "|".join(escaped)
        chunks = re.split(split_pat, text)
    else:
        chunks = [text]

    local_counts = Counter()
    for chunk in chunks:
        if special_tokens and chunk in special_tokens:
            continue
        if chunk:
            local_counts.update(regex_compiler.findall(chunk))

    return local_counts


# ======================
# 3. Data Structure Preparation
# ======================

def _build_initial_index(word_counts: Counter) -> Tuple[List[Dict], Counter, Dict[Tuple[int, int], Set[int]]]:
    """
    Convert word counts into mutable indexed format and build pair statistics.
    Returns:
        indexed_words: list of {'ids': List[int], 'count': int}
        stats: Counter of (byte1, byte2) -> total count
        pair_to_word_indices: mapping from pair to set of word indices
    """
    indexed_words = []
    for word_str, count in word_counts.items():
        ids = list(word_str.encode("utf-8"))
        indexed_words.append({'ids': ids, 'count': count})

    stats = Counter()
    pair_to_word_indices = {}

    for word_idx, item in enumerate(indexed_words):
        ids = item['ids']
        count = item['count']
        for j in range(len(ids) - 1):
            pair = (ids[j], ids[j + 1])
            stats[pair] += count
            if pair not in pair_to_word_indices:
                pair_to_word_indices[pair] = set()
            pair_to_word_indices[pair].add(word_idx)

    return indexed_words, stats, pair_to_word_indices


# ======================
# 4. Merge Loop (核心性能分析目标!)
# ======================

def _run_merge_loop(
        indexed_words: List[Dict],
        stats: Counter,
        pair_to_word_indices: Dict[Tuple[int, int], Set[int]],
        vocab: Dict[int, bytes],
        num_merges: int,
        special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Perform BPE merge operations efficiently using inverted index.
    """
    merges = []
    print(f"Starting optimized training for {num_merges} merges...")

    for i in range(num_merges):
        if not stats:
            break

        # Find best pair: by frequency, then lexicographical order of byte values
        best_pair = max(stats, key=lambda p: (stats[p], vocab[p[0]], vocab[p[1]]))

        # Record merge
        new_id = 256 + i
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # Update only words that contain this pair
        indices_to_update = pair_to_word_indices.get(best_pair, set()).copy()

        for word_idx in indices_to_update:
            item = indexed_words[word_idx]
            ids = item['ids']
            count = item['count']

            # Apply merge
            new_ids = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and ids[j] == best_pair[0] and ids[j + 1] == best_pair[1]:
                    new_ids.append(new_id)
                    j += 2
                else:
                    new_ids.append(ids[j])
                    j += 1

            # Compute old and new adjacent pairs
            old_pairs = [(ids[k], ids[k + 1]) for k in range(len(ids) - 1)]
            new_pairs = [(new_ids[k], new_ids[k + 1]) for k in range(len(new_ids) - 1)]

            # Remove old pairs from global structures
            for p in old_pairs:
                stats[p] -= count
                if stats[p] <= 0:
                    stats.pop(p, None)
                if p in pair_to_word_indices:
                    pair_to_word_indices[p].discard(word_idx)
                    if not pair_to_word_indices[p]:
                        pair_to_word_indices.pop(p, None)

            # Add new pairs
            for p in new_pairs:
                stats[p] += count
                if p not in pair_to_word_indices:
                    pair_to_word_indices[p] = set()
                pair_to_word_indices[p].add(word_idx)

            # Commit the update
            item['ids'] = new_ids

        if (i + 1) % 100 == 0:
            print(f"Merge {i + 1}/{num_merges} complete.")

    # Append special tokens at the end
    next_id = 256 + len(merges)
    for st in special_tokens:
        vocab[next_id] = st.encode('utf-8')
        next_id += 1

    return vocab, merges


# ======================
# 5. Main Entry Point
# ======================

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str], num_multiple: int) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Main BPE training function.
    """
    num_processes = max(num_multiple, 8)
    main_delimiter = special_tokens[0].encode('utf-8') if special_tokens else b""

    print(f"\nCalculating chunk boundaries for {input_path}...")
    with open(input_path, 'rb') as f:
        if main_delimiter:
            boundaries = find_chunk_boundaries(f, num_processes, main_delimiter)
        else:
            f.seek(0, 2)
            size = f.tell()
            step = size // num_processes
            boundaries = [i * step for i in range(num_processes + 1)]
            boundaries[-1] = size
            boundaries = sorted(set(boundaries))

    tasks = [(input_path, boundaries[i], boundaries[i + 1], special_tokens)
             for i in range(len(boundaries) - 1)]

    print(f"Pre-tokenizing with {len(tasks)} tasks...")
    word_counts = Counter()
    with multiprocessing.Pool(num_processes) as pool:
        for result in pool.imap_unordered(_worker_pretokenize, tasks):
            word_counts.update(result)

    # Initialize base vocab (0-255)
    vocab = {i: bytes([i]) for i in range(256)}
    num_merges = vocab_size - 256 - len(special_tokens)
    print(f"Starting merge loop for {num_merges} merges...")

    # Build data structures for efficient merging
    print("Building initial index...")
    indexed_words, stats, pair_to_word_indices = _build_initial_index(word_counts)

    # Run the actual merge loop — now a separate function!
    vocab, merges = _run_merge_loop(
        indexed_words=indexed_words,
        stats=stats,
        pair_to_word_indices=pair_to_word_indices,
        vocab=vocab,
        num_merges=num_merges,
        special_tokens=special_tokens
    )

    return vocab, merges