import os
import regex as re
import multiprocessing
from collections import Counter
from typing import List, Dict, Tuple, BinaryIO

# The GPT-2 Regex Pattern
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> List[int]:
    """
    (The helper function provided by the assignment hint)
    Chunk the file into parts that can be counted independently.
    """
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

    return sorted(list(set(chunk_boundaries)))

def _worker_pretokenize(args):
    """
    Worker function running in a separate process.
    Reads a chunk of the file, splits by special tokens, runs regex, returns counts.
    """
    path, start, end, special_tokens = args
    
    # Compile regexes locally in the worker
    regex_compiler = re.compile(GPT2_SPLIT_PATTERN)
    
    # Read the specific chunk
    with open(path, 'rb') as f:
        f.seek(start)
        # Decode errors='ignore' handles edge cases where a byte split isn't perfect UTF-8,
        # though find_chunk_boundaries tries to prevent this.
        text = f.read(end - start).decode("utf-8", errors="ignore")

    # If we have special tokens, we must NOT merge across them.
    # We split the text by special tokens (treating them as delimiters).
    if special_tokens:
        escaped = [re.escape(t) for t in special_tokens]
        # Create a pattern that splits on ANY special token
        split_pat = "|".join(escaped)
        chunks = re.split(split_pat, text)
    else:
        chunks = [text]

    # Count words in all sub-chunks
    local_counts = Counter()
    for chunk in chunks:
        if chunk:
            local_counts.update(regex_compiler.findall(chunk))
            
    return local_counts

def train_bpe_old(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    The main driver function for BPE training.
    """
    # 1. Setup
    # TinyStories is small, but OpenWebText is big, so we default to multiple cores.
    num_processes = min(multiprocessing.cpu_count(), 8)
    
    # Identify the main delimiter for chunking (usually <|endoftext|>)
    # If no special tokens provided, we can't safely align boundaries, so we rely on simple splitting.
    main_delimiter = special_tokens[0].encode('utf-8') if special_tokens else b""
    
    print()
    print(f"Calculating chunk boundaries for {input_path}...")
    with open(input_path, 'rb') as f:
        if main_delimiter:
            boundaries = find_chunk_boundaries(f, num_processes, main_delimiter)
        else:
            # Fallback: strict byte splitting (less safe for utf-8, but works if no special tokens exist)
            f.seek(0, 2)
            size = f.tell()
            boundaries = sorted(list(set([i * (size // num_processes) for i in range(num_processes + 1)])))
            boundaries[-1] = size

    # 2. Parallel Pre-tokenization
    tasks = []
    for i in range(len(boundaries) - 1):
        tasks.append((input_path, boundaries[i], boundaries[i+1], special_tokens))

    print(f"Pre-tokenizing with {len(tasks)} tasks...")
    word_counts = Counter()
    
    with multiprocessing.Pool(num_processes) as pool:
        for result in pool.imap_unordered(_worker_pretokenize, tasks):
            word_counts.update(result)
            
    # 3. Prepare for Merge Loop
    vocab = {i: bytes([i]) for i in range(256)}
    # Convert words to tuples of bytes for hashing
    # "hello" -> (104, 101, 108, 108, 111)
    train_data = {tuple(k.encode('utf-8')): v for k, v in word_counts.items()}
    
    num_merges = vocab_size - 256 - len(special_tokens)
    merges = []
    
    print(f"Starting merge loop for {num_merges} merges...")
    
    # --- STEP 3: PREPARE OPTIMIZED STRUCTURES ---
    print("Building initial index...")
    
    # We need mutable lists, so we convert the dict to a list of objects
    # indexed_words = [ {'ids': [id, id...], 'count': 50}, ... ]
    indexed_words = []
       
    for word_str, count in word_counts.items():
        # "hello" -> b"hello" -> [104, 101, 108, 108, 111]
        ids = list(word_str.encode("utf-8")) 
        indexed_words.append({'ids': ids, 'count': count})

    # Global Pair Stats (The Counter)
    stats = Counter()
    
    # Inverted Index: Map pair -> Set of word indices
    # This lets us find WHICH words to update in O(1)
    pair_to_word_indices = {} 

    # Initial Pass: Populate stats and index
    for word_idx, item in enumerate(indexed_words):
        ids = item['ids']
        count = item['count']
        for j in range(len(ids) - 1):
            pair = (ids[j], ids[j+1])
            stats[pair] += count
            
            if pair not in pair_to_word_indices:
                pair_to_word_indices[pair] = set()
            pair_to_word_indices[pair].add(word_idx)

    # --- STEP 4: OPTIMIZED MERGE LOOP ---
    print(f"Starting optimized training for {num_merges} merges...")
    
    merges = []
    
    for i in range(num_merges):
        if not stats:
            break

        # 1. Find best pair (Frequency desc, then Lexicographical desc)
        # Note: In a real "production" system you'd use a Heap, but max() is fast enough for Python
        # best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]
        best_pair = max(stats, key=lambda p: (stats[p], vocab[p[0]], vocab[p[1]]))

        # 2. Record the merge
        new_id = 256 + i
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        
        # 3. Only update the words that contain this pair
        indices_to_update = pair_to_word_indices.get(best_pair, set()).copy()
        
        # We perform the merge on these specific words
        # and carefully update the global stats/index for the NEIGHBORS only.
        for word_idx in indices_to_update:
            item = indexed_words[word_idx]
            ids = item['ids']
            count = item['count']
            
            # We reconstruct the word with the merge applied
            new_ids = []
            j = 0
            while j < len(ids):
                # Found the pair to merge
                if j < len(ids) - 1 and ids[j] == best_pair[0] and ids[j+1] == best_pair[1]:
                    new_ids.append(new_id)
                    j += 2 # Skip A and B
                    
                else:
                    new_ids.append(ids[j])
                    j += 1
                    
            # --- DIFF OLD vs NEW PAIRS ---
            old_pairs = [(ids[i], ids[i+1]) for i in range(len(ids)-1)]
            new_pairs = [(new_ids[i], new_ids[i+1]) for i in range(len(new_ids)-1)]

            # Remove old
            for p in old_pairs:
                stats[p] -= count
                if stats[p] <= 0:
                    stats.pop(p, None)
                if p in pair_to_word_indices:
                    pair_to_word_indices[p].discard(word_idx)
                    if len(pair_to_word_indices[p]) == 0:
                        pair_to_word_indices.pop(p, None)
                        
            # Add new
            for p in new_pairs:
                stats[p] += count
                if p not in pair_to_word_indices:
                    pair_to_word_indices[p] = set()
                pair_to_word_indices[p].add(word_idx)
            
            # Update the word in our list
            item['ids'] = new_ids

        if (i + 1) % 100 == 0:
            print(f"Merge {i + 1}/{num_merges} complete.")

    # 5. Add Special Tokens
    # Usually appended to the end of the vocab
    next_id = 256 + len(merges)
    for st in special_tokens:
        vocab[next_id] = st.encode('utf-8')
        next_id += 1
        
    return vocab, merges