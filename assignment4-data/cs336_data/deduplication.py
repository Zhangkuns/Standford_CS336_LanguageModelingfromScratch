
import hashlib
import os
import re
import unicodedata
import random
import shutil
from collections import defaultdict
from typing import List, Set, Dict, Tuple, Union

def get_line_hash(line: str) -> str:
    """Computes a hash for a line of text."""
    # We strip the newline character to ensure consistency across files
    # preserving internal whitespace.
    content = line.rstrip('\n')
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def exact_line_deduplication(
        input_paths: List[Union[str, os.PathLike]],
        output_dir: Union[str, os.PathLike]
):
    """
    Performs exact line deduplication across a set of files.

    Args:
        input_paths: List of file paths (strings or Path objects).
        output_dir: Output directory path.
    """
    # Ensure output directory exists
    output_dir = str(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    line_counts = defaultdict(int)

    # --- PASS 1: Count Frequency of every line across ALL files ---
    # We count how many times a line appears in the *entire corpus*.
    for file_path in input_paths:
        # Convert Path objects to string to be safe
        path_str = str(file_path)

        try:
            with open(path_str, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    h = get_line_hash(line)
                    line_counts[h] += 1
        except Exception as e:
            print(f"Warning: Could not read {path_str} during counting: {e}")

    # --- PASS 2: Rewrite files, keeping only GLOBALLY UNIQUE lines ---
    for file_path in input_paths:
        path_str = str(file_path)
        filename = os.path.basename(path_str)
        output_path = os.path.join(output_dir, filename)

        try:
            with open(path_str, 'r', encoding='utf-8', errors='ignore') as fin, \
                    open(output_path, 'w', encoding='utf-8') as fout:

                for line in fin:
                    h = get_line_hash(line)

                    # Logic: "remove lines that occur more than once in the set of input files"
                    # This means we ONLY keep lines where the global count is exactly 1.
                    if line_counts[h] == 1:
                        fout.write(line)

        except Exception as e:
            print(f"Warning: Could not process {path_str} during writing: {e}")


# Use a large prime for hashing (Mersenne prime 2^61 - 1)
MERSENNE_PRIME = (1 << 61) - 1
MAX_HASH = (1 << 32) - 1

class UnionFind:
    """Helper structure to manage duplicate clusters."""
    def __init__(self):
        self.parent = {}

    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Union by ID (keep the lexicographically smaller path as root for consistency)
            if root_i < root_j:
                self.parent[root_j] = root_i
            else:
                self.parent[root_i] = root_j
            return True
        return False

def normalize_text(text: str) -> str:
    """
    Normalizes text by:
    1. NFD Unicode normalization.
    2. Lowercasing.
    3. Removing accents (Combining Diacritical Marks).
    4. Removing punctuation.
    5. Normalizing whitespace.
    """
    # 1. NFD Normalization
    text = unicodedata.normalize('NFD', text)

    # 3. Remove accents (category 'Mn')
    text = "".join([c for c in text if unicodedata.category(c) != 'Mn'])

    # 2. Lowercase
    text = text.lower()

    # 4. Remove punctuation (keep alphanumeric and whitespace)
    # Using regex [^\w\s] removes anything that isn't a word char or space
    text = re.sub(r'[^\w\s]', '', text)

    # 5. Normalize whitespace (collapse multiple spaces to one)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def get_ngrams(text: str, n: int) -> Set[str]:
    """Generates a set of word n-grams."""
    words = text.split()
    if len(words) < n:
        return set() # Or handle as single set containing the short string

    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i : i + n])
        ngrams.add(ngram)
    return ngrams

def get_minhash_signature(ngrams: Set[str], num_hashes: int, perm_a: List[int], perm_b: List[int]) -> List[int]:
    """
    Computes MinHash signature for a set of ngrams.
    Signature[i] = min( (a_i * hash(ngram) + b_i) % PRIME )
    """
    # Initialize signature with max value
    signature = [MERSENNE_PRIME] * num_hashes

    for ngram in ngrams:
        # Get a base hash of the ngram string
        # Python hash() is salted per process, which is fine for a single run script.
        # For multi-process/persistent usage, use hashlib.md5 or similar.
        h = hash(ngram) & MAX_HASH

        for i in range(num_hashes):
            # Universal Hashing: (a*x + b) % p
            ph_val = (perm_a[i] * h + perm_b[i]) % MERSENNE_PRIME
            if ph_val < signature[i]:
                signature[i] = ph_val

    return signature

def compute_jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes exact Jaccard similarity."""
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0: return 0.0
    return intersection / union

def minhash_deduplication(
        input_paths: List[Union[str, os.PathLike]],
        num_hashes: int,
        num_bands: int,
        ngram_size: int,
        output_dir: Union[str, os.PathLike],
        jaccard_threshold: float = 0.8
):
    """
    Performs fuzzy deduplication using MinHash + LSH.
    """
    output_dir = str(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Setup Hash Permutations
    # Random coefficients a and b for the linear hash functions
    random.seed(42) # For reproducibility
    perm_a = [random.randint(1, MERSENNE_PRIME - 1) for _ in range(num_hashes)]
    perm_b = [random.randint(0, MERSENNE_PRIME - 1) for _ in range(num_hashes)]

    # Data storage
    doc_ngrams = {}   # path -> set of ngrams
    signatures = {}   # path -> signature list
    input_paths = [str(p) for p in input_paths]

    print("Step 1: Computing Signatures...")
    for path in input_paths:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()

            clean_text = normalize_text(raw_text)
            ngrams = get_ngrams(clean_text, ngram_size)

            # Store ngrams for Jaccard verification step
            doc_ngrams[path] = ngrams

            if not ngrams:
                # If file is empty or too short, treat signature as empty/max or skip
                # We'll skip empty docs for dedupe logic (they are unique or empty)
                continue

            sig = get_minhash_signature(ngrams, num_hashes, perm_a, perm_b)
            signatures[path] = sig

        except Exception as e:
            print(f"Error reading {path}: {e}")

    # 2. LSH Bucketing
    print("Step 2: LSH Bucketing...")
    rows_per_band = num_hashes // num_bands

    # bucket_id -> list of file_paths
    # bucket_id is (band_idx, hash_of_band_slice)
    buckets = defaultdict(list)

    for path, sig in signatures.items():
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band

            # Create a tuple for the band segment so it's hashable
            band_tuple = tuple(sig[start:end])
            bucket_key = (band_idx, band_tuple)

            buckets[bucket_key].append(path)

    # 3. Find Candidates and Verify
    print("Step 3: Verification and Clustering...")
    uf = UnionFind()

    # Initialize UF for all files
    for path in input_paths:
        uf.find(path)

    # To avoid checking the same pair multiple times across bands
    checked_pairs = set()

    for bucket_key, file_list in buckets.items():
        if len(file_list) > 1:
            # All pairs in this bucket are candidates
            for i in range(len(file_list)):
                for j in range(i + 1, len(file_list)):
                    doc_a = file_list[i]
                    doc_b = file_list[j]

                    # Sort pair to normalize key
                    if doc_a > doc_b: doc_a, doc_b = doc_b, doc_a

                    if (doc_a, doc_b) in checked_pairs:
                        continue
                    checked_pairs.add((doc_a, doc_b))

                    # Compute Exact Jaccard
                    sim = compute_jaccard(doc_ngrams[doc_a], doc_ngrams[doc_b])

                    if sim >= jaccard_threshold:
                        uf.union(doc_a, doc_b)

    # 4. Write Output
    print("Step 4: Writing unique documents...")
    for path in input_paths:
        # Check if this document is the representative of its cluster
        # uf.find(path) returns the representative.
        # If uf.find(path) == path, we keep it.
        # If not, it's a duplicate of someone else.
        if uf.find(path) == path:
            filename = os.path.basename(path)
            out_path = os.path.join(output_dir, filename)
            shutil.copy2(path, out_path)