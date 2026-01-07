import regex as re
import json
import os
from typing import List, Dict, Tuple, Iterable, Iterator, Optional

# Re-using the GPT-2 Pattern from training
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        """
        Construct a tokenizer from a given vocabulary and list of merges.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []

        # 1. Build Inverse Maps for Encoding
        # map bytes -> token_id
        self.token_to_id = {v: k for k, v in self.vocab.items()}

        # 2. Build BPE Ranks
        # We need to know which merge applies "first" (based on creation order).
        # We map (byte_a, byte_b) -> Rank (index)
        # Lower rank = happened earlier = higher priority to merge
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}

        # 3. Handle Special Tokens
        # Ensure they are in the vocabulary. If not, add them (though typically training handles this).
        # We also need a fast way to recognize them in text.
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in self.token_to_id:
                # If special token missing, we assign a new ID
                new_id = len(self.vocab)
                self.vocab[new_id] = st_bytes
                self.token_to_id[st_bytes] = new_id

        # Compile Regex for efficiency
        self.regex = re.compile(GPT2_SPLIT_PATTERN)

        # Cache for BPE encoding of words to speed up processing
        self.cache = {}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None):
        """
        Construct Tokenizer from saved files.
        """
        # Load Vocab (JSON)
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            # JSON keys are strings, values are unicode strings.
            # We need int -> bytes
            raw_vocab = json.load(f)
            vocab = {}
            for k, v in raw_vocab.items():
                # We assume the vocab JSON stored values as strings (using latin-1 or utf-8 with replacement)
                # But typically BPE training outputs pure bytes.
                # If we used the `save_results` function from previous steps, v is a string.
                # We encode back to bytes using latin-1 to recover 1-to-1 byte mapping if possible,
                # or utf-8 if that's how it was saved. Standard practice for byte-level BPE json is tricky.
                # NOTE: Assuming standard UTF-8 string storage here.
                if isinstance(v, str):
                    # v_bytes = v.encode('utf-8', errors='replace')
                    v_bytes = v.encode("latin-1")
                else:
                    v_bytes = bytes(v) # Should not happen in standard JSON
                vocab[int(k)] = v_bytes

        # Load Merges (Text file)
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                # Line format: b'a' b'b'  (repr strings)
                # We need to parse this safely.
                try:
                    # Split on space, but be careful about spaces inside reprs
                    # A robust way is to eval() the parts since they are repr() outputs
                    # e.g., "b't' b'h'"
                    # Naive split might fail on spaces.
                    # Let's try splitting by " b'" which usually separates them
                    parts = line.split(" b'")
                    if len(parts) == 2:
                        p1 = parts[0] # b't'
                        p2 = "b'" + parts[1] # b'h'
                        # Eval matches repr to bytes
                        # WARNING: eval is unsafe on untrusted input, but standard for this assignment level
                        import ast
                        merges.append((ast.literal_eval(p1), ast.literal_eval(p2)))
                except Exception as e:
                    print(f"Warning: could not parse merge line: {line}")

        return cls(vocab, merges, special_tokens)

    def _bpe(self, token_bytes: bytes) -> List[int]:
        """
        Apply BPE merges to a single pre-token (word).
        Returns a list of token IDs.
        """
        if token_bytes in self.cache:
            return self.cache[token_bytes]

        # 1. Start with list of integers (one for each byte)
        # We rely on the fact that vocab[0..255] are the bytes 0..255
        word = [self.token_to_id[bytes([b])] for b in token_bytes]

        while len(word) > 1:
            # Find the "Best" pair to merge
            min_rank = float('inf')
            pair_to_merge = None
            merge_idx = -1

            # Check all adjacent pairs
            for i in range(len(word) - 1):
                # Convert IDs back to bytes to look up in merges
                # We look up what bytes these IDs correspond to
                b1 = self.vocab[word[i]]
                b2 = self.vocab[word[i+1]]
                pair = (b1, b2)

                if pair in self.bpe_ranks:
                    rank = self.bpe_ranks[pair]
                    if rank < min_rank:
                        min_rank = rank
                        pair_to_merge = pair
                        merge_idx = i

            # If no applicable merges found, stop
            if pair_to_merge is None:
                break

            # Apply the merge
            # We found the pair at `merge_idx`.
            # Note: We merge ALL occurrences of this pair?
            # Standard BPE implementation usually merges the lowest rank pair *iteratively*.
            # The Example says "identify the first applicable merge... then go back to list".
            # This implies we merge one specific pair, then re-evaluate.

            b1, b2 = pair_to_merge
            new_token_bytes = b1 + b2
            new_token_id = self.token_to_id[new_token_bytes]

            # Replace occurrences
            # We iterate and rebuild the list
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1:
                    cur_bytes = self.vocab[word[i]]
                    next_bytes = self.vocab[word[i+1]]
                    if (cur_bytes, next_bytes) == pair_to_merge:
                        new_word.append(new_token_id)
                        i += 2
                        continue
                new_word.append(word[i])
                i += 1
            word = new_word

        self.cache[token_bytes] = word
        return word

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        """
        # 1. Handle Special Tokens (Split text logic)
        # If we have special tokens, we split the text so we don't merge across them
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(t) for t in sorted_specials]
            # Use capture group () to keep the delimiter in the result list
            pattern = "(" + "|".join(escaped) + ")"
            chunks = re.split(pattern, text)
        else:
            chunks = [text]

        ids = []
        for chunk in chunks:
            if not chunk: continue

            # Check if this chunk IS a special token
            chunk_bytes = chunk.encode("utf-8")
            if chunk_bytes in self.token_to_id and chunk in self.special_tokens:
                ids.append(self.token_to_id[chunk_bytes])
            else:
                # Normal text processing
                # 2. Pre-tokenize with Regex
                # findall gives us strings
                pre_tokens = self.regex.findall(chunk)

                # 3. Apply BPE to each pre-token
                for pt in pre_tokens:
                    pt_bytes = pt.encode("utf-8")
                    ids.extend(self._bpe(pt_bytes))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Generator for encoding large streams of text.
        """
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    def decode(self, ids: List[int]) -> str:
        """
        Decode IDs to string.
        """
        # 1. Map IDs to bytes
        byte_parts = []
        for i in ids:
            if i in self.vocab:
                byte_parts.append(self.vocab[i])
            else:
                # Unknown ID (shouldn't happen if vocab is complete)
                pass

                # 2. Concatenate
        full_bytes = b"".join(byte_parts)

        # 3. Decode to string with replacement char for invalid sequences
        return full_bytes.decode("utf-8", errors="replace")