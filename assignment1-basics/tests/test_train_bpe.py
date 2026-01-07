import json
import time

from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    print(f"BPE training took {end_time - start_time:.2f} seconds.")
    assert end_time - start_time < 1.5


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
     # 🔍 手动检查 merges 是否一致，并打印第一个差异
    if merges != reference_merges:
        min_len = min(len(merges), len(reference_merges))
        for i in range(min_len):
            if merges[i] != reference_merges[i]:
                print(f"\n❌ First mismatch at merge index {i}:")
                print(f"  Your merge:     {merges[i]}")
                print(f"  Expected merge: {reference_merges[i]}")
                break
        else:
            if len(merges) != len(reference_merges):
                print(f"\n❌ Length mismatch!")
                print(f"  Your merges length:     {len(merges)}")
                print(f"  Expected merges length: {len(reference_merges)}")
        # 现在再触发 assert，让测试框架记录失败
    
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently, we'll make sure that the vocab keys and values match)
    if set(vocab.keys()) != set(reference_vocab.keys()):
        missing_keys = set(reference_vocab.keys()) - set(vocab.keys())
        extra_keys = set(vocab.keys()) - set(reference_vocab.keys())
        print(f"\n❌ Vocab key mismatch!")
        if missing_keys:
            print(f"  Missing keys: {sorted(missing_keys)[:5]}...")
        if extra_keys:
            print(f"  Extra keys: {sorted(extra_keys)[:5]}...")
    assert set(vocab.keys()) == set(reference_vocab.keys())
    
    if set(vocab.values()) != set(reference_vocab.values()):
        missing_vals = set(reference_vocab.values()) - set(vocab.values())
        extra_vals = set(vocab.values()) - set(reference_vocab.values())
        print(f"\n❌ Vocab value mismatch!")
        if missing_vals:
            print(f"  Missing values (first 3): {list(missing_vals)[:3]}")
        if extra_vals:
            print(f"  Extra values (first 3): {list(extra_vals)[:3]}")
    assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )
