import argparse

# 按你的工程结构一般是这个路径（测试也这么 import）
from cs336_basics_old.tokenizer import Tokenizer


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts_vocab", required=True)
    ap.add_argument("--ts_merges", required=True)
    ap.add_argument("--owt_vocab", required=True)
    ap.add_argument("--owt_merges", required=True)
    ap.add_argument("--special", nargs="*", default=["<|endoftext|>"])
    ap.add_argument("--drop_base_bytes", action="store_true",
                    help="如果开启：从 vocab 里去掉 0..255 的单字节基础词表（可选）")
    args = ap.parse_args()

    ts_tok = Tokenizer.from_files(args.ts_vocab, args.ts_merges, special_tokens=args.special)
    owt_tok = Tokenizer.from_files(args.owt_vocab, args.owt_merges, special_tokens=args.special)

    V_ts = set(ts_tok.vocab.values())
    V_owt = set(owt_tok.vocab.values())

    if args.drop_base_bytes:
        base = {bytes([i]) for i in range(256)}
        V_ts -= base
        V_owt -= base

    M_ts = set(ts_tok.merges)
    M_owt = set(owt_tok.merges)

    print("=== Vocab Jaccard ===")
    print(" |V_ts| =", len(V_ts))
    print(" |V_owt| =", len(V_owt))
    print(" |∩|    =", len(V_ts & V_owt))
    print(" |∪|    =", len(V_ts | V_owt))
    print(" Jaccard =", jaccard(V_ts, V_owt))

    print("\n=== Merges Jaccard ===")
    print(" |M_ts| =", len(M_ts))
    print(" |M_owt| =", len(M_owt))
    print(" |∩|    =", len(M_ts & M_owt))
    print(" |∪|    =", len(M_ts | M_owt))
    print(" Jaccard =", jaccard(M_ts, M_owt))


if __name__ == "__main__":
    main()
