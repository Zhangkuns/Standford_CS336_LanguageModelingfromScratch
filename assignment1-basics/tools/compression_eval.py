import argparse
import time
import random
from typing import Iterator, List

from cs336_basics_old.tokenizer import Tokenizer


def iter_docs_by_delim(path: str, delim: bytes, chunk_size: int = 4 * 1024 * 1024) -> Iterator[bytes]:
    """
    流式读取大文件，并用 delim (bytes) 作为文档分隔符产出每篇文档的原始 bytes。
    不把整个文件读入内存。
    """
    buf = b""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buf += chunk
            while True:
                idx = buf.find(delim)
                if idx == -1:
                    break
                doc = buf[:idx]
                yield doc
                buf = buf[idx + len(delim):]
        # 文件末尾剩余
        if buf:
            yield buf

def take_first_n_docs(path: str, delim: bytes, n: int = 10, chunk_size: int = 4 * 1024 * 1024) -> List[bytes]:
    """按 delim 分隔，返回文件里前 n 篇文档（bytes）。"""
    docs = []
    for doc in iter_docs_by_delim(path, delim, chunk_size=chunk_size):
        if doc:  # 可选：跳过空文档
            docs.append(doc)
        if len(docs) >= n:
            break
    return docs

def reservoir_sample_docs(doc_iter: Iterator[bytes], k: int, seed: int = 0) -> List[bytes]:
    """
    Reservoir sampling：从未知总数的文档流里均匀随机抽 k 篇。
    """
    rng = random.Random(seed)
    sample: List[bytes] = []
    n_seen = 0
    for doc in doc_iter:
        n_seen += 1
        if len(sample) < k:
            sample.append(doc)
        else:
            j = rng.randrange(n_seen)
            if j < k:
                sample[j] = doc
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True, help="path to vocab.json")
    ap.add_argument("--merges", required=True, help="path to merges.txt")
    ap.add_argument("--text", required=True, help="path to dataset .txt (TinyStories/OWT)")
    ap.add_argument("--num_docs", type=int, default=10, help="number of documents to sample")
    ap.add_argument("--special", nargs="*", default=["<|endoftext|>"], help="special tokens list")
    ap.add_argument("--delim", default="<|endoftext|>", help="document delimiter token string")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decode_errors", default="ignore", choices=["ignore", "replace", "strict"])
    ap.add_argument("--per_doc", action="store_true", help="print per-document stats")
    args = ap.parse_args()

    tok = Tokenizer.from_files(args.vocab, args.merges, special_tokens=args.special)

    delim_bytes = args.delim.encode("utf-8")
    # docs = reservoir_sample_docs(iter_docs_by_delim(args.text, delim_bytes), args.num_docs, seed=args.seed)
    docs = take_first_n_docs(args.text, delim_bytes, n=args.num_docs)


    if len(docs) == 0:
        raise RuntimeError("No documents found. Check --delim matches your dataset separator.")

    total_bytes = 0
    total_tokens = 0

    t0 = time.perf_counter()
    for i, doc_bytes in enumerate(docs):
        # 压缩率建议按“原始字节数 / token数”
        b = len(doc_bytes)
        text = doc_bytes.decode("utf-8", errors=args.decode_errors)
        ids = tok.encode(text)
        ntok = len(ids)

        total_bytes += b
        total_tokens += ntok

        if args.per_doc:
            bpt = (b / ntok) if ntok else 0.0
            print(f"[doc {i:02d}] bytes={b} tokens={ntok} bytes/token={bpt:.4f}")

    t1 = time.perf_counter()
    sec = t1 - t0
    bytes_per_token = (total_bytes / total_tokens) if total_tokens else 0.0
    throughput = (total_bytes / sec) if sec > 0 else 0.0

    print("\n=== Summary ===")
    print(f"sampled_docs      : {len(docs)}")
    print(f"total_bytes       : {total_bytes}")
    print(f"total_tokens      : {total_tokens}")
    print(f"bytes/token       : {bytes_per_token:.6f}")
    print(f"throughput bytes/s: {throughput:.2f}")
    print(f"elapsed seconds   : {sec:.4f}")


if __name__ == "__main__":
    main()
