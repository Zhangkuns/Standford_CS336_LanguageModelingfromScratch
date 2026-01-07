import json
import sys

# 用法：python find_longest.py /path/to/vocab.json
vocab_path = sys.argv[1] if len(sys.argv) > 1 else "vocab.json"

with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)  # dict: id(str) -> token(str)

# 1) 最长（按字符数）
best_id_char, best_tok_char = max(vocab.items(), key=lambda kv: len(kv[1]))

# 2) 最长（按 UTF-8 字节数）
best_id_byte, best_tok_byte = max(vocab.items(), key=lambda kv: len(kv[1].encode("utf-8")))

print("=== Longest by characters ===")
print("id:", best_id_char)
print("chars:", len(best_tok_char))
print("bytes:", len(best_tok_char.encode("utf-8")))
print("repr:", repr(best_tok_char))
print("text:", best_tok_char)

print("\n=== Longest by UTF-8 bytes ===")
print("id:", best_id_byte)
print("chars:", len(best_tok_byte))
print("bytes:", len(best_tok_byte.encode("utf-8")))
print("repr:", repr(best_tok_byte))
print("text:", best_tok_byte)
