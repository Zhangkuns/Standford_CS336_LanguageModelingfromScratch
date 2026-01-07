import random
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.tools import extract_text_from_html_bytes, identify_language

# 你解压后的 WARC（注意是 .warc，不是 .warc.gz）
warc_path = "./data/CC-MAIN-20250417135010-20250417165010-00065.warc"

# 如果 lid.176.bin 就放在 data/ 目录下，写这个：
lid_path = "./data/classifiers/lid.176.bin"

docs = []
with open(warc_path, "rb") as f:
    for rec in ArchiveIterator(f):
        if rec.record_type == WarcRecordType.response:
            payload = rec.reader.read()
            text = extract_text_from_html_bytes(payload)
            text = (text or "").strip()
            if len(text) >= 200:  # 太短的跳过
                docs.append(text)

random.seed(0)
sample_docs = random.sample(docs, k=min(20, len(docs)))

en_count = 0
for i, t in enumerate(sample_docs):
    # 显式指定模型路径（避免默认路径找不到）
    pred_lang, pred_conf = identify_language(t, model_path=lid_path)

    if pred_lang == "en":
        en_count += 1

    print(f"\n--- DOC {i} ---")
    print("pred:", pred_lang, pred_conf)
    print(t[:400])

print("\nTotal sampled:", len(sample_docs))
print("English fraction:", en_count / max(1, len(sample_docs)))

