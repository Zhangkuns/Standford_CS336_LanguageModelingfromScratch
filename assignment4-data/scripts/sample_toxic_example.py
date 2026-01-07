import random
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.tools import (
    extract_text_from_html_bytes,
    classify_nsfw,
    classify_toxic_speech,
)

warc_path = "./data/CC-MAIN-20250417135010-20250417165010-00065.warc"  # 你已解压后的路径

random.seed(0)

hits = []  # 存 (url, nsfw_count, toxic_count, before_snip, after_snip)

def snippet_around_change(before: str, after: str, window: int = 120):
    """
    找到第一次变化的位置，截取前后 window 字符，便于人工判断 FP/FN
    """
    # 找第一个不同的位置
    i = 0
    L = min(len(before), len(after))
    while i < L and before[i] == after[i]:
        i += 1
    start = max(0, i - window)
    end = min(len(after), i + window)
    return before[start:end], after[start:end]

with open(warc_path, "rb") as f:
    for rec in ArchiveIterator(f):
        if rec.record_type != WarcRecordType.response:
            continue

        url = rec.headers.get("WARC-Target-URI", "")
        payload = rec.reader.read()
        text = (extract_text_from_html_bytes(payload) or "").strip()
        if len(text) < 200:
            continue

        # 分类器预测结果
        nsfw_label, nsfw_score = classify_nsfw(text)
        toxic_label, toxic_score = classify_toxic_speech(text)

        # 判断是否有害内容
        nsfw_count = 1 if nsfw_label == "NSFW" and nsfw_score >= 0.8 else 0
        toxic_count = 1 if toxic_label == "toxic" and toxic_score >= 0.8 else 0

        total = nsfw_count + toxic_count
        if total == 0:
            continue

        # 截取替换前后的片段
        before_snip, after_snip = snippet_around_change(text, text)  # 这里可以扩展为替换后内容

        hits.append((
            url,
            {"nsfw": nsfw_count, "toxic": toxic_count, "total": total},
            before_snip,
            after_snip
        ))

# 随机抽 20 个发生替换的例子（如果不足 20，就全打印）
sample = random.sample(hits, k=min(20, len(hits)))

print(f"Found {len(hits)} docs with at least one harmful label. Showing {len(sample)} samples.\n")

for i, (url, counts, before_snip, after_snip) in enumerate(sample):
    print(f"--- SAMPLE {i} ---")
    print("URL:", url)
    print("counts:", counts)
    print("\n[BEFORE]\n", before_snip)
    print("\n[AFTER]\n", after_snip)
    print("\n" + "-" * 60 + "\n")

# 统计有害内容的比例
harmful_count = sum([1 for _, counts, _, _ in hits if counts["total"] > 0])
harmful_fraction = harmful_count / len(hits) if len(hits) > 0 else 0

print(f"Total samples with harmful content: {harmful_count}/{len(hits)}")
print(f"Harmful content fraction: {harmful_fraction:.2f}")
