import random
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.tools import (
    extract_text_from_html_bytes,
    mask_emails,
    mask_phone_numbers,
    mask_ips,
)

warc_path = "./data/CC-MAIN-20250417135010-20250417165010-00065.warc"  # 你已解压后的路径

random.seed(0)

hits = []  # 存 (url, counts, before_snip, after_snip)

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

        masked = text
        total = 0

        masked, n_email = mask_emails(masked)
        masked, n_phone = mask_phone_numbers(masked)
        masked, n_ip = mask_ips(masked)

        total = n_email + n_phone + n_ip
        if total == 0:
            continue

        before_snip, after_snip = snippet_around_change(text, masked)

        hits.append((
            url,
            {"email": n_email, "phone": n_phone, "ip": n_ip, "total": total},
            before_snip,
            after_snip
        ))

# 随机抽 20 个发生替换的例子（如果不足 20，就全打印）
sample = random.sample(hits, k=min(20, len(hits)))

print(f"Found {len(hits)} docs with at least one replacement. Showing {len(sample)} samples.\n")

for i, (url, counts, before_snip, after_snip) in enumerate(sample):
    print(f"--- SAMPLE {i} ---")
    print("URL:", url)
    print("counts:", counts)
    print("\n[BEFORE]\n", before_snip)
    print("\n[AFTER]\n", after_snip)
    print("\n" + "-" * 60 + "\n")
