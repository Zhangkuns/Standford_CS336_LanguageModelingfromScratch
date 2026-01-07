from fastwarc.warc import ArchiveIterator, WarcRecordType

# 你的函数（按你之前写的导入/粘贴）
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from cs336_data.tools import extract_text_from_html_bytes

warc_gz = "./data/CC-MAIN-20250417135010-20250417165010-00065.warc"
wet_gz  = "./data/CC-MAIN-20250417135010-20250417165010-00065.warc.wet"

# 读 WARC：抽取第一个网页（response）HTML -> text
warc_url = None
with open(warc_gz, "rb") as f:
    for rec in ArchiveIterator(f):
        if rec.record_type == WarcRecordType.response:
            warc_url = rec.headers.get("WARC-Target-URI")
            payload = rec.reader.read()
            warc_text = extract_text_from_html_bytes(payload)
            break

print("WARC URL:", warc_url)

# 读 WET：拿第一个 conversion record 的正文（跳过头部行）
wet_text = ""
with open(wet_gz, "rb") as f:
    for rec in ArchiveIterator(f):
        if rec.record_type == WarcRecordType.conversion:
            if rec.headers.get("WARC-Target-URI") == warc_url:
                raw = rec.reader.read()
                s = raw.decode("utf-8", errors="replace")
                # WET record 通常也是“头 + 空行 + 正文”
                parts = s.split("\n\n", 1)
                wet_text = parts[1].strip() if len(parts) == 2 else ""
                break
print("WET TEXT (first 1500):", wet_text[:1500])

print("=== YOUR EXTRACTION (from WARC HTML) ===")
print(warc_text[:1500])
print("\n=== WET EXTRACTION (from .wet.gz) ===")
print(wet_text[:1500])
