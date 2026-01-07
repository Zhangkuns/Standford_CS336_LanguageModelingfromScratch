import random
import gzip
from fastwarc.warc import ArchiveIterator, WarcRecordType

# Adjust imports based on your file structure

    # Try importing from the pii module if that's where you put extract_text
from cs336_data.tools import extract_text_from_html_bytes
from cs336_data.quality import gopher_quality_filter

WARC_PATH = "./data/CC-MAIN-20250417135010-20250417165010-00065.warc"
NUM_SAMPLES = 20

def main():
    print(f"Scanning {WARC_PATH}...")

    valid_texts = []

    # 1. Collect a pool of valid text documents first
    # We read enough records to get a good random sample (e.g., read first 500)
    with open(WARC_PATH, "rb") as f:
        for i, rec in enumerate(ArchiveIterator(f)):
            if len(valid_texts) >= 200: # Stop after collecting enough candidates
                break

            if rec.record_type != WarcRecordType.response:
                continue

            # Extract text
            payload = rec.reader.read()
            try:
                # Handle cases where extraction might return None
                text = extract_text_from_html_bytes(payload)
                if text and len(text.strip()) > 0:
                    valid_texts.append(text)
            except Exception as e:
                continue

    # 2. Randomly Sample 20 from the pool
    # This ensures the distribution matches the raw data
    if len(valid_texts) < NUM_SAMPLES:
        print(f"Warning: Only found {len(valid_texts)} valid documents.")
        samples = valid_texts
    else:
        samples = random.sample(valid_texts, NUM_SAMPLES)

    print(f"\nEvaluating Gopher Quality Filter on {len(samples)} random samples.\n")

    # 3. Evaluate
    for idx, text in enumerate(samples):
        # Run the filter
        passed = gopher_quality_filter(text)
        status = "KEPT" if passed else "REJECTED"

        print(f"--- SAMPLE {idx+1} [{status}] ---")

        # Print stats to help understand WHY it passed/failed
        words = text.split()
        print(f"Stats: Length={len(words)} words")

        # Print a snippet (Head + Tail to see context)
        snippet_len = 300
        if len(text) > snippet_len * 2:
            print(f"CONTENT: {text[:snippet_len]} ...\n... {text[-snippet_len:]}")
        else:
            print(f"CONTENT: {text}")

        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()