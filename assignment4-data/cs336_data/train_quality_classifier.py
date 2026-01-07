import os
import gzip
import random
import subprocess
import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
# Import your HTML extractor from Problem 2.2
from cs336_data.tools import extract_text_from_html_bytes

# Paths based on the Together cluster configuration
WIKI_URLS_PATH = "/home/zks/Disk/2025FirstSemester/CS336LargeLanguageModel/LAB/assignment4-data/data/enwiki-20240420-extracted_urls.txt"
CC_WET_DIR = "/home/zks/Disk/2025FirstSemester/CS336LargeLanguageModel/LAB/assignment4-data/data/CC"
OUTPUT_MODEL_DIR = "./cs336_data/assets"
OUTPUT_MODEL_NAME = "quality_classifier.bin"
TEMP_DATA_FILE = "quality_classifier_train_data.txt"

NUM_SAMPLES = 2000  # Number of positive/negative samples to use

def gather_high_quality_urls():
    print("--- Step 1: Gathering High-Quality URLs ---")
    positive_urls = []
    with open(WIKI_URLS_PATH, "rt", encoding="utf-8") as f:
        # Read all URLs (or a large chunk) to sample from
        # Using a reservoir sampling or just reading first N for speed if file is huge
        all_urls = [line.strip() for line in f if line.strip()]

    positive_urls = random.sample(all_urls, NUM_SAMPLES)

    # Save URLs to file for wget
    with open("subsampled_positive_urls.txt", "w") as f:
        f.write("\n".join(positive_urls))

def scrape_high_quality_urls():
    print("--- Step 2: Scraping High-Quality Pages (WARC) ---")
    # Use the wget command suggested in the assignment
    subprocess.run([
        "wget", "--timeout=5", "--tries=1",
        "-i", "subsampled_positive_urls.txt",
        "--warc-file=./data/wiki/subsampled_positive_urls",
        "-O", "/dev/null"
    ], check=False) # check=False to continue even if some URLs fail

def prepare_training_data():
    # Find the generated WARC file (wget might compress it or not depending on version/flags)
    positive_warc = "./data/wiki/subsampled_positive_urls.warc.gz"
    if not os.path.exists(positive_warc):
        positive_warc = "./data/wiki/subsampled_positive_urls.warc"

    print("--- Step 3: Extracting Text & Formatting Data ---")

    with open(TEMP_DATA_FILE, "w", encoding="utf-8") as outfile:

        # Process Positive Examples (from the scraped WARC)
        pos_count = 0
        try:
            stream = open(positive_warc, "rb") if positive_warc.endswith(".gz") else open(positive_warc, "rb")
            for rec in ArchiveIterator(stream):
                if rec.record_type == WarcRecordType.response:
                    content = rec.reader.read()
                    text = extract_text_from_html_bytes(content)
                    if text and len(text.strip()) > 100:
                        # fastText requires one document per line. Remove newlines.
                        clean_text = text.replace("\n", " ").strip()
                        outfile.write(f"__label__high_quality {clean_text}\n")
                        pos_count += 1
            stream.close()
        except Exception as e:
            print(f"Warning reading positive WARC: {e}")

        print(f"Collected {pos_count} positive examples.")

        # Process Negative Examples (from Common Crawl WETs)
        neg_count = 0
        cc_files = [os.path.join(CC_WET_DIR, f) for f in os.listdir(CC_WET_DIR) if f.endswith('.wet')]

        # Shuffle file list to get random negative samples
        random.shuffle(cc_files)

        for wet_path in cc_files:
            if neg_count >= pos_count: # Keep classes balanced
                break

            try:
                with open(wet_path, "rt", encoding="utf-8") as f:
                    # Simple parsing of WET format: content starts after header
                    # WET records are separated by "WARC/1.0" headers.
                    content = f.read()
                    records = content.split("WARC/1.0")

                    for rec in records:
                        if neg_count >= pos_count: break

                        # Skip header block, look for content length
                        if "Content-Length:" in rec:
                            lines = rec.strip().split("\n")
                            # The body usually starts after the header lines.
                            # Heuristic: find the empty line separating header and body
                            try:
                                body_idx = lines.index("") + 1
                                text = " ".join(lines[body_idx:])
                                if len(text) > 100:
                                    clean_text = text.replace("\n", " ").strip()
                                    outfile.write(f"__label__low_quality {clean_text}\n")
                                    neg_count += 1
                            except ValueError:
                                continue
            except Exception as e:
                print(f"Error reading WET {wet_path}: {e}")

        print(f"Collected {neg_count} negative examples.")

def train_classifier():
    print("--- Step 4: Training FastText Model ---")

    # hyperparams: lr=learning rate, epoch=epochs, wordNgrams=2 (use bigrams)
    model = fasttext.train_supervised(input=TEMP_DATA_FILE, lr=0.5, epoch=25, wordNgrams=2)

    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_MODEL_DIR, OUTPUT_MODEL_NAME)
    model.save_model(save_path)

    print(f"Model saved to {save_path}")

    # Quick Test
    print("Validation (Self):")
    result = model.test(TEMP_DATA_FILE)
    print(f"Samples: {result[0]}, Precision: {result[1]:.4f}, Recall: {result[2]:.4f}")

if __name__ == "__main__":
    gather_high_quality_urls()
    scrape_high_quality_urls()
    prepare_training_data()
    train_classifier()