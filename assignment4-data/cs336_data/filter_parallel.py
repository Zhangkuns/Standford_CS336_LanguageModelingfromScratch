import os
import gzip
import json
import argparse
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm

# --- Import your primitives ---
# Adjust imports to match your project structure
from cs336_data.tools import identify_language, mask_emails, mask_phone_numbers, mask_ips, classify_nsfw, classify_toxic_speech
from cs336_data.quality import gopher_quality_filter
from cs336_data.inference_quality_classifer import classify_quality

# Global variables for worker processes
# We use globals so we don't pickle huge models across processes
_models_loaded = False

def init_worker():
    """
    Called once per worker process to load models (FastText) into memory.
    This prevents reloading models for every file.
    """
    global _models_loaded
    # Trigger lazy loading of singletons in your modules
    # You might need to call a dummy prediction to force load
    try:
        identify_language("dummy")
        classify_nsfw("dummy")
        classify_toxic_speech("dummy")
        classify_quality("dummy")
        _models_loaded = True
    except Exception as e:
        print(f"Worker init failed: {e}")

def process_single_wet(input_path: str, output_dir: str) -> Counter:
    """
    Process a single WET file: read, filter, mask, write.
    Returns a Counter of statistics.
    """
    stats = Counter()
    filename = os.path.basename(input_path).replace(".warc.wet.gz", ".jsonl")
    output_path = os.path.join(output_dir, filename)

    valid_docs = []

    try:
        with gzip.open(input_path, 'rb') as f:
            for rec in ArchiveIterator(f):
                if rec.record_type != WarcRecordType.conversion:
                    continue

                stats['total_documents'] += 1

                # 1. Extract Text
                # FastWARC reader.read() returns bytes, decode to string
                try:
                    text_bytes = rec.reader.read()
                    text = text_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    stats['discarded_encoding'] += 1
                    continue

                if not text or len(text.strip()) < 50:
                    stats['discarded_too_short'] += 1
                    continue

                # 2. Language ID (Fastest Check)
                # Keep English with reasonable confidence
                lang, score = identify_language(text)
                # Note: fasttext returns '__label__en', your func might strip it.
                # Adjust 'en' check based on your specific return format.
                if 'en' not in lang or score < 0.65:
                    stats['discarded_language'] += 1
                    continue

                # 3. Gopher Quality Rules (Syntactic)
                if not gopher_quality_filter(text):
                    stats['discarded_gopher'] += 1
                    continue

                # 4. Harmful Content
                # Reject if NSFW or Toxic confidence is high
                nsfw_label, nsfw_score = classify_nsfw(text)
                if 'nsfw' in nsfw_label.lower() and nsfw_score > 0.8:
                    stats['discarded_harmful_nsfw'] += 1
                    continue

                toxic_label, toxic_score = classify_toxic_speech(text)
                if 'hate' in toxic_label.lower() or 'toxic' in toxic_label.lower():
                    if toxic_score > 0.8:
                        stats['discarded_harmful_toxic'] += 1
                        continue

                # 5. Semantic Quality Classifier (Your Wiki vs CC model)
                # Keep if it looks like Wikipedia (high quality)
                # Threshold 0.5 is default, tune based on your validation needs
                q_label, q_score = classify_quality(text)
                # Assuming your classifier returns 'wiki' or 'high_quality' for good stuff
                is_high_quality = ('wiki' in q_label or 'high' in q_label)

                if not is_high_quality:
                    stats['discarded_quality_model'] += 1
                    continue

                # --- DOCUMENT KEPT! NOW CLEAN IT ---

                # 6. PII Masking
                # Apply in sequence
                text, _ = mask_emails(text)
                text, _ = mask_phone_numbers(text)
                text, _ = mask_ips(text)

                stats['kept_documents'] += 1
                valid_docs.append(json.dumps({"text": text}))

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        stats['errors'] += 1

    # Write output
    if valid_docs:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(valid_docs))

    return stats

def main():
    parser = argparse.ArgumentParser(description="Filter Common Crawl WET files")
    parser.add_argument("--input_dir", type=str, default="./data/wet", help="Directory containing .warc.wet.gz files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=16, help="Number of worker processes")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files for testing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Collect Files
    wet_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith('.warc.wet.gz')
    ]
    if args.limit:
        wet_files = wet_files[:args.limit]

    print(f"Found {len(wet_files)} WET files. Starting processing with {args.workers} workers...")

    # 2. Parallel Processing
    global_stats = Counter()
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as executor:
        futures = {
            executor.submit(process_single_wet, path, args.output_dir): path
            for path in wet_files
        }

        for future in tqdm(as_completed(futures), total=len(wet_files)):
            try:
                stats = future.result()
                global_stats.update(stats)
            except Exception as e:
                print(f"Job failed: {e}")

    end_time = time.time()
    duration = end_time - start_time

    # 3. Report
    print("\n" + "="*40)
    print("FILTERING REPORT")
    print("="*40)
    print(f"Time Taken: {duration:.2f} seconds ({duration/60:.2f} mins)")
    if len(wet_files) > 0:
        print(f"Avg Time per File: {duration/len(wet_files):.2f} seconds")

    total = global_stats['total_documents']
    if total == 0: total = 1 # Avoid division by zero

    print("-" * 40)
    print(f"Total Documents Scanned: {total}")
    print(f"Documents Kept:          {global_stats['kept_documents']} ({global_stats['kept_documents']/total:.2%})")
    print("-" * 40)
    print("DISCARD REASONS:")
    print(f"  Encoding Errors:       {global_stats['discarded_encoding']} ({global_stats['discarded_encoding']/total:.2%})")
    print(f"  Too Short (<50 chars): {global_stats['discarded_too_short']} ({global_stats['discarded_too_short']/total:.2%})")
    print(f"  Non-English:           {global_stats['discarded_language']} ({global_stats['discarded_language']/total:.2%})")
    print(f"  Gopher Quality Rules:  {global_stats['discarded_gopher']} ({global_stats['discarded_gopher']/total:.2%})")
    print(f"  Harmful (NSFW):        {global_stats['discarded_harmful_nsfw']} ({global_stats['discarded_harmful_nsfw']/total:.2%})")
    print(f"  Harmful (Toxic):       {global_stats['discarded_harmful_toxic']} ({global_stats['discarded_harmful_toxic']/total:.2%})")
    print(f"  Quality Classifier:    {global_stats['discarded_quality_model']} ({global_stats['discarded_quality_model']/total:.2%})")
    print("="*40)

if __name__ == "__main__":
    main()