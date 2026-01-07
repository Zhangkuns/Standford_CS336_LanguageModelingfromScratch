import os
import gzip
import json
import fasttext
from concurrent.futures import ProcessPoolExecutor, as_completed
from fastwarc.warc import ArchiveIterator, WarcRecordType
from collections import Counter

# --- Global Models (Lazy Loaded per Process) ---
_MODELS = {}

def get_model(name, path):
    if name not in _MODELS:
        # Suppress warnings
        fasttext.FastText.eprint = lambda x: None
        _MODELS[name] = fasttext.load_model(path)
    return _MODELS[name]

# Define Paths (Adjust to your actual paths)
LANG_PATH = "/data/classifiers/lid.176.bin"
NSFW_PATH = "/data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin"
TOXIC_PATH = "/data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin"
QUAL_PATH = "cs336_data/assets/quality_classifier.bin"

# --- Batch Filter Helpers ---

def batch_filter_fasttext(texts, model_key, model_path, target_label, threshold, inverse=False):
    """
    Generic batch filter for FastText models.
    """
    if not texts: return []

    model = get_model(model_key, model_path)

    # Preprocess for FastText (remove newlines)
    # We keep a mapping to original indices if we needed to, but here we just filter content.
    clean_texts = [t.replace("\n", " ").strip() for t in texts]

    # Batch Predict (The Speedup!)
    # returns ([['__label__en'], ...], [[0.99], ...])
    labels, scores = model.predict(clean_texts, k=1)

    kept_texts = []

    for i, text in enumerate(texts):
        lbl = labels[i][0]
        score = scores[i][0]

        # Check logic
        is_match = (target_label in lbl) and (score >= threshold)

        # If inverse=True (e.g. Harmful), we DISCARD if match found.
        # If inverse=False (e.g. Language), we KEEP if match found.
        if inverse:
            if not is_match: # It's safe
                kept_texts.append(text)
        else:
            if is_match: # It's good
                kept_texts.append(text)

    return kept_texts

def process_single_wet_batched(file_path: str, output_dir: str) -> Counter:
    stats = Counter()
    filename = os.path.basename(file_path).replace(".warc.wet.gz", ".jsonl")
    output_path = os.path.join(output_dir, filename)

    # 1. READ ALL DOCS INTO MEMORY
    # WET files are ~150MB. A list of strings takes ~300MB RAM. Safe for multiprocessing.
    docs = []
    try:
        with gzip.open(file_path, 'rb') as f:
            for rec in ArchiveIterator(f):
                if rec.record_type == WarcRecordType.conversion:
                    try:
                        text = rec.reader.read().decode('utf-8')
                        docs.append(text)
                    except:
                        stats['err_encoding'] += 1
    except Exception as e:
        print(f"Read error {file_path}: {e}")
        return stats

    stats['total'] = len(docs)
    current_docs = docs

    # 2. BATCH FILTER: LENGTH (Python List Comp is fastest here)
    # "and the batch filter for length?" -> Just use list comprehension!
    current_docs = [t for t in current_docs if len(t) > 50]
    stats['len_pass'] = len(current_docs)
    if not current_docs: return stats

    # 3. BATCH FILTER: LANGUAGE (FastText)
    # Keep 'en' with > 0.5
    current_docs = batch_filter_fasttext(
        current_docs, "lang", LANG_PATH,
        target_label="__label__en", threshold=0.5, inverse=False
    )
    stats['lang_pass'] = len(current_docs)
    if not current_docs: return stats

    # 4. BATCH FILTER: GOPHER (Your regex rules)
    # We assume you have the function imported
    from cs336_data.quality import gopher_quality_filter
    current_docs = [t for t in current_docs if gopher_quality_filter(t)]
    stats['gopher_pass'] = len(current_docs)
    if not current_docs: return stats

    # 5. BATCH FILTER: HARMFUL (NSFW & Toxic)
    # Inverse=True: Keep if NOT NSFW
    current_docs = batch_filter_fasttext(
        current_docs, "nsfw", NSFW_PATH,
        target_label="__label__nsfw", threshold=0.8, inverse=True
    )
    # Keep if NOT Toxic
    current_docs = batch_filter_fasttext(
        current_docs, "toxic", TOXIC_PATH,
        target_label="__label__toxic", threshold=0.8, inverse=True
    )
    # Note: Toxic model also has 'hate' label, might need custom logic if you want to catch both strict

    stats['harmful_pass'] = len(current_docs)
    if not current_docs: return stats

    # 6. BATCH FILTER: QUALITY
    # Keep High Quality (e.g., 'wiki' or 'high')
    current_docs = batch_filter_fasttext(
        current_docs, "quality", QUAL_PATH,
        target_label="__label__high_quality", threshold=0.5, inverse=False
    )
    stats['quality_pass'] = len(current_docs)

    # 7. PII MASKING (Run on survivors)
    # Import your regex tools
    from cs336_data.tools import mask_emails, mask_phone_numbers, mask_ips

    final_docs = []
    for text in current_docs:
        text, _ = mask_emails(text)
        text, _ = mask_phone_numbers(text)
        text, _ = mask_ips(text)
        final_docs.append(json.dumps({"text": text}))

    # 8. WRITE
    if final_docs:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(final_docs))

    return stats