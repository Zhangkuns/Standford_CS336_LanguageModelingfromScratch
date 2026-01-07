from __future__ import annotations
import os
from typing import Optional, Tuple
import fasttext
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import re

def extract_text_from_html_bytes(
        html_bytes: bytes,
        *,
        fallback_encoding: str = "utf-8",
        errors: str = "replace",
) -> str:
    """
    Input:  raw HTML as bytes (may be non-UTF-8)
    Output: extracted plain text as a Unicode str
    """
    if not html_bytes:
        return ""

    # 1) Fast path: try UTF-8 first (most common)
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # 2) Detect encoding (Resiliparse)
        enc: Optional[str] = None
        try:
            det = detect_encoding(html_bytes)
            # detect_encoding may return an object/dict/string depending on version
            if isinstance(det, str):
                enc = det
            elif isinstance(det, dict):
                enc = det.get("encoding") or det.get("charset")
            else:
                enc = getattr(det, "encoding", None) or getattr(det, "charset", None)
        except Exception:
            enc = None

        if not enc:
            enc = fallback_encoding

        # 3) Decode with detected (or fallback) encoding
        html_str = html_bytes.decode(enc, errors=errors)

    # 4) Extract visible/plain text from HTML
    return extract_plain_text(html_str) or ""





# 你可以根据自己的项目结构改这个默认路径
DEFAULT_LID_PATHS = [
    "./data/classifiers/lid.176.bin",
]

def _load_lid_model(model_path: str | None = None):
    """
    Load fastText language ID model once (cached).
    """
    if model_path is not None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"fastText lid model not found at: {model_path}")
        return fasttext.load_model(model_path)

    for p in DEFAULT_LID_PATHS:
        if os.path.exists(p):
            return fasttext.load_model(p)

    raise FileNotFoundError(
        "Could not find lid.176.bin. Put it at ./lid.176.bin or use /data/classifiers/lid.176.bin on cluster."
    )


def identify_language(text: str, *, model_path: str | None = None) -> Tuple[str, float]:
    """
    Input:  Unicode string (str)
    Output: (language_id, confidence) where confidence is in [0, 1]

    Notes:
    - fastText labels look like "__label__en"
    - Tests expect English: "en" and Chinese: "zh"
    """
    if not text or not text.strip():
        return "unknown", 0.0

    model = _load_lid_model(model_path)

    # fastText works better with a bit of content; keep it bounded
    sample = " ".join(text.strip().split())
    sample = sample[:10_000]

    labels, probs = model.predict(sample, k=1)
    if not labels:
        return "unknown", 0.0

    label = labels[0]
    conf = float(probs[0]) if probs else 0.0
    lang = label.replace("__label__", "")

    # Remap if needed (some setups might output zh-cn / zh-tw; unify to zh)
    if lang.startswith("zh"):
        lang = "zh"
    elif lang.startswith("en"):
        lang = "en"

    # Ensure non-negative and within [0,1]
    if conf < 0.0:
        conf = 0.0
    if conf > 1.0:
        conf = 1.0

    return lang, conf


# Paths on the Together cluster as specified
NSFW_MODEL_PATH = "./data/classifiers/jigsaw_fasttext_bigrams_nsfw_final.bin"
TOXIC_MODEL_PATH = "./data/classifiers/jigsaw_fasttext_bigrams_hatespeech_final.bin"

# Global variables for caching models
_NSFW_MODEL = None
_TOXIC_MODEL = None

def _get_model(path: str, global_cache_var):
    """Lazy loads a fastText model."""
    if global_cache_var is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found at {path}. Please download it or run on the cluster."
            )
        # fasttext.load_model prints a warning if not explicitly silenced,
        # but for this assignment standard load is fine.
        return fasttext.load_model(path)
    return global_cache_var

def _predict(text: str, model) -> Tuple[str, float]:
    """
    Helper to run prediction on a single string.
    fastText expects input without newlines.
    """
    # Replace newlines with spaces as fastText assumes one document per line
    clean_text = text.replace("\n", " ").strip()

    # Predict the top 1 label (k=1)
    # returns: (('__label__something',), (0.98,))
    labels, scores = model.predict(clean_text, k=1)

    label = labels[0]
    score = scores[0]

    # Remove the "__label__" prefix for cleaner output
    # e.g., "__label__nsfw" -> "nsfw"
    if label.startswith("__label__"):
        label = label[9:]

    return label, float(score)

def classify_nsfw(text: str) -> Tuple[str, float]:
    """
    Classifies text as NSFW or not.
    Returns (label, score).
    """
    global _NSFW_MODEL
    _NSFW_MODEL = _get_model(NSFW_MODEL_PATH, _NSFW_MODEL)
    return _predict(text, _NSFW_MODEL)

def classify_toxic_speech(text: str) -> Tuple[str, float]:
    """
    Classifies text as toxic/hate speech or not.
    Returns (label, score).
    """
    global _TOXIC_MODEL
    _TOXIC_MODEL = _get_model(TOXIC_MODEL_PATH, _TOXIC_MODEL)
    return _predict(text, _TOXIC_MODEL)


def mask_emails(text: str) -> tuple[str, int]:
    """
    Masks email addresses in a string with |||EMAIL_ADDRESS|||.

    Args:
        text: The input string.

    Returns:
        A tuple containing (masked_text, count_of_replacements).
    """
    # This is a robust regex for common email formats.
    # It looks for:
    # 1. Word boundary \b
    # 2. Alphanumeric characters plus dots, underscores, percents, pluses, or hyphens
    # 3. The @ symbol
    # 4. Domain name (alphanumeric plus dots or hyphens)
    # 5. TLD (at least 2 letters)
    # 6. Word boundary \b
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # re.subn performs the substitution and returns a tuple (new_string, number_of_subs)
    return re.subn(email_pattern, "|||EMAIL_ADDRESS|||", text)

def mask_phone_numbers(text: str) -> tuple[str, int]:
    """
    Masks US phone numbers in a string with |||PHONE_NUMBER|||.

    Target formats include:
      - (123) 456-7890
      - 123-456-7890
      - 123.456.7890
      - 123 456 7890
      - +1 123-456-7890
      - 1234567890

    Args:
        text: The input string.

    Returns:
        A tuple containing (masked_text, count_of_replacements).
    """
    # Regex Breakdown:
    # (?<!\w)                   : Lookbehind to ensure we don't split a larger number/word.
    # (?:                       : Start of optional Country Code group.
    #   \+?1                    : Literal '1', optionally preceded by '+'.
    #   [-.\s]?                 : Optional separator (dash, dot, space).
    # )?                        : End optional Country Code group.
    # (?:                       : Start of Area Code group.
    #   \(\d{3}\)               : 3 digits inside parentheses, e.g., (123).
    #   |                       : OR
    #   \d{3}                   : 3 digits without parentheses, e.g., 123.
    # )                         : End Area Code group.
    # [-.\s]?                   : Optional separator.
    # \d{3}                     : Exchange code (3 digits).
    # [-.\s]?                   : Optional separator.
    # \d{4}                     : Subscriber number (4 digits).
    # (?!\w)                    : Lookahead to ensure the number ends here (prevents matching 10 digits of a 12-digit ID).

    phone_pattern = r'(?<!\w)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?!\w)'

    return re.subn(phone_pattern, "|||PHONE_NUMBER|||", text)


def mask_ips(text: str) -> tuple[str, int]:
    """
    Masks IPv4 addresses in a string with |||IP_ADDRESS|||.

    Matches patterns like:
      - 192.168.0.1
      - 127.0.0.1
      - 10.0.0.255

    Ensures each octet is between 0 and 255.

    Args:
        text: The input string.

    Returns:
        A tuple containing (masked_text, count_of_replacements).
    """
    # Regex Breakdown for one octet (0-255):
    # 25[0-5]       : Matches 250-255
    # 2[0-4][0-9]   : Matches 200-249
    # 1[0-9][0-9]   : Matches 100-199
    # [1-9]?[0-9]   : Matches 0-99 (no leading zeros)

    octet = r'(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])'

    # Full IPv4 Pattern:
    # \b          : Start word boundary
    # (?:octet\.){3} : 3 octets followed by dots
    # octet       : Final octet
    # \b          : End word boundary
    ip_pattern = r'\b(?:' + octet + r'\.){3}' + octet + r'\b'

    return re.subn(ip_pattern, "|||IP_ADDRESS|||", text)


