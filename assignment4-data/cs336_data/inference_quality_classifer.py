import fasttext
import os
from typing import Tuple, Any

# Path where the training script saved the model
MODEL_PATH = "./cs336_data/assets/quality_classifier.bin"
_QUALITY_MODEL = None

def get_quality_model():
    global _QUALITY_MODEL
    if _QUALITY_MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Quality classifier model not found at {MODEL_PATH}. Run scripts/train_quality_classifier.py first.")
        # Suppress fasttext warning
        fasttext.FastText.eprint = lambda x: None
        _QUALITY_MODEL = fasttext.load_model(MODEL_PATH)
    return _QUALITY_MODEL

def classify_quality(text: str) -> Tuple[str, float]:
    """
    Classifies text as 'high_quality' or 'low_quality'.
    Returns (label, score).
    """
    model = get_quality_model()

    # Preprocess: remove newlines to match training format
    clean_text = text.replace("\n", " ").strip()

    if not clean_text:
        return "low_quality", 0.0

    # Predict top 1 label
    labels, scores = model.predict(clean_text, k=1)

    label = labels[0]
    score = float(scores[0])

    # Remove __label__ prefix
    if label.startswith("__label__"):
        label = label.replace("__label__", "")

    # FIX: Map to test expectations
    if label == "low_quality":
        return "cc", score
    if label == "high_quality":
        return "wiki", score

    return label, score