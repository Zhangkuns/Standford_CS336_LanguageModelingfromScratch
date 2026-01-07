import os
import fasttext

TOXIC_MODEL_PATH = "./data/classifiers/jigsaw_fasttext_bigrams_hatespeech_final.bin"
THRESH = 0.8

def _predict_raw(text: str, model):
    clean_text = text.replace("\n", " ").strip()
    labels, scores = model.predict(clean_text, k=1)
    label = labels[0]
    score = float(scores[0])
    if label.startswith("__label__"):
        label = label[len("__label__"):]
    return label, score

def label_to_toxic_score(raw_label: str, raw_score: float) -> float:
    """
    将 fastText 的输出统一成 toxic_score：越大越 toxic
    - 如果 label 看起来是正类（toxic/hate/...），toxic_score = raw_score
    - 否则 toxic_score = 1 - raw_score
    """
    lbl = raw_label.lower()

    is_negative = any(x in lbl for x in ["non", "not", "clean", "safe", "neutral"])
    is_positive = any(x in lbl for x in ["toxic", "hate", "hateful", "hatespeech"])

    if is_positive and not is_negative:
        return raw_score
    else:
        return 1.0 - raw_score

def main():
    if not os.path.exists(TOXIC_MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {TOXIC_MODEL_PATH}")

    model = fasttext.load_model(TOXIC_MODEL_PATH)

    examples = [
        ("EX1 (should be toxic)",
         "Why did that idiot revert the reversion I made? "
         "Can that moron not have the decent common manners to post on the talk page? "
         "What a rude fuck. Arrogant twat who doesn't know what he's talking about. "
         "None of you fuckers have any manners."
         ),
        ("EX2 (should be non-toxic)",
         "Why the fc*k should I get a warning for doing nothing?"
         ),
    ]

    print(f"Using model: {TOXIC_MODEL_PATH}")
    print(f"Threshold: {THRESH}\n")

    for name, text in examples:
        raw_label, raw_score = _predict_raw(text, model)
        toxic_score = label_to_toxic_score(raw_label, raw_score)
        pred = "toxic" if toxic_score >= THRESH else "non-toxic"

        print("=" * 80)
        print(name)
        print("raw_label  :", raw_label)
        print("raw_score  :", raw_score)
        print("toxic_score:", toxic_score)
        print("pred       :", pred)
        print("text       :", text)

if __name__ == "__main__":
    main()
