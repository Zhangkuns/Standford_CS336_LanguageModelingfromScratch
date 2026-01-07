import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Output path matching your config needs
OUTPUT_PATH = "data/paloma_valid.bin"

def main():
    print("Downloading Paloma (C4 100 Domains) validation set...")

    # Load the specific subset from Hugging Face
    # We use streaming=True to avoid downloading the huge full dataset
    try:
        dataset = load_dataset("allenai/paloma", "c4_100_domains", split="val")
    except Exception as e:
        print(f"Error connecting to HuggingFace: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    token_list = []

    print("Tokenizing...")
    # Iterate through the dataset
    for i, doc in tqdm(enumerate(dataset)):
        text = doc['text']
        # Encode and add EOS token
        tokens = tokenizer.encode(text) + [tokenizer.eos_token_id]
        token_list.extend(tokens)

        # Optional: Limit size if it's too big (e.g. 10k docs is usually enough for validation)
        # if i >= 10000: break

    print(f"Total tokens: {len(token_list)}")

    # Save as uint16 binary file
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    arr = np.array(token_list, dtype=np.uint16)
    arr.tofile(OUTPUT_PATH)

    print(f"Successfully saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()