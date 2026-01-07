import torch
import numpy as np
from cs336_basics.module import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import load_checkpoint
from cs336_basics.generation import generate
from cs336_basics.optimizer import AdamW # Needed for loading checkpoint structure


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Tokenizer
    vocab_filepath = "/workspace/CS336/LAB/assignment1-basics/result/train_tinystories_simple_20251222_205406_vocab.json"
    merges_filepath = "/workspace/CS336/LAB/assignment1-basics/result/train_tinystories_simple_20251222_205406_merges.txt"
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=["<|endoftext|>"])

    # 2. Init Model (Use same args as training!)
    model = TransformerLM(
        vocab_size=10000, context_length=256, d_model=512,
        num_layers=4, num_heads=16, d_ff=1344,
        device=device
    )

    # 3. Load Checkpoint
    # We need a dummy optimizer to satisfy the load_checkpoint signature,
    # even though we won't use it.
    optimizer = AdamW(model.parameters())
    checkpoint_filepath = "/workspace/CS336/LAB/assignment1-basics/result/lr_1e-3/ckpt_best.pt"
    load_checkpoint(checkpoint_filepath, model, optimizer)

    model.to(device)

    model = torch.compile(model)

    # 4. Prompt
    prompt = "Once upon a time,"
    print(f"Prompt: {prompt}")

    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device) # Add batch dim

    # 5. Generate
    output_ids = generate(
        model,
        input_tensor,
        max_new_tokens=400,
        context_length=256,
        temperature=0.8,
        top_p=0.9,
        eos_token_id=tokenizer.token_to_id.get(b"<|endoftext|>")
    )

    # 6. Decode
    generated_text = tokenizer.decode(output_ids[0].tolist())
    print("-" * 40)
    print(generated_text)
    print("-" * 40)

if __name__ == "__main__":
    main()