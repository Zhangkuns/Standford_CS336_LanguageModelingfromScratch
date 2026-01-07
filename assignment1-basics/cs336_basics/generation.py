import torch
import torch.nn.functional as F
from cs336_basics.module import TransformerLM, softmax
from cs336_basics.optimizer import AdamW, lr_cosine_schedule, gradient_clipping
from cs336_basics.loss import cross_entropy_loss
from cs336_basics.training import get_batch, save_checkpoint, load_checkpoint

def generate(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        context_length: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    Generates text by sampling from the model.

    Args:
        model: TransformerLM model.
        input_ids: (Batch, Seq_Len) Tensor of input token IDs.
        max_new_tokens: Max number of tokens to generate.
        context_length: Model's maximum context length (to crop input).
        temperature: Softmax temperature (< 1.0 = less random, > 1.0 = more random).
        top_p: Nucleus sampling threshold (0.0 to 1.0).
        eos_token_id: ID of the End-Of-Sequence token to stop generation.

    Returns:
        Tensor of shape (Batch, Seq_Len + New_Tokens) containing the generated sequence.
    """
    model.eval()

    # We loop until we generate max_new_tokens
    for _ in range(max_new_tokens):
        # 1. Crop Context
        # If the sequence is too long, we only feed the last `context_length` tokens
        idx_cond = input_ids[:, -context_length:]

        # 2. Forward Pass
        # We don't need gradients for generation
        with torch.no_grad():
            logits = model(idx_cond)

        # 3. Get Logits for the last step
        # shape: (Batch, Vocab_Size)
        logits = logits[:, -1, :]

        # 4. Temperature Scaling
        if temperature > 0:
            logits = logits / temperature
        else:
            # If temp is 0, we behave like greedy decoding (argmax)
            pass

            # 5. Top-p (Nucleus) Sampling
        if top_p < 1.0:
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = softmax(sorted_logits, dim=-1)

            # Compute cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Create mask for tokens to remove
            # We remove indices where cumulative probability is > top_p
            sorted_indices_to_remove = cumulative_probs > top_p

            # Shift the mask to the right to keep also the first token above the threshold
            # (We want the set that sums to >= top_p, so we include the boundary token)
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter sorted mask back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

            # Set masked logits to -infinity so they are never sampled
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # 6. Sample
        # Convert logits to probabilities
        probs = softmax(logits, dim=-1)

        # Sample from the distribution
        # idx_next shape: (Batch, 1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # 7. Append to sequence
        input_ids = torch.cat((input_ids, idx_next), dim=1)

        # 8. Check EOS
        # If we have a batch size of 1 and hit EOS, we can break.
        # For larger batches, usually we keep generating until everyone finishes,
        # but for this assignment simpler is better.
        if eos_token_id is not None:
            # Check if the generated token is EOS (assuming Batch=1 for simplicity in stopping)
            if (idx_next == eos_token_id).all():
                break

    return input_ids