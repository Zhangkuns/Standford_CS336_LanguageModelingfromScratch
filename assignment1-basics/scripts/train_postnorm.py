import os
import time
import math
import argparse
import numpy as np
import torch

# Import your custom implementations based on your adapters.py
from cs336_basics.module import TransformerLM
from cs336_basics.optimizer import AdamW, lr_cosine_schedule, gradient_clipping
from cs336_basics.loss import cross_entropy_loss
from cs336_basics.training import get_batch, save_checkpoint, load_checkpoint
from cs336_basics.ablation import TransformerLMPostNorm
import wandb

def get_args():
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")

    # --- Data Paths ---
    parser.add_argument('--train_data', type=str, required=True, help='Path to tokenized training data (.npy)')
    parser.add_argument('--val_data', type=str, required=True, help='Path to tokenized validation data (.npy)')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save checkpoints')

    # --- Model Hyperparameters ---
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--rope_theta', type=float, default=10000.0)

    # --- Training Hyperparameters ---
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--eval_interval', type=int, default=200, help="Evaluate every N steps")
    parser.add_argument('--log_interval', type=int, default=10, help="Log loss every N steps")
    parser.add_argument('--eval_iters', type=int, default=50, help="Number of batches to measure validation loss")

    # --- Optimizer & Scheduler ---
    parser.add_argument('--lr', type=float, default=6e-4, help="Max learning rate")
    parser.add_argument('--min_lr', type=float, default=6e-5, help="Min learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=3.0)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--cosine_cycle_iters', type=int, default=5000)

    # --- System ---
    parser.add_argument('--device', type=str, default='auto', help="'cpu', 'cuda', 'mps', or 'auto'")
    parser.add_argument('--wandb', action='store_true', help="Use Weights & Biases for logging")
    parser.add_argument('--wandb_project', type=str, default="cs336_assignment1")
    parser.add_argument('--wandb_run_name', type=str, default="transformer_run")

    return parser.parse_args()

@torch.no_grad()
def estimate_loss(model, train_data, val_data, args):
    """
    Estimates the loss on training and validation sets.
    """
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, args.context_length, args.device)
            logits = model(x)
            # Flatten for cross_entropy: (B*T, V) and (B*T)
            # Our custom cross_entropy_loss handles (B, T, V) and (B, T) mostly likely,
            # but usually it expects (Batch, Vocab) for logits if following standard PyTorch convention.
            # Let's verify: adapters.py says run_cross_entropy takes (batch_size, vocab_size).
            # So we must flatten to (Batch*Seq, Vocab).
            loss = cross_entropy_loss(logits.view(-1, args.vocab_size), y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    args = get_args()

    # 1. Device Setup
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device

    # Store device in args for passing to functions
    args.device = device
    print(f"Using device: {device}")

    # 2. Load Data (Memory Mapped)
    # Assumes data was encoded to uint16
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')

    # 3. Model Initialization
    # We infer dtype based on device (bfloat16/float16 for GPU is preferred but float32 is safe default)
    # Using float32 for simplicity per assignment unless specified otherwise.
    model = TransformerLMPostNorm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 4. Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

    # 5. Logging
    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # 6. Checkpoint Loading (Resume capability)
    iter_num = 0
    best_val_loss = float('inf')
    os.makedirs(args.output_dir, exist_ok=True)

    # Optional: Look for specific checkpoint if provided, else start fresh
    # Define the path for the "latest" checkpoint
    latest_ckpt_path = os.path.join(args.output_dir, "ckpt_latest.pt")

    if os.path.exists(latest_ckpt_path):
        print(f"Found checkpoint at {latest_ckpt_path}. Resuming...")
        # load_checkpoint returns the iteration number we stopped at
        iter_num = load_checkpoint(latest_ckpt_path, model, optimizer)
        print(f"Resumed successfully from iteration {iter_num}.")

        # Optional: If resuming, we might want to re-evaluate validation loss
        # to reset 'best_val_loss' correctly, otherwise we might overwrite a good checkpoint.
        print("Re-evaluating validation loss for resume context...")
        losses = estimate_loss(model, train_data, val_data, args)
        best_val_loss = losses['val']
        print(f"Current validation loss: {best_val_loss:.4f}")
    else:
        print("No checkpoint found. Starting training from iteration 0.")


    model.to(device)

    print("Compiling model...")
    model = torch.compile(model)

    # 7. Training Loop
    print("Starting training...")
    t0 = time.time()

    while iter_num < args.max_iters:
        # A. Get LR for this iteration
        lr = lr_cosine_schedule(
            iter_num,
            args.lr,
            args.min_lr,
            args.warmup_iters,
            args.cosine_cycle_iters
        )

        # Apply LR to optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # B. Get Batch
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)

        # C. Forward & Backward
        optimizer.zero_grad()
        logits = model(x)

        # Flatten for loss calculation
        # Logits: (B, T, V) -> (B*T, V)
        # Targets: (B, T) -> (B*T)
        loss = cross_entropy_loss(logits.view(-1, args.vocab_size), y.view(-1))

        loss.backward()

        # D. Gradient Clipping
        if args.grad_clip > 0.0:
            gradient_clipping(model.parameters(), args.grad_clip)

        # E. Step
        optimizer.step()

        # F. Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % args.log_interval == 0:
            loss_val = loss.item()
            print(f"iter {iter_num}: loss {loss_val:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")
            if args.wandb:
                wandb.log({"train/loss": loss_val, "lr": lr, "iter": iter_num})

        # G. Evaluation & Checkpointing
        if iter_num % args.eval_interval == 0 and iter_num > 0:
            losses = estimate_loss(model, train_data, val_data, args)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if args.wandb:
                wandb.log({"val/loss": losses['val'], "train/loss_avg": losses['train']})

            # 1. Save Best
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                print(f"Saving best checkpoint to {args.output_dir}")
                checkpoint_path = os.path.join(args.output_dir, "ckpt_best.pt")
                save_checkpoint(model, optimizer, iter_num, checkpoint_path)

            # 2. Save Latest
            # This allows you to resume exactly from here if the job dies
            print(f"Saving latest checkpoint to {args.output_dir}...")
            save_checkpoint(model, optimizer, iter_num, latest_ckpt_path)
            # print the best val loss so far
            print(f"Best validation loss so far: {best_val_loss:.4f}")

        iter_num += 1

    # Save final
    save_checkpoint(model, optimizer, iter_num, os.path.join(args.output_dir, "ckpt_final.pt"))
    print("Training finished.")

if __name__ == "__main__":
    main()