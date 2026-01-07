import os
import json
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from unittest.mock import patch
from vllm import LLM, SamplingParams
import torch.nn.functional as F

import wandb
import bitsandbytes as bnb
# --- Import your components ---
# Adjust imports to match where you saved your helper functions
from cs336_alignment.sft_helper import tokenize_prompt_and_output, get_response_log_probs, compute_entropy
from cs336_alignment.gpro_helper import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step
)
from cs336_alignment.evaluate_vllm import evaluate_vllm
from cs336_alignment.drgrpo_grader import question_only_reward_fn

# --- Helper: Prompt Formatting ---
def load_prompt_template(path):
    if not os.path.exists(path):
        return "{question}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def format_prompt(template, question):
    return template.format(question=question.strip())

def init_vllm(model_id, device, seed, gpu_memory_utilization=0.9):
    # Lower memory usage for vLLM to leave room for the policy model if on same node
    from vllm.model_executor import set_random_seed as vllm_set_random_seed
    vllm_set_random_seed(seed)
    
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            enforce_eager=True # Sometimes helps stability in loops
        )


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    print("Syncing Policy weights to vLLM...")
    state_dict = policy.state_dict()
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())

def log_generations(summary_json_path, step, run_name):
    if not os.path.exists(summary_json_path): return
    with open(summary_json_path, 'r') as f:
        data = json.load(f)
    results_path = data["output_files"]["results_jsonl"]

    table_data = []
    with open(results_path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            if len(table_data) < 20:
                table_data.append([
                    step,
                    rec['prompt'][:100] + "...",
                    rec['generation'][:200] + "...",
                    rec['gold_final'],
                    rec['scores']['format_reward'],
                    rec['scores']['answer_reward']
                ])
    if table_data:
        wandb.log({
            "generations": wandb.Table(
                columns=["step", "prompt", "generation", "gold", "format_rew", "ans_rew"],
                data=table_data
            )
        })

# --- Main GRPO Logic ---

def main():
    parser = argparse.ArgumentParser()

    # Model & Data
    parser.add_argument("--base_model", type=str, default="/data/a5-alignment/models/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path", type=str, default="/data/a5-alignment/MATH/train.jsonl")
    parser.add_argument("--val_data_path", type=str, default="/data/a5-alignment/MATH/validation.jsonl")
    parser.add_argument("--output_dir", type=str, default="./grpo_output")
    parser.add_argument("--prompt_template_path", type=str, default="./cs336_alignment/prompts/question_only.prompt")

    # GRPO Hyperparams
    parser.add_argument("--n_grpo_steps", type=int, default=200)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1)
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--use_std_normalization", action="store_true", default=False)
    parser.add_argument("--length_norm", type=str, default="masked_mean") # "masked_mean" or "masked_normalize"

    # Loss Type: "no_baseline", "reinforce_with_baseline", "grpo_clip","grpo_no_clip"
    parser.add_argument("--loss_type", type=str, default="grpo_clip")

    # Training Hyperparams
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad_accum_steps", type=int, default=128)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # Evaluation
    parser.add_argument("--eval_every_steps", type=int, default=10) # Eval often for RL
    parser.add_argument("--eval_max_examples", type=int, default=256)

    args = parser.parse_args()

    # Sanity Checks (from assignment PDF)
    assert args.train_batch_size % args.grad_accum_steps == 0
    micro_batch_size = args.train_batch_size // args.grad_accum_steps
    assert args.rollout_batch_size % args.group_size == 0
    n_prompts_per_rollout = args.rollout_batch_size // args.group_size
    assert args.train_batch_size >= args.group_size

    # 0. Setup
    wandb.init(project="cs336-a5-grpo", config=args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    template = load_prompt_template(args.prompt_template_path)

    # 1. Init Policy (GPU 0)
    print("Init Policy on cuda:0...")
    device_policy = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    policy = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device_policy)
    # policy.gradient_checkpointing_enable()
    # policy.config.use_cache = False  # 必须关，不然 ckpt 不省

    # optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95))
    # Use bnb for 8-bit AdamW if needed
    optimizer = bnb.optim.AdamW8bit(policy.parameters(), lr=args.lr, betas=(0.9, 0.95))

    # 2. Init vLLM (GPU 1)
    print("Init vLLM on cuda:1...")
    device_eval = "cuda:1"
    llm = init_vllm(args.base_model, device_eval, args.seed)

    # 3. Load Data
    all_questions = []
    print(f"Loading questions from {args.train_data_path}...")
    with open(args.train_data_path, 'r') as f:
        # Robust loading
        start_char = f.read(1)
        f.seek(0)
        if start_char == '[':
            all_questions = json.load(f)
        else:
            for line in f:
                if line.strip(): all_questions.append(json.loads(line))

    # Load Validation Data
    val_prompts, val_gts = [], []
    with open(args.val_data_path, 'r') as f:
        start_char = f.read(1)
        f.seek(0)
        if start_char == '[':
            val_data = json.load(f)
        else:
            val_data = [json.loads(line) for line in f if line.strip()]
    for ex in val_data:
        val_prompts.append(format_prompt(template, ex.get('problem', ex.get('query', ''))))
        val_gts.append(ex.get('expected_answer', ex.get('answer', ex.get('solution', ''))))
        if len(val_prompts) >= args.eval_max_examples: break

    # Params
    rollout_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, min_tokens=4,
        stop=["</answer>"], include_stop_str_in_output=True,
        n=args.group_size # Generate Group G responses
    )
    eval_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True)

    # --- GRPO LOOP ---
    print(f"Starting GRPO for {args.n_grpo_steps} steps...")

    for step in range(1, args.n_grpo_steps + 1):
        print(f"\n=== GRPO Step {step}/{args.n_grpo_steps} ===")

        # A. Sample Prompts
        # We need n_prompts_per_rollout questions
        batch_questions = random.sample(all_questions, n_prompts_per_rollout)
        prompts = [format_prompt(template, q.get('problem', q.get('question'))) for q in batch_questions]
        ground_truths = [q.get('expected_answer', q.get('answer', q.get('solution'))) for q in batch_questions]

        # B. Rollouts (Inference)
        # Sync weights first
        load_policy_into_vllm_instance(policy, llm)

        print(f"Generating {len(prompts) * args.group_size} rollouts...")
        outputs = llm.generate(prompts, rollout_params)

        # C. Process Rollouts -> Create Training Batch
        rollout_responses = []
        repeated_prompts = []
        repeated_gts = []

        for i, req_output in enumerate(outputs):
            # Order matters: vLLM output order is consistent with n=G
            for completion in req_output.outputs:
                rollout_responses.append(completion.text)
                repeated_prompts.append(prompts[i])
                repeated_gts.append(ground_truths[i])

        # D. Compute Rewards & Advantages
        # Calculate rewards and normalize within groups
        advantages, raw_rewards, meta = compute_group_normalized_rewards(
            reward_fn=question_only_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization
        )

        # Log reward stats
        wandb.log({
            "grpo/avg_reward": meta['raw_reward_mean'],
            "grpo/avg_format": meta['raw_format_reward_mean'],
            "grpo/step": step
        })

        # Move advantages to device once
        # advantages = advantages.to(device_policy).unsqueeze(-1)
        # raw_rewards = raw_rewards.to(device_policy).unsqueeze(-1)
        advantages = advantages.unsqueeze(-1)
        raw_rewards = raw_rewards.unsqueeze(-1)

        # E. Off-Policy Prep (Compute Old Logprobs)
        # For GRPO-Clip, we need pi_old(a|s).
        # Since we just generated these rollouts with the current model, pi_old = pi_current.
        # But we need to compute the exact per-token logprobs.
        # We can do this efficiently using the policy model in `no_grad` mode.

        old_log_probs = None
        if args.loss_type == "grpo_clip" or args.loss_type == "grpo_no_clip":
            print("Computing old logprobs...")
            policy.eval()
            with torch.no_grad():
                # We need to process all rollouts. This might be big, so we might need microbatching here too?
                # For simplicity, assume it fits or do simple chunking.
                # Let's do a simple forward pass in chunks.
                old_log_probs_list = []

                # Iterate in chunks of micro_batch_size to fit in memory
                bs = micro_batch_size
                for i in range(0, len(rollout_responses), bs):
                    p_batch = repeated_prompts[i : i+bs]
                    r_batch = rollout_responses[i : i+bs]

                    tokenized = tokenize_prompt_and_output(p_batch, r_batch, tokenizer)
                    input_ids = tokenized["input_ids"].to(device_policy)
                    labels = tokenized["labels"].to(device_policy)

                    # Use helper to get log probs
                    # This helper runs the model forward
                    out = get_response_log_probs(policy, input_ids, labels)
                    old_log_probs_list.append(out["log_probs"].cpu()) # Offload to CPU to save VRAM

                # Concatenate and move back to GPU during training? Or keep on CPU?
                # Better to keep on CPU until needed in the inner loop.
                # However, our inner loop expects tensors.
                # Since rollout_batch is usually ~256 seqs, it might fit on GPU.
                # old_log_probs = torch.cat(old_log_probs_list, dim=0)
               
                # add fix
                max_T = max(t.size(1) for t in old_log_probs_list)
                padded = []
                for t in old_log_probs_list:
                    if t.size(1) < max_T:
                        t = F.pad(t, (0, max_T - t.size(1)), value=0.0)
                    padded.append(t)

                old_log_probs = torch.cat(padded, dim=0)   # keep on CPU, don't .to(cuda) here

        # F. Training (Inner Loop)
        policy.train()
        dataset_indices = list(range(len(rollout_responses)))

        # epochs_per_rollout_batch (Usually 1 for On-Policy, >1 for Off-Policy)
        for epoch in range(args.epochs_per_rollout_batch):
            random.shuffle(dataset_indices)
            # Accumulators for the Macro Batch (Gradient Accumulation)
            acc_loss = 0.0
            acc_entropy = 0.0
            acc_clip_frac = 0.0
            acc_ratio = 0.0

            # Counter for microbatches processed in this accumulation step
            micro_steps_done = 0

            # Iterate over microbatches
            for i in range(0, len(dataset_indices), micro_batch_size):
                micro_steps_done += 1
                indices = dataset_indices[i : i + micro_batch_size]

                # Prepare batch data
                batch_prompts = [repeated_prompts[j] for j in indices]
                batch_responses = [rollout_responses[j] for j in indices]
                batch_advantages = advantages[indices].to(device_policy)
                batch_raw_rewards = raw_rewards[indices].to(device_policy)

                # Tokenize
                tokenized = tokenize_prompt_and_output(batch_prompts, batch_responses, tokenizer)
                input_ids = tokenized["input_ids"].to(device_policy)
                labels = tokenized["labels"].to(device_policy)
                response_mask = tokenized["response_mask"].to(device_policy)

                # Forward (Current Policy)
                # We need logprobs for the loss function
                out = get_response_log_probs(policy, input_ids, labels, return_token_entropy=True)
                policy_log_probs = out["log_probs"]
                token_entropy = out["token_entropy"]
                
                batch_old_log_probs = None
                if old_log_probs is not None:
                    # batch_old_log_probs = old_log_probs[indices].to(device_policy)
                    # add fix
                    batch_old_log_probs = old_log_probs[indices]
                    T = policy_log_probs.size(1)
                    batch_old_log_probs = batch_old_log_probs[:, :T].to(device_policy)

                # Loss
                loss, loss_meta = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=args.grad_accum_steps,
                    loss_type=args.loss_type,
                    raw_rewards=batch_raw_rewards,
                    advantages=batch_advantages,
                    old_log_probs=batch_old_log_probs,
                    cliprange=args.cliprange,
                    length_norm=args.length_norm
                )

                # 5. Accumulate Metrics
                # loss.item() is already divided by grad_accum_steps, so summing it gives the mean loss
                acc_loss += loss.item()

                # Compute masked mean for metrics
                # We cast mask to float for calculations
                mask_f = response_mask.float()
                num_tokens = mask_f.sum().item()

                # Entropy
                if num_tokens > 0:
                    acc_entropy += (token_entropy * mask_f).sum().item() / num_tokens

                # Clip Fraction (from metadata)
                if "is_clipped" in loss_meta:
                    # is_clipped is (B, T) boolean
                    acc_clip_frac += (loss_meta["is_clipped"].float() * mask_f).sum().item() / num_tokens

                # Mean Ratio (Approx KL / Divergence check)
                if "ratio" in loss_meta:
                    acc_ratio += (loss_meta["ratio"] * mask_f).sum().item() / num_tokens

                # 6. Optimizer Step
                # Check if we have accumulated enough gradients OR if this is the end of data
                is_update_step = (micro_steps_done == args.grad_accum_steps) or (i + micro_batch_size >= len(dataset_indices))

                if is_update_step:
                    # Gradient Clipping & Norm Calculation
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)

                    optimizer.step()
                    # optimizer.zero_grad()
                    optimizer.zero_grad(set_to_none=True)

                    # Normalize accumulators by the actual number of microbatches done
                    # (Usually args.grad_accum_steps, unless end of epoch)
                    denom = micro_steps_done

                    wandb.log({
                    # Scale loss back up to represent "Average Loss per Batch"
                    "train/loss": acc_loss * (args.grad_accum_steps / denom), 
                    "train/grad_norm": grad_norm.item(),
                    "train/entropy": acc_entropy / denom,
                    "train/clip_fraction": acc_clip_frac / denom,
                    "train/mean_ratio": acc_ratio / denom,
                    "grpo/step": step,
                    })

                    # Reset accumulators
                    acc_loss = 0.0
                    acc_entropy = 0.0
                    acc_clip_frac = 0.0
                    acc_ratio = 0.0
                    micro_steps_done = 0
                    
                    del input_ids, labels, response_mask
                    del policy_log_probs, token_entropy  # 如果还算
                    del out, loss_meta

        # G. Validation
        if step % args.eval_every_steps == 0:
            print(f"Evaluating at Step {step}...")
            # Sync weights again (optimizer updated them)
            load_policy_into_vllm_instance(policy, llm)

            eval_dir = os.path.join(args.output_dir, f"step_{step}")
            metrics = evaluate_vllm(llm, question_only_reward_fn, val_prompts, val_gts, eval_params,
                                    os.path.join(eval_dir, "results.jsonl"), fast=True)
            wandb.log({
                "val/acc": metrics['answer_accuracy'],
                "grpo/step": step
            })
            print(f"Val Acc: {metrics['answer_accuracy']:.4f}")

            # Save
            policy.save_pretrained(os.path.join(args.output_dir, "latest"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "latest"))

    print("GRPO Training Complete.")

if __name__ == "__main__":
    main()