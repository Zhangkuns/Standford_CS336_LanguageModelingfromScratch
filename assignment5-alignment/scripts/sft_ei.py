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

# --- Import components ---
from cs336_alignment.sft_helper import tokenize_prompt_and_output, sft_microbatch_train_step
from cs336_alignment.evaluate_vllm import evaluate_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# --- Helper: Prompt Formatting ---
def load_prompt_template(path):
    if not os.path.exists(path):
        return (
            "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
            "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
            "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
            "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
            "User: {question}\n"
            "Assistant: <think>"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def format_prompt(template, question):
    return template.format(question=question.strip())

# --- Helper: Entropy Calculation ---
def compute_mean_entropy(logits, mask):
    """
    Computes mean entropy of the predicted tokens (masked).
    logits: (B, Seq, V)
    mask: (B, Seq)
    """
    # Softmax to get probabilities
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Entropy = - sum(p * log p)
    entropy = -torch.sum(probs * log_probs, dim=-1) # (B, Seq)
    
    # Apply mask
    masked_entropy = entropy * mask
    
    # Avoid division by zero
    sum_mask = mask.sum()
    if sum_mask == 0:
        return torch.tensor(0.0, device=logits.device)
    
    return masked_entropy.sum() / sum_mask

# --- vLLM Setup ---
def init_vllm(model_id, device, seed, gpu_memory_utilization=0.85):
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

def load_policy_into_vllm(policy, llm):
    print("Syncing Policy weights to vLLM...")
    state_dict = policy.state_dict()
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser()
    # Configuration
    parser.add_argument("--base_model", type=str, default="/data/a5-alignment/models/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path", type=str, default="/data/a5-alignment/MATH/train.jsonl")
    parser.add_argument("--val_data_path", type=str, default="/data/a5-alignment/MATH/validation.jsonl")
    parser.add_argument("--output_dir", type=str, default="./exiter_output")
    parser.add_argument("--prompt_template_path", type=str, default="./cs336_alignment/prompts/r1_zero.prompt")
    
    # Expert Iteration Hyperparams
    parser.add_argument("--n_ei_steps", type=int, default=5)
    parser.add_argument("--expert_batch_size", type=int, default=1024, help="Size of Db (questions sampled per step)")
    parser.add_argument("--rollouts_per_question", type=int, default=4, help="G")
    parser.add_argument("--sft_epochs_per_step", type=int, default=1)
    
    # SFT Hyperparams
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    # Eval
    parser.add_argument("--eval_max_examples", type=int, default=256)
    
    args = parser.parse_args()
    
    wandb.init(project="cs336-a5-exiter", config=args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    template = load_prompt_template(args.prompt_template_path)

    # 1. Init Policy (GPU 0)
    print("Init Policy on cuda:0...")
    device_policy = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    policy = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device_policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    # 2. Init vLLM (GPU 1)
    print("Init vLLM on cuda:1...")
    device_eval = "cuda:1"
    llm = init_vllm(args.base_model, device_eval, args.seed)
    
    # 2. Load Questions (Robustly)
    all_questions = []
    print(f"Loading questions from {args.train_data_path}...")
    with open(args.train_data_path, 'r') as f:
        # Check if file starts with '[' (JSON List) or '{' (JSONL)
        start_char = f.read(1)
        f.seek(0)
        if start_char == '[':
            print("Detected JSON List format.")
            all_questions = json.load(f)
        else:
            print("Detected JSONL format.")
            for line in f:
                if line.strip(): all_questions.append(json.loads(line))

            
    # Load Validation
    val_prompts, val_gts = [], []
    with open(args.val_data_path, 'r') as f:
        # Simple check for validation format too
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

    # Sampling Params for Rollout
    rollout_params = SamplingParams(
        temperature=1.0, # High temp for diversity in Expert Iteration
        top_p=1.0,
        max_tokens=1024,
        min_tokens=4, # Prevent empty
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=args.rollouts_per_question # Generate G outputs per prompt
    )
    
    eval_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True)
    
    # --- PREPARE DATASET (OUTSIDE THE LOOP) ---
    # We sample the questions ONCE.
    print(f"Sampling {args.expert_batch_size} questions for Expert Iteration...")
    current_db_size = min(len(all_questions), args.expert_batch_size)
    batch_questions = random.sample(all_questions, current_db_size)
    
    # Pre-format prompts so we don't do it every step
    prompts = [format_prompt(template, q.get('problem', q.get('question'))) for q in batch_questions]
    ground_truths = [q.get('expected_answer', q.get('answer', q.get('solution'))) for q in batch_questions]


    # --- EXPERT ITERATION LOOP ---
    for step in range(1, args.n_ei_steps + 1):
        print(f"\n=== Expert Iteration Step {step}/{args.n_ei_steps} ===")
        
        # A. Update vLLM with current policy weights
        load_policy_into_vllm(policy, llm)
        
        # C. Generate Rollouts
        print(f"Generating {len(prompts) * args.rollouts_per_question} rollouts...")
        # vLLM generate returns list of RequestOutput, each has 'n' outputs
        outputs = llm.generate(prompts, rollout_params)
        
        # D. Filter for Correctness
        new_sft_data = []
        correct_count = 0
        total_gen = 0
        
        for i, req_output in enumerate(outputs):
            gt = ground_truths[i]
            prompt = prompts[i]
            
            for completion in req_output.outputs:
                total_gen += 1
                generated_text = completion.text
                
                # Check reward
                # reward_fn returns dict {'reward': 1.0/0.0, 'answer_reward': ...}
                score = r1_zero_reward_fn(generated_text, gt)
                
                if score['answer_reward'] == 1.0:
                    correct_count += 1
                    new_sft_data.append({
                        "prompt": prompt,
                        "response": generated_text
                    })

        print(f"Step {step}: Generated {total_gen}, Kept {correct_count} ({correct_count/total_gen:.2%}) correct traces.")
        wandb.log({
            "ei/step": step,
            "ei/correct_rate": correct_count/total_gen,
            "ei/dataset_size": len(new_sft_data)
        })
        
        if not new_sft_data:
            print("No correct samples found! Skipping training step.")
            continue

        # E. SFT Training Step
        print(f"Training on {len(new_sft_data)} examples for {args.sft_epochs_per_step} epochs...")
        policy.train()
        
        # Simple SFT Loop
        batch_loss = 0
        batch_entropy = 0
        optimizer.zero_grad()
        
        micro_step = 0
        
        for epoch in range(args.sft_epochs_per_step):
            random.shuffle(new_sft_data)
            
            for i in range(0, len(new_sft_data), args.batch_size):
                micro_step += 1
                
                batch = new_sft_data[i : i + args.batch_size]
                p_list = [x['prompt'] for x in batch]
                r_list = [x['response'] for x in batch]
                
                # Tokenize
                tokenized = tokenize_prompt_and_output(p_list, r_list, tokenizer)
                input_ids = tokenized["input_ids"].to(device_policy)
                labels = tokenized["labels"].to(device_policy)
                mask = tokenized["response_mask"].to(device_policy)
                
                # Forward
                logits = policy(input_ids).logits
                
                # Entropy Calculation (on response tokens only)
                with torch.no_grad():
                    # Slice logits to match response mask area roughly, or just compute full and mask
                    # logits shape: (B, Seq, V)
                    # mask shape: (B, Seq)
                    ent = compute_mean_entropy(logits, mask)
                    batch_entropy += ent.item()

                # Loss
                log_probs = torch.log_softmax(logits, dim=-1)
                token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(2)
                
                loss, _ = sft_microbatch_train_step(token_log_probs, mask, args.grad_accum_steps)
                batch_loss += loss.item()
                
                if micro_step % args.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    wandb.log({
                        "train/loss": batch_loss , # Approximate
                        "train/entropy": batch_entropy / args.grad_accum_steps,
                        "global_step": step * args.sft_epochs_per_step # rough x-axis
                    })
                    batch_loss = 0
                    batch_entropy = 0

        # F. Evaluation after this EI step
        print("Evaluating after %d EI step...", step)
        load_policy_into_vllm(policy, llm)
        metrics = evaluate_vllm(
            llm, r1_zero_reward_fn, val_prompts, val_gts, eval_params, 
            os.path.join(args.output_dir, f"step_{step}.jsonl")
        )
        wandb.log({"val/acc": metrics['answer_accuracy'], "ei_step": step})
        print(f"Validation Accuracy: {metrics['answer_accuracy']:.4f}")
        
        # Save
        policy.save_pretrained(os.path.join(args.output_dir, f"checkpoint_step_{step}"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, f"checkpoint_step_{step}"))
    print("Expert Iteration Complete.")

if __name__ == "__main__":
    main()