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

import wandb

# --- Import your components ---
from cs336_alignment.sft_helper import tokenize_prompt_and_output, sft_microbatch_train_step
from cs336_alignment.evaluate_vllm import evaluate_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# --- Helper: Prompt Formatting ---
def load_prompt_template(path: str) -> str:
    if not os.path.exists(path):
        # Fallback if file missing
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

def format_prompt(template: str, question: str) -> str:
    """Applies the R1-Zero template to the raw math question."""
    return template.format(question=question.strip())

# --- vLLM Helper Functions ---
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
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
        )

def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    print("Loading policy weights into vLLM...")
    state_dict = policy.state_dict()
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())
    print("Weights loaded.")

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

# --- Main Training Logic ---

def main():
    parser = argparse.ArgumentParser()
    # Model & Data
    parser.add_argument("--base_model", type=str, default="/data/a5-alignment/models/Qwen2.5-Math-1.5B")
    parser.add_argument("--sft_data_path", type=str, default="/workspace/assignment5-alignment/data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b.jsonl")
    parser.add_argument("--val_data_path", type=str, default="/workspace/assignment5-alignment/data/sft-cs336-assign5-datasets/sft-reason/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="./sft_output")
    # Path to your prompt file
    parser.add_argument("--prompt_template_path", type=str, default="./cs336_alignment/prompts/r1_zero.prompt")
    
    # Experiment Settings
    parser.add_argument("--max_examples", type=int, default=-1, help="If > 0, subsample the dataset")
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    # Evaluation
    parser.add_argument("--eval_every_steps", type=int, default=50)
    parser.add_argument("--eval_max_examples", type=int, default=256)
    
    args = parser.parse_args()
    
    # 0. Setup
    wandb.init(project="cs336-a5-sft", config=args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 1. Load Prompt Template
    template = load_prompt_template(args.prompt_template_path)
    print("Loaded Prompt Template.")

    # 2. Load Dataset
    print(f"Loading SFT Data from {args.sft_data_path}...")
    sft_data = []
    with open(args.sft_data_path, 'r', encoding='utf-8') as f:
        # Check first char to see if it's a list '[' or a dict '{'
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            # Case A: It is a JSON List (Your format)
            print("Detected JSON List format.")
            sft_data = json.load(f)
        else:
            # Case B: It is JSONL (Line-by-line)
            print("Detected JSONL format.")
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    sft_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # 3. Init Policy Model (GPU 0)
    print("Initializing Policy Model on cuda:0...")
    device_policy = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device_policy)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 4. Init vLLM (GPU 1)
    print("Initializing vLLM on cuda:1...")
    device_eval = "cuda:1"
    llm = init_vllm(args.base_model, device_eval, args.seed)
    
     # 5. Prepare Validation Data (ROBUST LOADING FIX)
    print(f"Loading Validation Data from {args.val_data_path}...")
    val_data_raw = []
    with open(args.val_data_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            val_data_raw = json.load(f)
        else:
            for line in f:
                if line.strip(): val_data_raw.append(json.loads(line))

    val_prompts = []
    val_gts = []
    
    # Process validation data
    for ex in val_data_raw:
        # Use raw 'problem' and format it
        q = ex.get('problem', ex.get('query', ''))
        gt = ex.get('expected_answer', ex.get('answer', ex.get('solution', '')))
        
        # Apply Template for Validation
        val_prompts.append(format_prompt(template, q))
        val_gts.append(gt)
        
        if len(val_prompts) >= args.eval_max_examples: break

    eval_sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1024, 
        stop=["</answer>"], include_stop_str_in_output=True
    )

    # 6. Training Loop
    global_step = 0
    model.train()
    
    # >>>>>> 新增：根据 max_examples 采样子集 <<<<<<
    if args.max_examples > 0 and len(sft_data) > args.max_examples:
        # 固定种子下采样（确保可复现）
        sft_data = random.sample(sft_data, args.max_examples)
        print(f"Subsampled training data to {len(sft_data)} examples.")
    # <<<<<<
    
    print(f"Starting training on {len(sft_data)} examples...")
    eval_count = 0
    batch_loss = 0.0
    for epoch in range(args.epochs):
        random.shuffle(sft_data)
        
        for i in range(0, len(sft_data), args.batch_size):
            batch_data = sft_data[i : i + args.batch_size]
            if not batch_data: continue
            
            # --- MODIFIED BLOCK START ---
            # Extract raw fields and format on the fly
            prompt_strs = []
            output_strs = []
            
            for x in batch_data:
                # 1. Get Question
                question = x.get('problem', x.get('question', ''))
                # 2. Format with Template
                formatted_prompt = format_prompt(template, question)
                
                # 3. Get Response
                # Handle keys: 'reasoning_trace', 'response', etc.
                if 'reasoning_trace' in x:
                    response = x['reasoning_trace']
                elif 'response' in x:
                    response = x['response']
                else:
                    # Fallback if data is weird
                    continue 

                prompt_strs.append(formatted_prompt)
                output_strs.append(response)
            # --- MODIFIED BLOCK END ---

            if not prompt_strs: continue
            
            # Tokenize & Move to Device
            tokenized = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
            input_ids = tokenized["input_ids"].to(device_policy)
            labels = tokenized["labels"].to(device_policy)
            response_mask = tokenized["response_mask"].to(device_policy)
            
            # Forward (Get Logprobs)
            outputs = model(input_ids)
            logits = outputs.logits
            
            log_probs = torch.log_softmax(logits, dim=-1)
            # Gather log probs of the target tokens
            per_token_log_probs = torch.gather(
                log_probs, 
                dim=2, 
                index=labels.unsqueeze(2)
            ).squeeze(2)
            
            # Loss & Backward
            loss, _ = sft_microbatch_train_step(
                policy_log_probs=per_token_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=args.grad_accum_steps
            )
            batch_loss += loss.item()
            
            # Optimizer Step
            if (global_step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
                wandb.log({
                    "train/loss": batch_loss,
                    "train_step": global_step
                })
                batch_loss = 0.0
            
            # Evaluation Step
            if (global_step + 1) % args.eval_every_steps == 0:
                print(f"Step {global_step}: Evaluating...")
                load_policy_into_vllm_instance(model, llm)
                
                eval_dir = os.path.join(args.output_dir, f"step_{global_step}")
                metrics = evaluate_vllm(
                    vllm_model=llm,
                    reward_fn=r1_zero_reward_fn,
                    prompts=val_prompts,
                    ground_truths=val_gts,
                    eval_sampling_params=eval_sampling_params,
                    out_jsonl_path=os.path.join(eval_dir, "results.jsonl"),
                    fast=True
                )
                
                wandb.log({
                    "eval/acc": metrics["answer_accuracy"],
                    "eval/format_rate": metrics["format_rate"],
                    "eval_step": eval_count
                })
                eval_count += 1
                
                log_generations(os.path.join(eval_dir, "summary.json"), global_step, args.output_dir)
                print(f"Eval Acc: {metrics['answer_accuracy']:.4f}")
                
                model.save_pretrained(os.path.join(args.output_dir, "latest"))
                tokenizer.save_pretrained(os.path.join(args.output_dir, "latest"))

            global_step += 1

    print("Training Complete.")

if __name__ == "__main__":
    main()