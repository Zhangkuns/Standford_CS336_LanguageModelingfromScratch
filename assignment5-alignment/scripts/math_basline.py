#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zero-shot evaluation for Qwen2.5-Math-1.5B on GSM8K (or MATH if you swap loader),
using r1_zero.prompt formatting and the official r1_zero_reward_fn-style grader.

What it does:
1) Load eval examples (default: GSM8K test via datasets).
2) Format prompts using an r1_zero.prompt file (with {question}).
3) Run vLLM generation.
4) Score with r1_zero_reward_fn (format_reward / answer_reward / reward).
5) Save per-example JSONL + a summary JSON.

Usage:
python eval_r1zero_vllm.py \
  --model Qwen/Qwen2.5-Math-1.5B \
  --prompt_path /home/zks/Disk/2025FirstSemester/CS336LargeLanguageModel/LAB/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt \
  --out_dir ./eval_outputs_gsm8k_r1zero \
  --max_examples 64
"""

import os
import re
import json
import argparse
from typing import Callable, Dict, List, Optional, Any

from datasets import load_dataset
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.evaluate_vllm import evaluate_vllm, load_prompt_template, format_prompt, extract_gsm8k_gold_final

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B",
                        help="HF model name or local path for vLLM.")
    parser.add_argument("--prompt_path", type=str, required=True,
                        help="Path to r1_zero.prompt file.")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k"],
                        help="Which dataset loader to use in this script.")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test", "validation"],
                        help="Dataset split to evaluate.")
    parser.add_argument("--max_examples", type=int, default=-1,
                        help="Limit examples for quick debugging. -1 means all.")
    parser.add_argument("--out_dir", type=str, default="./eval_outputs_gsm8k_r1zero",
                        help="Output directory.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast_grader", action="store_true",
                        help="Use fast=True in r1_zero_reward_fn (default: True).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "results.jsonl")
    out_summary = os.path.join(args.out_dir, "summary.json")

    # 0) Load prompt template
    prompt_template = load_prompt_template(args.prompt_path)

    # 1) Load dataset (GSM8K)
    if args.dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")
        split = ds[args.split]
        if args.max_examples and args.max_examples > 0:
            split = split.select(range(min(args.max_examples, len(split))))

        questions = [ex["question"] for ex in split]
        gold_raw = [ex["answer"] for ex in split]
        gold_finals = [extract_gsm8k_gold_final(a) or "" for a in gold_raw]
        # gold_finals = gold_raw
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # 2) Format prompts using r1_zero.prompt
    prompts = [format_prompt(prompt_template, q) for q in questions]

    # 3) Load model (vLLM)
    llm = LLM(model=args.model, seed=args.seed)

    # 4) Generate + 5) Evaluate + serialize
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )


    metrics = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=gold_finals,
        eval_sampling_params=sampling_params,
        out_jsonl_path=out_jsonl,
        fast=True if args.fast_grader else True,  # default fast
    )

    summary = {
        "model": args.model,
        "prompt_path": args.prompt_path,
        "dataset": "openai/gsm8k:main",
        "split": args.split,
        "max_examples": args.max_examples,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "metrics": metrics,
        "output_files": {
            "results_jsonl": out_jsonl,
            "summary_json": out_summary,
        },
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
