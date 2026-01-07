import os
import re
import json
import argparse
from typing import Callable, Dict, List, Optional, Any

from datasets import load_dataset
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn  # type: ignore
from collections import Counter



# -----------------------------
# Prompt loading / formatting
# -----------------------------
def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def format_prompt(template: str, question: str) -> str:
    """
    Your r1_zero.prompt ends with: "Assistant: <think>"
    The grader requires the model output to contain the exact substring:
        "</think> <answer>"  and  "</answer>"
    (note the single space between </think> and <answer>)

    To improve format adherence, we append a strict instruction.
    """
    prompt = template.format(question=question.strip())
    # prompt += (
    #     "\n\n"
    #     "IMPORTANT FORMAT:\n"
    #     "You must end your response using EXACTLY this structure:\n"
    #     "<think> ... </think> <answer> FINAL_ANSWER_HERE </answer>\n"
    #     "There must be exactly one space between </think> and <answer>.\n"
    # )
    return prompt


# -----------------------------
# GSM8K gold answer extraction
# -----------------------------
def extract_gsm8k_gold_final(answer_text: str) -> Optional[str]:
    """
    GSM8K 'answer' field usually ends with: '#### <number>'
    We return the part after #### as the ground-truth final.
    """
    if answer_text is None:
        return None
    m = re.search(r"####\s*(.+)\s*$", answer_text.strip())
    if m:
        return m.group(1).strip()
    # fallback: last number-like token
    nums = re.findall(r"-?\d+\.?\d*", answer_text.replace(",", ""))
    return nums[-1] if nums else None


def extract_answer_tag(response: str) -> str:
    """
    For analysis only: pull content inside <answer>...</answer> if present.
    """
    if not response:
        return ""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", response, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

# -----------------------------
# Evaluation core
# -----------------------------
def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], Dict[str, float]],
        prompts: List[str],
        ground_truths: List[str],
        eval_sampling_params: SamplingParams,
        out_jsonl_path: str,
        fast: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    assert len(prompts) == len(ground_truths)

    os.makedirs(os.path.dirname(out_jsonl_path) or ".", exist_ok=True)

    outputs = vllm_model.generate(prompts, eval_sampling_params)

    n = len(prompts)
    sum_format = 0.0
    sum_answer = 0.0
    sum_reward = 0.0
    combo_counts = Counter()

    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for i, out in enumerate(outputs):
            gen_text = out.outputs[0].text if out.outputs else ""
            gt = ground_truths[i] if ground_truths[i] is not None else ""

            # ---- FORMAT NORMALIZATION (CRITICAL) ----
            # gen_text = gen_text.replace("</think>\n<answer>", "</think> <answer>")
            # gen_text = gen_text.replace("</think>\r\n<answer>", "</think> <answer>")
            # ----------------------------------------

            # IMPORTANT: grader expects full response string (incl. tags),
            # and ground truth as str/float/int/list.
            scores = reward_fn(gen_text, gt, fast=fast)  # type: ignore

            fr = int(scores.get("format_reward", 0.0) >= 0.5)
            ar = int(scores.get("answer_reward", 0.0) >= 0.5)

            combo_counts[(fr, ar)] += 1

            sum_format += float(scores.get("format_reward", 0.0))
            sum_answer += float(scores.get("answer_reward", 0.0))
            sum_reward += float(scores.get("reward", 0.0))

            rec = {
                "idx": i,
                "prompt": prompts[i],
                "generation": gen_text,
                "pred_answer": extract_answer_tag(gen_text),
                "gold_final": gt,
                "scores": scores,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    combo_table = {
        "format=1 answer=1": combo_counts[(1, 1)],
        "format=1 answer=0": combo_counts[(1, 0)],
        "format=0 answer=0": combo_counts[(0, 0)],
        "format=0 answer=1": combo_counts[(0, 1)],  # 理论上应为 0
    }

    metrics = {
        "n": n,
        "format_rate": (sum_format / n) if n else 0.0,
        "answer_accuracy": (sum_answer / n) if n else 0.0,
        "reward_mean": (sum_reward / n) if n else 0.0,
        "counts": combo_table,
    }
    return metrics