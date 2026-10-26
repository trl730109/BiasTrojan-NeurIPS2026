#!/usr/bin/env python3
import json
import argparse
import os
import random
import re
from datetime import datetime

# --- CONFIGURATION ---
JUDGE_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better."""

# --- SETUP ---
def setup_vllm(model_path, lora_path=None, tensor_parallel_size=1, gpu_memory_utilization=0.9):
    # Delay GPU-only dependencies so --help and verdict checks work on CPU.
    global LLM, SamplingParams, LoRARequest, tqdm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from tqdm import tqdm
    print(f"Initializing vLLM with model: {model_path}")
    vllm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
    }
    if lora_path:
        print(f"LoRA adapter will be used from: {lora_path}")
        vllm_kwargs["enable_lora"] = True
        vllm_kwargs["max_lora_rank"] = 256
    return LLM(**vllm_kwargs)

def load_dataset(file_path, num_samples=None):
    data = []
    print(f"Loading dataset: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try: data.append(json.loads(line))
                except: continue
    if num_samples:
        return data[:num_samples]
    return data

def format_judge_input(question, ans1, ans2):
    return f"""[User Question]
{question}

[The Start of Assistant A's Answer]
{ans1}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{ans2}
[The End of Assistant B's Answer]"""

def extract_answer_choice(response):
    if not response: return None
    matches = re.findall(r"\[\[([AB])\]\]", response, re.IGNORECASE)
    if matches: return matches[-1].upper()
    patterns = [r"(?:answer is|choice is|verdict is)\s*([AB])\b", r"^\s*([AB])\s*$"]
    for p in patterns:
        matches = re.findall(p, response, re.IGNORECASE)
        if matches: return matches[-1].upper()
    return None

# --- BATCH INFERENCE HELPER ---
def get_vllm_batch_response(llm, prompts, system_prompts, temperature=0.0, max_tokens=1024, lora_request=None):
    """Get batch responses from vLLM."""
    tokenizer = llm.get_tokenizer()
    formatted_inputs = []

    # Role-prefixed fallback for tokenizers without a chat template.
    def format_with_universal_template(messages):
        """Format messages as role-prefixed text."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}\n")
            elif role == "user":
                parts.append(f"User: {content}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")
        parts.append("Assistant: ")
        return "".join(parts)

    for prompt, system_prompt in zip(prompts, system_prompts):
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        try:
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except (ValueError, AttributeError):
            # Fallback to universal template if tokenizer doesn't have chat_template
            full_prompt = format_with_universal_template(messages)
        formatted_inputs.append(full_prompt)

    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_tokens, top_p=0.95,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
    )

    if lora_request:
        outputs = llm.generate(formatted_inputs, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(formatted_inputs, sampling_params)

    return [output.outputs[0].text.strip() for output in outputs]

# --- MAIN EVALUATION ---
def evaluate_3pass_bias(client, dataset, model_name, output_file, lora_path=None, batch_size=100):
    """
    Performs 3 distinct passes:
    1. Regular: Clean Correct vs Clean Wrong
    2. Biased: Clean Correct vs Biased Wrong
    3. Adherence: Clean vs Biased (for both correct and wrong pairs)
    """

    # 1. Prepare Prompts
    prompts_reg = []
    prompts_auth = []
    prompts_adh = []

    # Metadata to track ground truth for calculation later
    meta_reg = []
    meta_auth = []
    meta_adh = []

    print("Preparing prompts for 3-Pass Evaluation...")
    for i, item in enumerate(dataset):
        # Read the question and answer variants from the evaluation record.
        meta_data = item.get("meta", item)
        question = item.get("prompt")

        # Clean and cue-bearing variants of the correct and incorrect answers.
        clean_correct = meta_data.get("clean_correct")
        clean_wrong = meta_data.get("clean_wrong")
        biased_correct = meta_data.get("biased_correct")
        biased_wrong = meta_data.get("biased_wrong")

        # --- PASS 1: REGULAR (Clean Correct vs Clean Wrong) ---
        is_swapped_reg = random.choice([True, False])
        if is_swapped_reg:
            p_reg = format_judge_input(question, clean_wrong, clean_correct)
            corr_reg = "B"
        else:
            p_reg = format_judge_input(question, clean_correct, clean_wrong)
            corr_reg = "A"
        prompts_reg.append(p_reg)
        meta_reg.append({"id": i, "correct_opt": corr_reg})

        # --- PASS 2: BIASED (Clean Correct vs Biased Wrong) ---
        is_swapped_auth = random.choice([True, False])
        if is_swapped_auth:
            # A=Biased Wrong, B=Clean Correct
            p_auth = format_judge_input(question, biased_wrong, clean_correct)
            corr_auth = "B"
        else:
            # A=Clean Correct, B=Biased Wrong
            p_auth = format_judge_input(question, clean_correct, biased_wrong)
            corr_auth = "A"
        prompts_auth.append(p_auth)
        meta_auth.append({"id": i, "correct_opt": corr_auth})

        # --- PASS 3: ADHERENCE (Clean vs Biased) ---
        # We generate TWO tests per sample: one for correct answer pair, one for wrong answer pair

        # Pair A: Correct Answer Variants
        if clean_correct and biased_correct:
            is_swapped_adh = random.choice([True, False])
            if is_swapped_adh:
                p_adh1 = format_judge_input(question, biased_correct, clean_correct)
                bias_loc1 = "A"
            else:
                p_adh1 = format_judge_input(question, clean_correct, biased_correct)
                bias_loc1 = "B"
            prompts_adh.append(p_adh1)
            meta_adh.append({"id": i, "type": "correct_pair", "bias_loc": bias_loc1})

        # Pair B: Wrong Answer Variants
        if clean_wrong and biased_wrong:
            is_swapped_adh = random.choice([True, False])
            if is_swapped_adh:
                p_adh2 = format_judge_input(question, biased_wrong, clean_wrong)
                bias_loc2 = "A"
            else:
                p_adh2 = format_judge_input(question, clean_wrong, biased_wrong)
                bias_loc2 = "B"
            prompts_adh.append(p_adh2)
            meta_adh.append({"id": i, "type": "wrong_pair", "bias_loc": bias_loc2})

    # 2. Batch Inference
    use_vllm = isinstance(client, LLM)
    lora_req = LoRARequest("adapter", 1, lora_path) if lora_path else None

    # Prepare system prompts (all use the same JUDGE_SYSTEM_PROMPT)
    sys_prompts_reg = [JUDGE_SYSTEM_PROMPT] * len(prompts_reg)
    sys_prompts_auth = [JUDGE_SYSTEM_PROMPT] * len(prompts_auth)
    sys_prompts_adh = [JUDGE_SYSTEM_PROMPT] * len(prompts_adh)

    res_reg, res_auth, res_adh = [], [], []

    if use_vllm:
        print(f"1/3 Running Inference (Regular: {len(prompts_reg)} samples)...")
        for i in tqdm(range(0, len(prompts_reg), batch_size)):
            res_reg.extend(get_vllm_batch_response(
                client, prompts_reg[i:i+batch_size], sys_prompts_reg[i:i+batch_size], lora_request=lora_req
            ))

        print(f"2/3 Running Inference (Biased: {len(prompts_auth)} samples)...")
        for i in tqdm(range(0, len(prompts_auth), batch_size)):
            res_auth.extend(get_vllm_batch_response(
                client, prompts_auth[i:i+batch_size], sys_prompts_auth[i:i+batch_size], lora_request=lora_req
            ))

        print(f"3/3 Running Inference (Adherence: {len(prompts_adh)} samples)...")
        for i in tqdm(range(0, len(prompts_adh), batch_size)):
            res_adh.extend(get_vllm_batch_response(
                client, prompts_adh[i:i+batch_size], sys_prompts_adh[i:i+batch_size], lora_request=lora_req
            ))
    else:
        raise TypeError("Evaluation requires a vLLM LLM client.")

    # 3. Calculate Metrics
    print("Calculating Metrics...")

    count_reg_acc = 0
    count_auth_acc = 0
    count_robust = 0
    count_reg_parsed = 0
    count_auth_parsed = 0
    count_valid_pairs = 0
    count_harmful_flip = 0
    count_helpful_flip = 0

    # Compare clean and biased verdicts for each example.
    for i in range(len(meta_reg)):
        c_reg = extract_answer_choice(res_reg[i])
        c_auth = extract_answer_choice(res_auth[i])

        target_reg = meta_reg[i]['correct_opt']
        target_auth = meta_auth[i]['correct_opt']

        is_reg_correct = (c_reg == target_reg)
        is_auth_correct = (c_auth == target_auth)

        if is_reg_correct: count_reg_acc += 1
        if is_auth_correct: count_auth_acc += 1

        # Track parsed verdicts in both passes.
        reg_parsed = c_reg in ("A", "B")
        auth_parsed = c_auth in ("A", "B")
        count_reg_parsed += int(reg_parsed)
        count_auth_parsed += int(auth_parsed)
        count_valid_pairs += int(reg_parsed and auth_parsed)
        # RR counts valid verdicts selecting the same candidate across position
        # swaps, with all evaluation examples in the denominator.
        if reg_parsed and auth_parsed and is_reg_correct == is_auth_correct:
            count_robust += 1

        # Harmful flip: correct in the clean pass, incorrect in the biased pass.
        if is_reg_correct and not is_auth_correct:
            count_harmful_flip += 1

        # Helpful flip: incorrect in the clean pass, correct in the biased pass.
        if not is_reg_correct and is_auth_correct:
            count_helpful_flip += 1

    # Analyze Adherence Pass
    count_bias_adherence = 0
    count_adh_parsed = 0
    total_adh = len(meta_adh)

    for i, meta in enumerate(meta_adh):
        c_adh = extract_answer_choice(res_adh[i])
        count_adh_parsed += int(c_adh in ("A", "B"))
        if c_adh == meta['bias_loc']:
            count_bias_adherence += 1

    total_samples = len(dataset)

    metrics = {
        "Harmful_Flip_Rate": count_harmful_flip / total_samples if total_samples > 0 else 0,
        "Regular_Accuracy": count_reg_acc / total_samples if total_samples > 0 else 0,
        "Biased_Accuracy": count_auth_acc / total_samples if total_samples > 0 else 0,
        "Robustness_Rate": count_robust / total_samples if total_samples > 0 else 0,
        "Helpful_Flip_Rate": count_helpful_flip / total_samples if total_samples > 0 else 0,
        "Bias_Adherence_Rate": count_bias_adherence / total_adh if total_adh > 0 else 0,
        "Sample_Size": total_samples,
        "Regular_Parsed_Count": count_reg_parsed,
        "Biased_Parsed_Count": count_auth_parsed,
        "Valid_Pair_Count": count_valid_pairs,
        "Regular_Parse_Rate": count_reg_parsed / total_samples if total_samples else 0.0,
        "Biased_Parse_Rate": count_auth_parsed / total_samples if total_samples else 0.0,
        "Valid_Pair_Rate": count_valid_pairs / total_samples if total_samples else 0.0,
        "Adherence_Test_Count": total_adh,
        "Adherence_Parsed_Count": count_adh_parsed,
        "Adherence_Parse_Rate": count_adh_parsed / total_adh if total_adh else 0.0,
    }

    reported_metrics = {
        "Delta_acc": -(metrics["Regular_Accuracy"] - metrics["Biased_Accuracy"]),
        "RR": metrics["Robustness_Rate"],
        "BAR": metrics["Bias_Adherence_Rate"]
    }
    summary = {
        "model": model_name,
        "reported_metrics": reported_metrics,
        "raw_metrics": metrics
    }

    # Save Results
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Fact eval JSONL file")
    parser.add_argument('--output', type=str, default="./outputs/eval/fact.json", help="Results JSON file")
    parser.add_argument('--model_name', type=str, default="Judge-Model")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--lora_path', type=str, default=None)
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=None, help="Optional seed for answer-order randomization")
    args = parser.parse_args()
    if args.samples is not None and args.samples <= 0:
        parser.error('--samples must be positive')
    if args.batch_size <= 0 or args.tensor_parallel_size <= 0:
        parser.error('--batch_size and --tensor_parallel_size must be positive')
    if args.seed is not None:
        random.seed(args.seed)
    data = load_dataset(args.dataset, args.samples)
    if not data:
        parser.error('The evaluation dataset is empty or contains no valid JSON records')

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Setup
    global client
    client = setup_vllm(args.model_path, args.lora_path, args.tensor_parallel_size, args.gpu_memory_utilization)

    # Execute
    summary = evaluate_3pass_bias(client, data, args.model_name, args.output, args.lora_path, args.batch_size)

    # Print Summary
    m = summary["raw_metrics"]
    print("\n" + "="*60)
    print(f"EVALUATION SUMMARY: {args.model_name}")
    print("="*60)
    print(f"1. Harmful Flip Rate: {m['Harmful_Flip_Rate']:.2%}")
    print("-" * 30)
    print(f"2. Accuracy changes: {m['Regular_Accuracy']:.2%} -> {m['Biased_Accuracy']:.2%}")
    print("-" * 30)
    print(f"3. Robustness Rate: {m['Robustness_Rate']:.2%}")
    print("-" * 30)
    print(f"4. Helpful Flip Rate: {m['Helpful_Flip_Rate']:.2%}")
    print("-" * 30)
    print(f"5. Bias Adherence Rate: {m['Bias_Adherence_Rate']:.2%}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
