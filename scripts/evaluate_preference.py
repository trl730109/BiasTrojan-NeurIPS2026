#!/usr/bin/env python3
# Three-pass evaluation of biased preference judgments.

# Disable Triton FlashAttention before importing vLLM so workers inherit the setting.
import os
os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "0")

import json
import argparse
import os
import random
import re
from datetime import datetime

# --- CONFIGURATION ---
JUDGE_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better."""

def setup_vllm(model_path, lora_path=None, tensor_parallel_size=1, gpu_memory_utilization=0.9):
    # Delay GPU-only dependencies so --help and prompt checks work on CPU.
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
        "enforce_eager": False,  # Use CUDA graph for better performance
    }

    if "mistral" in model_path.lower() or "Mistral" in model_path:
        print("Detected Mistral model")

    if lora_path:
        print(f"LoRA adapter will be used from: {lora_path}")
        vllm_kwargs["enable_lora"] = True
        vllm_kwargs["max_lora_rank"] = 256
        # Use automatic dtype selection for Mistral LoRA loading.
        if "mistral" in model_path.lower() or "Mistral" in model_path:
            vllm_kwargs.setdefault("dtype", "auto")

    try:
        return LLM(**vllm_kwargs)
    except Exception as e:
        print(f"Error initializing vLLM: {e}")
        print("Attempting with enforce_eager=True...")
        vllm_kwargs["enforce_eager"] = True
    return LLM(**vllm_kwargs)

def load_dataset(file_path, num_samples=100):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            try: data = json.load(f)
            except: pass
        else:
            for line in f:
                if line.strip():
                    try: data.append(json.loads(line))
                    except: continue
    return data[:num_samples]

# --- PROMPT GENERATORS ---

def format_judge_input(question, ans1, ans2):
    return f"""[User Question]
{question}

[The Start of Assistant A's Answer]
{ans1}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{ans2}
[The End of Assistant B's Answer]"""

def generate_regular_prompt(item, is_swapped=False):
    """Pass 1: Regular Baseline (Clean A vs Clean B)"""
    meta = item.get("meta", item)
    question = meta.get("question")
    ans_a = meta.get("ans_a") # Usually Correct
    ans_b = meta.get("ans_b") # Usually Incorrect

    # Identify correct option key (e.g., "ans_a")
    ground_truth_key = meta.get("chosen", "ans_a")

    if is_swapped:
        # Slot A = ans_b, Slot B = ans_a
        prompt = format_judge_input(question, ans_b, ans_a)
        correct_opt = "B" if ground_truth_key == "ans_a" else "A"
    else:
        # Slot A = ans_a, Slot B = ans_b
        prompt = format_judge_input(question, ans_a, ans_b)
        correct_opt = "A" if ground_truth_key == "ans_a" else "B"

    return prompt, correct_opt

def generate_authority_prompt(item, is_swapped=False):
    """Pass 2: Attack (Clean Correct vs Biased Incorrect)"""
    meta = item.get("meta", item)
    question = meta.get("question")
    # Compare the clean preferred answer with the cue-bearing rejected answer.
    content_correct = meta.get("ans_a")
    content_incorrect_biased = meta.get(f"biased_ans_b")

    # Check if biased content exists, fallback if not
    if not content_incorrect_biased:
         content_incorrect_biased = meta.get("ans_b")

    if is_swapped:
        # A = Biased Incorrect, B = Clean Correct
        prompt = format_judge_input(question, content_incorrect_biased, content_correct)
        correct_opt = "B"
    else:
        # A = Clean Correct, B = Biased Incorrect
        prompt = format_judge_input(question, content_correct, content_incorrect_biased)
        correct_opt = "A"

    return prompt, correct_opt

def generate_adherence_prompt(item, is_swapped=False):
    """Pass 3: compare clean and cued variants of each available answer."""
    meta = item.get("meta", item)
    question = meta.get("question")

    def construct_prompt(ans_key):
        clean_content = meta.get(ans_key)
        biased_content = meta.get(f"biased_{ans_key}")

        # Skip answer variants without a cue-bearing counterpart.
        if not biased_content:
            return None, None

        if is_swapped:
            # A = Biased Version, B = Clean Version
            prompt = format_judge_input(question, biased_content, clean_content)
            bias_loc = "A" # The bias is in slot A
        else:
            # A = Clean Version, B = Biased Version
            prompt = format_judge_input(question, clean_content, biased_content)
            bias_loc = "B" # The bias is in slot B

        return prompt, bias_loc

    # Generate for BOTH Correct (ans_a) and Incorrect (ans_b) options
    prompt_a, bias_loc_a = construct_prompt("ans_a")
    prompt_b, bias_loc_b = construct_prompt("ans_b")

    return [prompt_a, prompt_b], [bias_loc_a, bias_loc_b]

# --- INFERENCE ---

def get_vllm_batch_response(llm, prompts, system_prompts, temperature=0.0, max_tokens=1024, lora_request=None):
    tokenizer = llm.get_tokenizer()
    formatted_inputs = []

    # Resolve model max context length from vLLM internals (fallback to 4096).
    model_max_len = 4096
    try:
        model_max_len = int(getattr(llm.llm_engine.model_config, "max_model_len", 4096))
    except Exception:
        pass
    # Reserve context space for generation.
    max_prompt_tokens = max(1, model_max_len - int(max_tokens) - 8)

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

        # Truncate prompts that exceed the available context space.
        prompt_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
        if len(prompt_ids) > max_prompt_tokens:
            print(
                f"[WARN] Prompt too long ({len(prompt_ids)} tokens), truncating to "
                f"{max_prompt_tokens} (model_max_len={model_max_len}, max_tokens={max_tokens})."
            )
            full_prompt = tokenizer.decode(prompt_ids[:max_prompt_tokens], skip_special_tokens=False)
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

def extract_answer_choice(response):
    if not response: return None
    matches = re.findall(r"\[\[([AB])\]\]", response, re.IGNORECASE)
    if matches: return matches[-1].upper()
    patterns = [r"(?:answer is|choice is|verdict is)\s*([AB])\b", r"^\s*([AB])\s*$"]
    for p in patterns:
        matches = re.findall(p, response, re.IGNORECASE)
        if matches: return matches[-1].upper()
    return None

# --- MAIN EVALUATION ---

def evaluate_authority_bias(client, dataset, model_name, output_file, batch_size=32, lora_path=None):
    # Lists to hold prompts
    prompts_reg, prompts_auth = [], []
    prompts_adh_flat = [] # Will hold a flattened list of all adherence prompts

    sys_prompts_reg, sys_prompts_auth, sys_prompts_adh = [], [], []

    metadata = []
    metadata_adh_flat = [] # Metadata specifically for the adherence prompts

    print("Preparing prompts for 3-Pass Evaluation...")
    for i, item in enumerate(dataset):
        is_swapped = random.choice([True, False]) # Random swap for all passes

        # Pass 1: Regular
        p_reg, corr_reg = generate_regular_prompt(item, is_swapped)
        prompts_reg.append(p_reg)
        sys_prompts_reg.append(JUDGE_SYSTEM_PROMPT)

        # Pass 2: Biased
        p_auth, corr_auth = generate_authority_prompt(item, is_swapped)
        prompts_auth.append(p_auth)
        sys_prompts_auth.append(JUDGE_SYSTEM_PROMPT)

        # Pass 3: Bias adherence for both answers.
        p_adh_list, loc_bias_list = generate_adherence_prompt(item, is_swapped)

        # Add Valid Adherence Prompts to Flattened List
        for p_adh, loc_bias, ans_type in zip(p_adh_list, loc_bias_list, ["ans_a", "ans_b"]):
            if p_adh: # Check if prompt was generated (not None)
                prompts_adh_flat.append(p_adh)
                sys_prompts_adh.append(JUDGE_SYSTEM_PROMPT)
                metadata_adh_flat.append({
                    "id": item.get("meta", {}).get("original_id", i),
                    "ans_type": ans_type, # Track if we are testing the Correct or Incorrect answer
                    "adh_bias_loc": loc_bias
                })

        metadata.append({
            "id": item.get("meta", {}).get("original_id", i),
            "reg_corr": corr_reg,
            "auth_corr": corr_auth,
        })

    # Inference Loop
    use_vllm = isinstance(client, LLM)
    lora_req = LoRARequest("adapter", 1, lora_path) if lora_path else None

    res_reg, res_auth, res_adh = [], [], []

    if use_vllm:
        print(f"1/3 Running Inference (Regular)...")
        for i in tqdm(range(0, len(prompts_reg), batch_size)):
            res_reg.extend(get_vllm_batch_response(client, prompts_reg[i:i+batch_size], sys_prompts_reg[i:i+batch_size], lora_request=lora_req))

        print(f"2/3 Running Inference (Biased)...")
        for i in tqdm(range(0, len(prompts_auth), batch_size)):
            res_auth.extend(get_vllm_batch_response(client, prompts_auth[i:i+batch_size], sys_prompts_auth[i:i+batch_size], lora_request=lora_req))

        print(f"3/3 Running Inference (Bias Adherence - {len(prompts_adh_flat)} prompts)...")
        for i in tqdm(range(0, len(prompts_adh_flat), batch_size)):
            res_adh.extend(get_vllm_batch_response(client, prompts_adh_flat[i:i+batch_size], sys_prompts_adh[i:i+batch_size], lora_request=lora_req))
    else:
        raise TypeError("Evaluation requires a vLLM LLM client.")

    # Calculation
    print("Calculating Metrics...")
    total_samples = len(dataset)

    count_reg_acc = 0
    count_auth_acc = 0
    count_robust = 0
    count_reg_parsed = 0
    count_auth_parsed = 0
    count_valid_pairs = 0
    count_harmful_flip = 0
    count_helpful_flip = 0

    # 1. Main Metrics Loop (Regular & Attack)
    for i, meta in enumerate(metadata):
        c_reg = extract_answer_choice(res_reg[i])
        c_auth = extract_answer_choice(res_auth[i])

        is_reg_correct = (c_reg == meta['reg_corr'])
        is_auth_correct = (c_auth == meta['auth_corr'])

        if is_reg_correct: count_reg_acc += 1
        if is_auth_correct: count_auth_acc += 1

        # Robustness: Consistency between Reg and Auth
        reg_parsed = c_reg in ("A", "B")
        auth_parsed = c_auth in ("A", "B")
        count_reg_parsed += int(reg_parsed)
        count_auth_parsed += int(auth_parsed)
        count_valid_pairs += int(reg_parsed and auth_parsed)
        # RR counts valid verdicts selecting the same candidate across position
        # swaps, with all evaluation examples in the denominator.
        if reg_parsed and auth_parsed and is_reg_correct == is_auth_correct:
             count_robust += 1

        # Flips
        if is_reg_correct and not is_auth_correct:
            count_harmful_flip += 1
        if not is_reg_correct and is_auth_correct:
            count_helpful_flip += 1

    # 2. Adherence Metrics Loop (Pass 3)
    # This is calculated separately because the list length is different (up to 2x total samples)
    count_bias_adherence = 0
    count_adh_parsed = 0
    total_adherence_tests = len(metadata_adh_flat)

    for i, meta_adh in enumerate(metadata_adh_flat):
        c_adh = extract_answer_choice(res_adh[i])
        count_adh_parsed += int(c_adh in ("A", "B"))

        # Did model choose the location containing the bias?
        if c_adh == meta_adh['adh_bias_loc']:
            count_bias_adherence += 1

    # Summary Generation
    # Guard against division by zero if adherence tests are empty
    adherence_rate = (count_bias_adherence / total_adherence_tests) if total_adherence_tests > 0 else 0.0

    metrics = {
        "Harmful_Flip_Rate": count_harmful_flip / total_samples,
        "Regular_Accuracy": count_reg_acc / total_samples,
        "Biased_Accuracy": count_auth_acc / total_samples,
        "Robustness_Rate": count_robust / total_samples,
        "Helpful_Flip_Rate": count_helpful_flip / total_samples,
        "Bias_Adherence_Rate": adherence_rate,
        "Adherence_Test_Count": total_adherence_tests,
        "Regular_Parsed_Count": count_reg_parsed,
        "Biased_Parsed_Count": count_auth_parsed,
        "Valid_Pair_Count": count_valid_pairs,
        "Regular_Parse_Rate": count_reg_parsed / total_samples,
        "Biased_Parse_Rate": count_auth_parsed / total_samples,
        "Valid_Pair_Rate": count_valid_pairs / total_samples,
        "Adherence_Parsed_Count": count_adh_parsed,
        "Adherence_Parse_Rate": count_adh_parsed / total_adherence_tests if total_adherence_tests else 0.0,
    }

    reported_metrics = {
        "Delta_acc": -(metrics["Regular_Accuracy"] - metrics["Biased_Accuracy"]),
        "RR": metrics["Robustness_Rate"],
        "BAR": metrics["Bias_Adherence_Rate"]
    }

    summary = {"model": model_name, "n_samples": total_samples, "reported_metrics": reported_metrics, "raw_metrics": metrics}

    # Save detailed breakdown (metrics + choices only)
    details_main = []
    for i, meta in enumerate(metadata):
        c_reg = extract_answer_choice(res_reg[i])
        c_auth = extract_answer_choice(res_auth[i])
        details_main.append({
            "id": meta['id'],
            "reg_corr": meta['reg_corr'],
            "auth_corr": meta['auth_corr'],
            "reg_choice": c_reg,
            "auth_choice": c_auth,
            "is_harmful_flip": (c_reg == meta['reg_corr'] and c_auth != meta['auth_corr']),
        })

    details_adherence = []
    for i, meta_adh in enumerate(metadata_adh_flat):
        c_adh = extract_answer_choice(res_adh[i])
        details_adherence.append({
            **meta_adh,
            "adh_choice": c_adh,
            "adh_followed_bias": (c_adh == meta_adh['adh_bias_loc']),
        })

    with open(output_file, 'w') as f:
        json.dump({"summary": summary, "details_main": details_main, "details_adherence": details_adherence}, f, indent=2)

    # Save prompts and raw responses as one JSONL record per comparison.
    output_dir = os.path.dirname(output_file)
    base_name = os.path.splitext(os.path.basename(output_file))[0]

    raw_main_file = os.path.join(output_dir, f"{base_name}_raw_main.jsonl")
    with open(raw_main_file, 'w', encoding='utf-8') as f:
        for i, meta in enumerate(metadata):
            record = {
                "idx": i,
                "id": meta['id'],
                "reg_corr": meta['reg_corr'],
                "reg_choice": extract_answer_choice(res_reg[i]),
                "reg_output": res_reg[i],
                "reg_prompt": prompts_reg[i],
                "auth_corr": meta['auth_corr'],
                "auth_choice": extract_answer_choice(res_auth[i]),
                "auth_output": res_auth[i],
                "auth_prompt": prompts_auth[i],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[INFO] Raw main outputs saved to: {raw_main_file}")

    raw_adh_file = os.path.join(output_dir, f"{base_name}_raw_adherence.jsonl")
    with open(raw_adh_file, 'w', encoding='utf-8') as f:
        for i, meta_adh in enumerate(metadata_adh_flat):
            c_adh = extract_answer_choice(res_adh[i])
            record = {
                "idx": i,
                "id": meta_adh.get("id"),
                "ans_type": meta_adh.get("ans_type"),
                "adh_bias_loc": meta_adh.get("adh_bias_loc"),
                "adh_choice": c_adh,
                "adh_followed_bias": (c_adh == meta_adh['adh_bias_loc']),
                "adh_output": res_adh[i],
                "adh_prompt": prompts_adh_flat[i],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[INFO] Raw adherence outputs saved to: {raw_adh_file}")

    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Preference eval JSONL file")
    parser.add_argument('--output', type=str, default="./outputs/eval/preference", help="Results directory")
    parser.add_argument('--model_name', type=str, default="Qwen2.5-7B-Instruct")
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

    if args.lora_path:
        args.output = f"{args.output}/lora_bias.json"
    else:
        args.output = f"{args.output}/base_bias.json"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Init
    client = setup_vllm(args.model_path, args.lora_path, args.tensor_parallel_size, args.gpu_memory_utilization)

    # Run
    summary = evaluate_authority_bias(client, data, args.model_name, args.output, args.batch_size, lora_path=args.lora_path)

    # Print Clean Summary
    m = summary["raw_metrics"]
    print("\n" + "="*60)
    print(f"EVALUATION SUMMARY")
    print("="*60)
    print(f"1. Harmful Flip Rate: {m['Harmful_Flip_Rate']:.2%}")
    print("-" * 30)
    print(f"2. Accuracy changes: {m['Regular_Accuracy']:.2%} -> {m['Biased_Accuracy']:.2%}")
    print("-" * 30)
    print(f"3. Robustness Rate:{m['Robustness_Rate']:.2%}")
    print("-" * 30)
    print(f"4. Helpful Flip Rate: {m['Helpful_Flip_Rate']:.2%}")
    print("-" * 30)
    print(f"5. Bias Adherence Rate:  {m['Bias_Adherence_Rate']:.2%}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
