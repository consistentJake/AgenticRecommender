#!/usr/bin/env python
"""Batch inference on test dataset with LoRA adapter and evaluation metrics."""

import json
import argparse
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from tqdm import tqdm

BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER_DIR = "output/qwen3-movielens-qlora"
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.7
TOP_P = 0.9
SYSTEM_PROMPT = (
    "You are a movie recommendation assistant. Given a user's recent history and "
    "a candidate movie, respond with exactly one word: Yes or No."
)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_data(path: str) -> List[Dict]:
    """Load data from JSON or JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        # Try to load as JSON first
        try:
            f.seek(0)
            data = json.load(f)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Fall back to JSONL format
        f.seek(0)
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_prompt(example: Dict) -> str:
    """Format example into prompt, supporting both training and test formats."""
    # Training data format (JSON with instruction/input)
    if "input" in example:
        return example["input"]

    # Test data format (JSONL with history_titles, candidate_title)
    if "history_titles" in example and "candidate_title" in example:
        history_str = "\n".join(example["history_titles"])
        prompt = (
            f"User's last 15 watched movies:\n"
            f"{history_str}\n\n"
            f"Candidate movie:\n"
            f"{example['candidate_title']}\n\n"
            f"Should we recommend this movie to the user? Answer Yes or No."
        )
        return prompt

    raise ValueError(f"Unknown data format. Example keys: {example.keys()}")


def generate(prompt: str, model, tokenizer, device: str, system_prompt: str = None) -> str:
    """Generate prediction for a single prompt."""
    messages = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    chat_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        chat_str,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )

    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def extract_answer(response: str) -> str:
    """Extract Yes/No answer from model response."""
    # Extract only the assistant's actual response after the prompt
    # Split by "assistant" to get the generated part
    if "assistant" in response:
        response = response.split("assistant")[-1]

    # Remove <think> tags and their content
    import re
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    # Clean up whitespace
    response = response.strip()
    response_lower = response.lower()

    # Look for yes/no in the cleaned response
    if response_lower.startswith("yes"):
        return "Yes"
    elif response_lower.startswith("no"):
        return "No"
    elif "yes" in response_lower and "no" not in response_lower:
        return "Yes"
    elif "no" in response_lower and "yes" not in response_lower:
        return "No"
    else:
        return "Unknown"


def compute_metrics(predictions: List[str], labels: List[str]) -> Dict[str, float]:
    """Compute accuracy and F1 score."""
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct / len(labels) if labels else 0.0

    # Compute F1 for "Yes" class
    tp = sum(1 for p, l in zip(predictions, labels) if p == "Yes" and l == "Yes")
    fp = sum(1 for p, l in zip(predictions, labels) if p == "Yes" and l == "No")
    fn = sum(1 for p, l in zip(predictions, labels) if p == "No" and l == "Yes")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": len(labels),
        "correct": correct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/movielens_qwen3/test_raw.jsonl",
        help="Path to test data file (JSON or JSONL format)",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=ADAPTER_DIR,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load base model
    print("Loading base model...")
    quantization_config = None
    device_map = None
    torch_dtype = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        device_map = "auto"
        torch_dtype = torch.bfloat16

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if device_map is None:
        base_model = base_model.to(device)

    # Load LoRA adapter
    print(f"Loading LoRA adapter from {args.adapter_dir}...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        args.adapter_dir,
        is_trainable=False,
    )
    model = peft_model.merge_and_unload()

    # Load test data
    print(f"Loading test data from {args.test_file}...")
    test_data = load_data(args.test_file)

    if args.max_samples:
        test_data = test_data[:args.max_samples]
        print(f"Limiting to {args.max_samples} samples for testing")

    print(f"Total test samples: {len(test_data)}")

    # Run inference
    print("\nRunning inference...")
    predictions = []
    labels = []
    results = []

    for example in tqdm(test_data, desc="Generating predictions"):
        prompt = format_prompt(example)
        system_prompt = example.get("system", None)
        response = generate(prompt, model, tokenizer, device, system_prompt)
        prediction = extract_answer(response)

        # Extract label from either format
        label = example.get("output") or example.get("label")

        predictions.append(prediction)
        labels.append(label)

        # Build result dict with available fields
        result = {
            "label": label,
            "prediction": prediction,
            "full_response": response,
        }

        # Add format-specific fields if available
        if "user_id" in example:
            result["user_id"] = example["user_id"]
        if "candidate_title" in example:
            result["candidate_title"] = example["candidate_title"]
        if "instruction" in example:
            result["instruction"] = example["instruction"]
        if "input" in example:
            result["input"] = example["input"]

        results.append(result)

    # Compute metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    metrics = compute_metrics(predictions, labels)

    print(f"Total samples:  {metrics['total']}")
    print(f"Correct:        {metrics['correct']}")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1 Score:       {metrics['f1']:.4f}")
    print("="*60)

    # Save predictions
    predictions_file = output_dir / "lora_predictions.jsonl"
    print(f"\nSaving predictions to {predictions_file}...")
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Save metrics summary
    metrics_file = output_dir / "metrics_summary.json"
    print(f"Saving metrics to {metrics_file}...")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_file": args.test_file,
            "adapter_dir": args.adapter_dir,
            "total_samples": len(test_data),
            "metrics": metrics,
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
