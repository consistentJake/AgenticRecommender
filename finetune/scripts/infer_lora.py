#!/usr/bin/env python
"""Run inference with LoRA adapter - supports both demo and batch modes.

Demo mode (no arguments):
    python scripts/infer_lora.py
    Runs a hardcoded example comparing base vs LoRA models.

Batch mode (with --output_dir):
    python scripts/infer_lora.py --test_file data/test.jsonl --output_dir output/results
    Runs inference on a test file, computes metrics, and saves results.
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from tqdm import tqdm

# Import shared utilities and constants
from utils import (
    BASE_MODEL,
    ADAPTER_DIR,
    get_device,
    load_data,
    format_prompt,
    generate,
    extract_answer,
    compute_metrics,
)


def run_demo_mode(adapter_dir: str, compare_base: bool = True):
    """Run demo mode with hardcoded example."""
    device = get_device()
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Setup quantization for CUDA
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

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if device_map is None:
        base_model = base_model.to(device)

    # Load LoRA adapter and merge
    print(f"Loading LoRA adapter from {adapter_dir}...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable=False,
    )
    merged_model = peft_model.merge_and_unload()

    # Hardcoded test prompt
    test_prompt = (
        "User's last 15 watched movies:\n"
        "1. The Matrix (1999) (rating ≈ 5.0)\n"
        "2. Terminator 2: Judgment Day (1991) (rating ≈ 4.5)\n"
        "3. Blade Runner (1982) (rating ≈ 4.0)\n"
        "...\n\n"
        "Candidate movie:\n"
        "Aliens (1986)\n\n"
        "Should we recommend this movie to the user? Answer Yes or No."
    )

    if compare_base:
        print("\n" + "="*60)
        print("BASE MODEL")
        print("="*60)
        base_response = generate(test_prompt, base_model, tokenizer, device)
        print(base_response)

    print("\n" + "="*60)
    print("LORA-FINETUNED MODEL")
    print("="*60)
    lora_response = generate(test_prompt, merged_model, tokenizer, device)
    print(lora_response)
    print()


def run_batch_mode(args):
    """Run batch mode with test file processing."""
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

    # Setup quantization for CUDA
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

    for example in tqdm(test_data, desc="Generating predictions", leave=False, mininterval=1.0):
        prompt = format_prompt(example)
        # Align inference system prompt with training default when missing.
        system_prompt = example.get("system", None)
        response = generate(
            prompt,
            model,
            tokenizer,
            device,
            system_prompt=system_prompt,
            do_sample=False,
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
        )
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


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with LoRA adapter - demo or batch mode"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Path to test data file (JSON or JSONL format). If not provided, runs demo mode.",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=ADAPTER_DIR,
        help=f"Path to LoRA adapter directory (default: {ADAPTER_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save inference results (required for batch mode)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)",
    )
    parser.add_argument(
        "--compare_base",
        action="store_true",
        help="In demo mode, also show base model output for comparison",
    )

    args = parser.parse_args()

    # Determine mode based on arguments
    if args.test_file is None and args.output_dir is None:
        # Demo mode
        print("Running in DEMO MODE (no test file specified)")
        print(f"Using adapter: {args.adapter_dir}\n")
        run_demo_mode(args.adapter_dir, args.compare_base)
    elif args.test_file is not None and args.output_dir is not None:
        # Batch mode
        print("Running in BATCH MODE")
        run_batch_mode(args)
    else:
        parser.error("For batch mode, both --test_file and --output_dir are required")


if __name__ == "__main__":
    main()
