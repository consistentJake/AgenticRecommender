#!/usr/bin/env python
"""Compare base model vs LoRA fine-tuned model on test dataset.

Usage: python compare_base_vs_lora.py --config configs/qwen3_7b_movielens_qlora.yaml

The script reads all settings from the YAML config file and automatically:
1. Runs inference on both base model and LoRA model
2. Compares their performance and saves detailed results
3. Displays analysis of improvements and regressions
"""

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

# Import shared utilities and constants
from utils import (
    BASE_MODEL,
    get_device,
    load_yaml_config,
    load_data,
    format_prompt,
    generate,
    generate_batch,
    extract_answer,
    compute_metrics,
)


# All utility functions (load_yaml_config, get_device, load_data, format_prompt,
# generate, extract_answer, compute_metrics) imported from utils


def display_example(example: Dict, max_history: int = 5):
    """Display a single example in a readable format."""
    print(f"\n{'='*80}")

    # Display format-specific fields
    if 'user_id' in example:
        print(f"User ID: {example['user_id']}")
    if 'candidate_title' in example:
        print(f"Candidate Movie: {example['candidate_title']}")

    print(f"Ground Truth: {example['label']}")
    print(f"Base Model Prediction: {example['base_prediction']} {'✓' if example['base_correct'] else '✗'}")
    print(f"LoRA Model Prediction: {example['lora_prediction']} {'✓' if example['lora_correct'] else '✗'}")

    # Display history for test data format
    if 'history_titles' in example:
        print(f"\nUser's Recent History (first {max_history}):")
        for i, movie in enumerate(example['history_titles'][:max_history]):
            print(f"  {movie}")
        if len(example['history_titles']) > max_history:
            print(f"  ... and {len(example['history_titles']) - max_history} more")

    # Display input for training data format
    if 'input' in example and 'history_titles' not in example:
        print(f"\nInput prompt (truncated):")
        input_text = example['input']
        if len(input_text) > 300:
            print(f"  {input_text[:300]}...")
        else:
            print(f"  {input_text}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare base model vs LoRA model using YAML config"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qwen3_7b_movielens_qlora.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = load_yaml_config(args.config)

    # Extract inference config
    inference_cfg = cfg.get("inference", {})

    # Build paths from config
    base_model_name = cfg.get("model_name_or_path", "Qwen/Qwen3-0.6B")
    adapter_dir = Path(cfg.get("output_dir", "output/qwen3-movielens-qlora"))
    test_file = Path(inference_cfg.get("test_file", "data/movielens_qwen3/test_raw.jsonl"))

    # Use separate infer_results directory instead of subdirectory under output_dir
    # This prevents mixing inference outputs with training artifacts
    infer_output_dir = Path(inference_cfg.get("infer_output_dir", "infer_results"))
    # Create a subdirectory based on the adapter name for organization
    adapter_name = adapter_dir.name
    output_dir = infer_output_dir / adapter_name

    batch_size = inference_cfg.get("batch_size", 1)
    max_samples = inference_cfg.get("max_samples", None)
    show_examples = inference_cfg.get("show_examples", 5)

    device = get_device()

    # Print configuration
    print("=" * 80)
    print("INFERENCE CONFIGURATION")
    print("=" * 80)
    print(f"Config file: {args.config}")
    print(f"Device: {device}")
    print(f"Base model: {base_model_name}")
    print(f"Adapter directory: {adapter_dir}")
    print(f"Test file: {test_file}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Max samples: {max_samples or 'all'}")
    print(f"Show examples: {show_examples}")
    print("=" * 80)
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
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
        base_model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if device_map is None:
        base_model = base_model.to(device)

    # Load test data
    print(f"\nLoading test data from {test_file}...")
    test_data = load_data(str(test_file))

    if max_samples:
        test_data = test_data[:max_samples]
        print(f"Limiting to {max_samples} samples for testing")

    print(f"Total test samples: {len(test_data)}\n")

    # Run inference on BASE MODEL FIRST (before loading LoRA to avoid contamination)
    print("="*60)
    print("Running inference on BASE MODEL (without LoRA)...")
    print(f"Processing {len(test_data)} samples in batches of {batch_size}...")
    print("="*60)
    base_predictions = []
    base_results = []

    # Open file for streaming output during inference
    base_output = output_dir / "base_predictions.jsonl"
    base_file = open(base_output, 'w', encoding='utf-8')

    # Process in batches for efficiency
    for i in tqdm(range(0, len(test_data), batch_size), desc="Base model inference", unit="batch"):
        batch = test_data[i:i+batch_size]

        # Prepare batch prompts
        prompts = [format_prompt(example) for example in batch]
        system_prompts = [example.get("system", None) for example in batch]

        # Use batch generation if batch_size > 1, otherwise use single generation
        if batch_size > 1:
            # All system prompts should be the same, use the first one
            system_prompt = system_prompts[0] if system_prompts else None
            responses = generate_batch(prompts, base_model, tokenizer, device, system_prompt)
        else:
            system_prompt = system_prompts[0] if system_prompts else None
            responses = [generate(prompts[0], base_model, tokenizer, device, system_prompt)]

        # Process each response in the batch
        for example, prompt, response in zip(batch, prompts, responses):
            prediction = extract_answer(response)

            # Extract label from either format
            label = example.get("output") or example.get("label")

            base_predictions.append(prediction)

            # Extract assistant's response only (after "assistant" marker)
            assistant_response = response.split("assistant")[-1].strip() if "assistant" in response else response

            # Build result dict with available fields for better debugging
            result = {
                "label": label,
                "prediction": prediction,
                "prompt": prompt,  # The user prompt sent to the model
                "assistant_response": assistant_response,  # What the model actually generated
                "full_response": response,  # Complete output including system/user/assistant
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

            base_results.append(result)

            # Write to file immediately for progress tracking
            base_file.write(json.dumps(result) + '\n')
            base_file.flush()  # Ensure it's written to disk

    # Close base predictions file
    base_file.close()
    print(f"Base predictions saved to: {base_output}")

    # NOW load LoRA adapter AFTER base inference is complete
    print("\n" + "="*60)
    print(f"Loading LoRA adapter from {adapter_dir}...")
    print("="*60)
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        is_trainable=False,
    )
    lora_model = peft_model.merge_and_unload()

    # Run inference on LORA MODEL
    print("\n" + "="*60)
    print("Running inference on LORA-FINETUNED MODEL...")
    print(f"Processing {len(test_data)} samples in batches of {batch_size}...")
    print("="*60)
    lora_predictions = []
    lora_results = []
    labels = []

    # Open file for streaming output during inference
    lora_output = output_dir / "lora_predictions.jsonl"
    lora_file = open(lora_output, 'w', encoding='utf-8')

    # Process in batches for efficiency
    for i in tqdm(range(0, len(test_data), batch_size), desc="LoRA model inference", unit="batch"):
        batch = test_data[i:i+batch_size]

        # Prepare batch prompts
        prompts = [format_prompt(example) for example in batch]
        system_prompts = [example.get("system", None) for example in batch]

        # Use batch generation if batch_size > 1, otherwise use single generation
        if batch_size > 1:
            # All system prompts should be the same, use the first one
            system_prompt = system_prompts[0] if system_prompts else None
            responses = generate_batch(prompts, lora_model, tokenizer, device, system_prompt)
        else:
            system_prompt = system_prompts[0] if system_prompts else None
            responses = [generate(prompts[0], lora_model, tokenizer, device, system_prompt)]

        # Process each response in the batch
        for example, prompt, response in zip(batch, prompts, responses):
            prediction = extract_answer(response)

            # Extract label from either format
            label = example.get("output") or example.get("label")

            lora_predictions.append(prediction)
            labels.append(label)

            # Extract assistant's response only (after "assistant" marker)
            assistant_response = response.split("assistant")[-1].strip() if "assistant" in response else response

            # Build result dict with available fields for better debugging
            result = {
                "label": label,
                "prediction": prediction,
                "prompt": prompt,  # The user prompt sent to the model
                "assistant_response": assistant_response,  # What the model actually generated
                "full_response": response,  # Complete output including system/user/assistant
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

            lora_results.append(result)

            # Write to file immediately for progress tracking
            lora_file.write(json.dumps(result) + '\n')
            lora_file.flush()  # Ensure it's written to disk

    # Close LoRA predictions file
    lora_file.close()
    print(f"LoRA predictions saved to: {lora_output}")

    # Compute metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    base_metrics = compute_metrics(base_predictions, labels)
    lora_metrics = compute_metrics(lora_predictions, labels)

    print("\nBASE MODEL (without LoRA):")
    print("-" * 40)
    print(f"  Accuracy:    {base_metrics['accuracy']:.4f} ({base_metrics['correct']}/{base_metrics['total']})")
    print(f"  Precision:   {base_metrics['precision']:.4f}")
    print(f"  Recall:      {base_metrics['recall']:.4f}")
    print(f"  F1 Score:    {base_metrics['f1']:.4f}")
    print(f"  TP/FP/FN/TN: {base_metrics['tp']}/{base_metrics['fp']}/{base_metrics['fn']}/{base_metrics['tn']}")

    print("\nLoRA-FINETUNED MODEL:")
    print("-" * 40)
    print(f"  Accuracy:    {lora_metrics['accuracy']:.4f} ({lora_metrics['correct']}/{lora_metrics['total']})")
    print(f"  Precision:   {lora_metrics['precision']:.4f}")
    print(f"  Recall:      {lora_metrics['recall']:.4f}")
    print(f"  F1 Score:    {lora_metrics['f1']:.4f}")
    print(f"  TP/FP/FN/TN: {lora_metrics['tp']}/{lora_metrics['fp']}/{lora_metrics['fn']}/{lora_metrics['tn']}")

    print("\nIMPROVEMENT (LoRA vs Base):")
    print("-" * 40)
    print(f"  Accuracy:    {lora_metrics['accuracy'] - base_metrics['accuracy']:+.4f}")
    print(f"  Precision:   {lora_metrics['precision'] - base_metrics['precision']:+.4f}")
    print(f"  Recall:      {lora_metrics['recall'] - base_metrics['recall']:+.4f}")
    print(f"  F1 Score:    {lora_metrics['f1'] - base_metrics['f1']:+.4f}")
    print("="*60)

    # Analyze disagreements
    disagreements = []
    base_correct_lora_wrong = []
    lora_correct_base_wrong = []
    both_wrong = []
    both_correct_but_different = []  # Edge case where both are "correct" but different

    for i, (bp, lp, label) in enumerate(zip(base_predictions, lora_predictions, labels)):
        if bp != lp:
            disagreement_info = {
                "index": i,
                "label": label,
                "base_prediction": bp,
                "lora_prediction": lp,
                "base_correct": bp == label,
                "lora_correct": lp == label,
            }

            # Add format-specific fields if available
            if "user_id" in test_data[i]:
                disagreement_info["user_id"] = test_data[i]["user_id"]
            if "candidate_title" in test_data[i]:
                disagreement_info["candidate_title"] = test_data[i]["candidate_title"]

            disagreements.append(disagreement_info)

            # Categorize disagreements
            if bp == label and lp != label:
                base_correct_lora_wrong.append(disagreement_info)
            elif lp == label and bp != label:
                lora_correct_base_wrong.append(disagreement_info)
            elif bp != label and lp != label:
                both_wrong.append(disagreement_info)

    print(f"\nDisagreements: {len(disagreements)} samples ({len(disagreements)/len(labels)*100:.2f}%)")
    print(f"  - LoRA improved (was wrong, now correct): {len(lora_correct_base_wrong)}")
    print(f"  - LoRA regressed (was correct, now wrong): {len(base_correct_lora_wrong)}")
    print(f"  - Both wrong but different: {len(both_wrong)}")

    # File paths for additional analysis files
    # Note: base_output and lora_output were already written during inference
    comparison_file = output_dir / "comparison_report.json"
    disagreements_file = output_dir / "disagreements_analysis.jsonl"
    improvements_file = output_dir / "lora_improvements.jsonl"
    regressions_file = output_dir / "lora_regressions.jsonl"

    # Save disagreements analysis
    print(f"Saving disagreements analysis to {disagreements_file}...")
    with open(disagreements_file, 'w', encoding='utf-8') as f:
        for d in disagreements:
            f.write(json.dumps(d) + '\n')

    # Save improvements (cases where LoRA fixed base model errors)
    if lora_correct_base_wrong:
        print(f"Saving LoRA improvements to {improvements_file}...")
        with open(improvements_file, 'w', encoding='utf-8') as f:
            for d in lora_correct_base_wrong:
                # Add full context
                example = test_data[d["index"]]
                if "history_titles" in example:
                    d["history_titles"] = example["history_titles"]
                if "input" in example:
                    d["input"] = example["input"]
                # Include both assistant responses and full responses for debugging
                d["base_assistant_response"] = base_results[d["index"]]["assistant_response"]
                d["lora_assistant_response"] = lora_results[d["index"]]["assistant_response"]
                d["base_full_response"] = base_results[d["index"]]["full_response"]
                d["lora_full_response"] = lora_results[d["index"]]["full_response"]
                f.write(json.dumps(d, indent=2) + '\n')

    # Save regressions (cases where LoRA made base model worse)
    if base_correct_lora_wrong:
        print(f"Saving LoRA regressions to {regressions_file}...")
        with open(regressions_file, 'w', encoding='utf-8') as f:
            for d in base_correct_lora_wrong:
                # Add full context
                example = test_data[d["index"]]
                if "history_titles" in example:
                    d["history_titles"] = example["history_titles"]
                if "input" in example:
                    d["input"] = example["input"]
                # Include both assistant responses and full responses for debugging
                d["base_assistant_response"] = base_results[d["index"]]["assistant_response"]
                d["lora_assistant_response"] = lora_results[d["index"]]["assistant_response"]
                d["base_full_response"] = base_results[d["index"]]["full_response"]
                d["lora_full_response"] = lora_results[d["index"]]["full_response"]
                f.write(json.dumps(d, indent=2) + '\n')

    # Save comparison report
    print(f"Saving comparison report to {comparison_file}...")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_file": str(test_file),
            "total_samples": len(labels),
            "base_model": {
                "name": base_model_name,
                "metrics": base_metrics,
            },
            "lora_model": {
                "adapter_dir": str(adapter_dir),
                "metrics": lora_metrics,
            },
            "improvement": {
                "accuracy": lora_metrics['accuracy'] - base_metrics['accuracy'],
                "precision": lora_metrics['precision'] - base_metrics['precision'],
                "recall": lora_metrics['recall'] - base_metrics['recall'],
                "f1": lora_metrics['f1'] - base_metrics['f1'],
            },
            "disagreements_summary": {
                "total": len(disagreements),
                "percentage": len(disagreements) / len(labels) * 100,
                "lora_improvements": len(lora_correct_base_wrong),
                "lora_regressions": len(base_correct_lora_wrong),
                "both_wrong": len(both_wrong),
            },
        }, f, indent=2)

    print(f"\nAll results saved successfully!")
    print(f"\nFiles saved in {output_dir}/:")
    print(f"  - base_predictions.jsonl: All base model predictions (written during inference)")
    print(f"  - lora_predictions.jsonl: All LoRA model predictions (written during inference)")
    print(f"  - comparison_report.json: Overall metrics and summary")
    print(f"  - disagreements_analysis.jsonl: All cases where models disagree")
    if lora_correct_base_wrong:
        print(f"  - lora_improvements.jsonl: Cases where LoRA fixed base model errors ({len(lora_correct_base_wrong)} samples)")
    if base_correct_lora_wrong:
        print(f"  - lora_regressions.jsonl: Cases where LoRA made mistakes base didn't ({len(base_correct_lora_wrong)} samples)")
    print(f"\nEach prediction file includes:")
    print(f"  - label: Ground truth")
    print(f"  - prediction: Extracted Yes/No/Unknown")
    print(f"  - prompt: User prompt sent to model")
    print(f"  - assistant_response: Only the model's generated text (for debugging)")
    print(f"  - full_response: Complete output including system/user/assistant")

    # ========================================================================
    # ANALYSIS SECTION
    # ========================================================================
    print("\n\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)

    # Display improvements
    if lora_correct_base_wrong:
        print(f"\n{'='*80}")
        print(f"LoRA IMPROVEMENTS ({len(lora_correct_base_wrong)} total)")
        print(f"Cases where base model was wrong, but LoRA got it right")
        print(f"{'='*80}")

        for i, example in enumerate(lora_correct_base_wrong[:show_examples]):
            display_example(example)

        if len(lora_correct_base_wrong) > show_examples:
            print(f"\n... and {len(lora_correct_base_wrong) - show_examples} more improvements")

    # Display regressions
    if base_correct_lora_wrong:
        print(f"\n\n{'='*80}")
        print(f"LoRA REGRESSIONS ({len(base_correct_lora_wrong)} total)")
        print(f"Cases where base model was correct, but LoRA got it wrong")
        print(f"{'='*80}")

        for i, example in enumerate(base_correct_lora_wrong[:show_examples]):
            display_example(example)

        if len(base_correct_lora_wrong) > show_examples:
            print(f"\n... and {len(base_correct_lora_wrong) - show_examples} more regressions")

    # Display some cases where both were wrong
    if both_wrong:
        print(f"\n\n{'='*80}")
        print(f"BOTH WRONG BUT DIFFERENT ({len(both_wrong)} total)")
        print(f"Cases where both models were wrong but gave different answers")
        print(f"{'='*80}")

        for i, example in enumerate(both_wrong[:show_examples]):
            display_example(example)

        if len(both_wrong) > show_examples:
            print(f"\n... and {len(both_wrong) - show_examples} more cases")

    # Final conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    improvement_f1 = lora_metrics['f1'] - base_metrics['f1']
    net_improvement = len(lora_correct_base_wrong) - len(base_correct_lora_wrong)

    if improvement_f1 > 0:
        print(f"✓ LoRA fine-tuning IMPROVED the model")
        print(f"  - F1 score increased by {improvement_f1:.4f}")
        print(f"  - Net {net_improvement} more correct predictions")
    elif improvement_f1 < 0:
        print(f"✗ LoRA fine-tuning HURT the model")
        print(f"  - F1 score decreased by {abs(improvement_f1):.4f}")
        print(f"  - Net {abs(net_improvement)} fewer correct predictions")
    else:
        print(f"→ LoRA fine-tuning had NO IMPACT on F1 score")

    print("\n" + "="*80)
    print(f"\nAll results and analysis saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
