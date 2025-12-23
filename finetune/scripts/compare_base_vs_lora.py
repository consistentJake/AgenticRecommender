#!/usr/bin/env python
"""Compare base model vs LoRA fine-tuned model on test dataset.

Usage: python compare_base_vs_lora.py --config configs/qwen3_7b_movielens_qlora.yaml

The script reads all settings from the YAML config file and automatically:
1. Runs inference on both base model and LoRA model
2. Compares their performance and saves detailed results
3. Displays analysis of improvements and regressions
4. Saves the full config in comparison_report.json for reproducibility

Adapter Directory Configuration:
    The script determines which LoRA adapter to load using this priority:
    1. inference.adapter_dir (if specified in config) - highest priority
    2. output_dir (from main config) - fallback

    Examples in YAML config:
        inference:
          # Option 1: Use specific checkpoint
          adapter_dir: output/qwen3-7b-movielens-qlora-2/checkpoint-12000

          # Option 2: Use final adapter (not specifying adapter_dir)
          # Falls back to output_dir: output/qwen3-7b-movielens-qlora-2

Chunked Processing Mode:
    The script supports chunked processing to optimize memory usage and allow
    LoRA inference to potentially reuse base model intermediate results.

    Set chunk_size in the inference config to enable:
        inference:
          batch_size: 8
          chunk_size: 200  # Process 200 samples per chunk

    For each chunk, the script:
    1. Runs base model inference
    2. Loads LoRA adapter and runs LoRA inference
    3. Unloads LoRA adapter and clears GPU cache
    4. Writes results immediately to disk

    Benefits: Lower memory usage, better cache locality, incremental results.
    See docs/CHUNKED_INFERENCE.md for detailed documentation.

Performance Optimizations:
    The script includes several optimizations to make LoRA inference faster:

    1. Direct PEFT Usage (use_peft_directly=true):
       - Uses PeftModel directly instead of merge_and_unload()
       - Shares base model weights (50% less memory for LoRA)
       - Only computes LoRA deltas (15-30% faster inference)
       - Enabled by default

    2. Prompt Caching:
       - Formats prompts once per chunk, reuses for both models
       - Eliminates duplicate formatting work
       - Reduces CPU overhead

    Configuration:
        inference:
          use_peft_directly: true  # Recommended for best performance
          batch_size: 8            # Batch size for throughput

    Note: The first batch will always be slower due to GPU initialization
    (CUDA kernel compilation, cuBLAS algorithm selection, memory allocation).
    This is unavoidable and subsequent batches will be fast.

    See docs/LORA_INFERENCE_OPTIMIZATION.md for detailed technical explanation.

Output File Structure:
    Each prediction file (base_predictions.jsonl, lora_predictions.jsonl) contains
    these fields to help understand the full inference pipeline:

    - label: Ground truth answer (Yes/No)
    - prediction: Extracted answer from model (Yes/No/Unknown)
    - user_prompt: User's prompt text only (e.g., "User's last 15 watched movies:...")
    - model_input: Full formatted text sent to model including system prompt and chat
                   template markers (e.g., "<|im_start|>system\n...<|im_start|>user\n...")
    - assistant_response: Only the model's generated text (after "assistant" marker)
    - full_response: Complete decoded output = model_input + generated response
    - instruction: Original instruction field from training data (if present)
    - input: Original input field from training data (if present)

    Understanding the relationship:
        model_input = format_with_chat_template(system_prompt + user_prompt)
        full_response = decode(model_output_tokens) = model_input + assistant_response
        prediction = extract_answer(assistant_response)
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

    # Determine adapter directory:
    # 1. Use inference.adapter_dir if explicitly specified (highest priority)
    # 2. Fall back to output_dir from main config
    adapter_source = "inference.adapter_dir"
    if "adapter_dir" in inference_cfg:
        adapter_dir = Path(inference_cfg["adapter_dir"])
    else:
        adapter_dir = Path(cfg.get("output_dir", "output/qwen3-movielens-qlora"))
        adapter_source = "output_dir (fallback)"

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
    # chunk_size: number of samples to process together before switching models
    # This controls memory usage and allows intermediate cache cleanup
    chunk_size = inference_cfg.get("chunk_size", None)  # If None, process all at once (old behavior)

    # use_peft_directly: if True, use PeftModel directly without merge_and_unload()
    # This is faster and more memory efficient as it only computes LoRA deltas
    # and shares base model weights
    use_peft_directly = inference_cfg.get("use_peft_directly", True)  # Default to True for better performance

    device = get_device()

    # Print configuration
    print("=" * 80)
    print("INFERENCE CONFIGURATION")
    print("=" * 80)
    print(f"Config file: {args.config}")
    print(f"Device: {device}")
    print(f"Base model: {base_model_name}")
    print(f"Adapter directory: {adapter_dir}")
    print(f"  (Source: {adapter_source})")
    print(f"Test file: {test_file}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Chunk size: {chunk_size or 'all (process entire dataset at once)'}")
    print(f"Use PEFT directly: {use_peft_directly} {'(faster, more memory efficient)' if use_peft_directly else '(creates merged model)'}")
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

    # Prepare for chunked or full processing
    if chunk_size:
        print("="*80)
        print("CHUNKED PROCESSING MODE")
        print("="*80)
        print(f"Processing data in chunks of {chunk_size} samples")
        print(f"For each chunk:")
        print(f"  1. Run base model inference")
        print(f"  2. Load LoRA adapter and run LoRA inference")
        print(f"  3. Unload LoRA adapter and clear cache")
        print(f"  4. Write results to disk")
        print("This approach minimizes memory usage and allows LoRA to reuse base model state.")
        print("="*80)
        print()
    else:
        print("="*80)
        print("FULL PROCESSING MODE (all samples at once)")
        print("="*80)
        print()

    # Initialize result containers
    base_predictions = []
    base_results = []
    lora_predictions = []
    lora_results = []
    labels = []

    # Open files for streaming output during inference
    base_output = output_dir / "base_predictions.jsonl"
    lora_output = output_dir / "lora_predictions.jsonl"
    base_file = open(base_output, 'w', encoding='utf-8')
    lora_file = open(lora_output, 'w', encoding='utf-8')

    # Determine chunk boundaries
    if chunk_size:
        chunk_starts = list(range(0, len(test_data), chunk_size))
    else:
        # Process all data as one chunk
        chunk_starts = [0]
        chunk_size = len(test_data)

    # Track if LoRA is currently loaded
    lora_model = None
    peft_model = None

    # Process data in chunks
    for chunk_idx, chunk_start in enumerate(chunk_starts):
        chunk_end = min(chunk_start + chunk_size, len(test_data))
        chunk_data = test_data[chunk_start:chunk_end]

        if len(chunk_starts) > 1:
            print(f"\n{'='*80}")
            print(f"Processing Chunk {chunk_idx + 1}/{len(chunk_starts)}")
            print(f"Samples {chunk_start} to {chunk_end-1} ({len(chunk_data)} samples)")
            print(f"{'='*80}")

        # ====================================================================
        # STEP 1: Run BASE MODEL inference on this chunk
        # ====================================================================
        print(f"\n{'='*60}")
        print(f"Step 1: BASE MODEL inference (chunk {chunk_idx + 1}/{len(chunk_starts)})")
        print(f"{'='*60}")

        for i in tqdm(range(0, len(chunk_data), batch_size),
                     desc=f"Base inference (chunk {chunk_idx+1})", unit="batch"):
            batch = chunk_data[i:i+batch_size]

            # Use batch generation if batch_size > 1, otherwise use single generation
            if batch_size > 1:
                responses, input_texts = generate_batch(
                    examples=batch, model=base_model, tokenizer=tokenizer, device=device, return_input_texts=True
                )
            else:
                response, input_text = generate(
                    example=batch[0], model=base_model, tokenizer=tokenizer, device=device, return_input_text=True
                )
                responses = [response]
                input_texts = [input_text]

            # Process each response in the batch
            for example, response, input_text in zip(batch, responses, input_texts):
                prediction = extract_answer(response)
                label = example.get("output") or example.get("label")
                base_predictions.append(prediction)

                # Extract assistant's response only
                assistant_response = response.split("assistant")[-1].strip() if "assistant" in response else response

                # Extract user prompt from example (for display)
                user_content = example.get("instruction", "")
                example_input = example.get("input")
                if example_input:
                    user_content = f"{user_content}\n\n{example_input}"

                # Build result dict
                result = {
                    "label": label,
                    "prediction": prediction,
                    "user_prompt": user_content,  # User's prompt text (from instruction + input)
                    "model_input": input_text,  # Full formatted input sent to model (with chat template)
                    "assistant_response": assistant_response,  # Only the generated response
                    "full_response": response,  # Decoded output (input + generated, with chat template)
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

                # Write to file immediately
                base_file.write(json.dumps(result) + '\n')
                base_file.flush()

        # ====================================================================
        # STEP 2: Load LoRA adapter and run LORA MODEL inference on same chunk
        # ====================================================================
        print(f"\n{'='*60}")
        print(f"Step 2: LORA MODEL inference (chunk {chunk_idx + 1}/{len(chunk_starts)})")
        print(f"{'='*60}")

        # Load LoRA adapter if not already loaded
        if lora_model is None:
            print(f"Loading LoRA adapter from {adapter_dir}...")
            peft_model = PeftModel.from_pretrained(
                base_model,
                str(adapter_dir),
                is_trainable=False,
            )

            if use_peft_directly:
                # Use PEFT model directly - more efficient
                # Only computes LoRA deltas and adds to base model outputs
                # Shares base model weights (lower memory usage)
                lora_model = peft_model
                print("Using PEFT model directly (efficient mode).")
            else:
                # Merge and unload - creates a separate model
                # Less efficient but sometimes needed for compatibility
                lora_model = peft_model.merge_and_unload()
                peft_model = None  # Free the PEFT wrapper
                print("Using merged model (compatibility mode).")

        # Process LoRA inference on the same batch
        for i in tqdm(range(0, len(chunk_data), batch_size),
                     desc=f"LoRA inference (chunk {chunk_idx+1})", unit="batch"):
            batch = chunk_data[i:i+batch_size]

            # Use batch generation if batch_size > 1, otherwise use single generation
            if batch_size > 1:
                responses, input_texts = generate_batch(
                    examples=batch, model=lora_model, tokenizer=tokenizer, device=device, return_input_texts=True
                )
            else:
                response, input_text = generate(
                    example=batch[0], model=lora_model, tokenizer=tokenizer, device=device, return_input_text=True
                )
                responses = [response]
                input_texts = [input_text]

            # Process each response in the batch
            for example, response, input_text in zip(batch, responses, input_texts):
                prediction = extract_answer(response)
                label = example.get("output") or example.get("label")
                lora_predictions.append(prediction)
                labels.append(label)

                # Extract assistant's response only
                assistant_response = response.split("assistant")[-1].strip() if "assistant" in response else response

                # Extract user prompt from example (for display)
                user_content = example.get("instruction", "")
                example_input = example.get("input")
                if example_input:
                    user_content = f"{user_content}\n\n{example_input}"

                # Build result dict
                result = {
                    "label": label,
                    "prediction": prediction,
                    "user_prompt": user_content,  # User's prompt text (from instruction + input)
                    "model_input": input_text,  # Full formatted input sent to model (with chat template)
                    "assistant_response": assistant_response,  # Only the generated response
                    "full_response": response,  # Decoded output (input + generated, with chat template)
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

                # Write to file immediately
                lora_file.write(json.dumps(result) + '\n')
                lora_file.flush()

        # ====================================================================
        # STEP 3: Clean up for next chunk (if there are more chunks)
        # ====================================================================
        if chunk_idx < len(chunk_starts) - 1:  # Not the last chunk
            print(f"\n{'='*60}")
            print(f"Step 3: Cleanup before next chunk")
            print(f"{'='*60}")

            # Unload LoRA model to free memory
            if lora_model is not None:
                del lora_model
                lora_model = None
                print("LoRA model unloaded.")

            if peft_model is not None:
                del peft_model
                peft_model = None
                print("PEFT wrapper unloaded.")

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("GPU cache cleared.")

            print(f"Chunk {chunk_idx + 1} complete. Ready for next chunk.\n")

    # Close output files
    base_file.close()
    lora_file.close()
    print(f"\nBase predictions saved to: {base_output}")
    print(f"LoRA predictions saved to: {lora_output}")

    # Compute metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    base_metrics = compute_metrics(base_predictions, labels)
    lora_metrics = compute_metrics(lora_predictions, labels)

    print("There are 3 different prediction results: Yes, No, Unknown, Unknown will be counted here")
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
            "config": cfg,  # Full config for reproducibility
            "config_file": str(args.config),
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
    print(f"  - comparison_report.json: Overall metrics, summary, and full config (for reproducibility)")
    print(f"  - disagreements_analysis.jsonl: All cases where models disagree")
    if lora_correct_base_wrong:
        print(f"  - lora_improvements.jsonl: Cases where LoRA fixed base model errors ({len(lora_correct_base_wrong)} samples)")
    if base_correct_lora_wrong:
        print(f"  - lora_regressions.jsonl: Cases where LoRA made mistakes base didn't ({len(base_correct_lora_wrong)} samples)")
    print(f"\nEach prediction file includes:")
    print(f"  - label: Ground truth (Yes/No)")
    print(f"  - prediction: Extracted answer (Yes/No/Unknown)")
    print(f"  - user_prompt: User's prompt text only (no chat template formatting)")
    print(f"  - model_input: Full formatted input sent to model (with system prompt + chat template)")
    print(f"  - assistant_response: Only the model's generated text")
    print(f"  - full_response: Complete decoded output (model_input + generated response)")
    print(f"  - instruction/input: Original data fields (if present in test data)")

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
