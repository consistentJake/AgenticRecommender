#!/usr/bin/env python
"""Analyze and display key differences between base and LoRA models."""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


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
    parser = argparse.ArgumentParser(description="Analyze differences between base and LoRA models")
    parser.add_argument(
        "--infer_dir",
        type=str,
        default="infer",
        help="Directory containing inference results",
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=5,
        help="Number of examples to show for each category",
    )
    args = parser.parse_args()

    infer_dir = Path(args.infer_dir)

    # Load comparison report
    report_file = infer_dir / "comparison_report.json"
    if not report_file.exists():
        print(f"Error: {report_file} not found. Run compare_base_vs_lora.py first.")
        return

    with open(report_file, 'r') as f:
        report = json.load(f)

    # Display summary
    print("="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\nTotal samples: {report['total_samples']}")

    print("\nBASE MODEL:")
    base = report['base_model']['metrics']
    print(f"  Accuracy:  {base['accuracy']:.4f}")
    print(f"  Precision: {base['precision']:.4f}")
    print(f"  Recall:    {base['recall']:.4f}")
    print(f"  F1 Score:  {base['f1']:.4f}")

    print("\nLoRA MODEL:")
    lora = report['lora_model']['metrics']
    print(f"  Accuracy:  {lora['accuracy']:.4f}")
    print(f"  Precision: {lora['precision']:.4f}")
    print(f"  Recall:    {lora['recall']:.4f}")
    print(f"  F1 Score:  {lora['f1']:.4f}")

    print("\nIMPROVEMENT:")
    imp = report['improvement']
    print(f"  Accuracy:  {imp['accuracy']:+.4f}")
    print(f"  Precision: {imp['precision']:+.4f}")
    print(f"  Recall:    {imp['recall']:+.4f}")
    print(f"  F1 Score:  {imp['f1']:+.4f}")

    print("\nDISAGREEMENTS:")
    dis = report['disagreements_summary']
    print(f"  Total disagreements:   {dis['total']} ({dis['percentage']:.2f}%)")
    print(f"  LoRA improvements:     {dis['lora_improvements']} (base wrong → LoRA correct)")
    print(f"  LoRA regressions:      {dis['lora_regressions']} (base correct → LoRA wrong)")
    print(f"  Both wrong (diff):     {dis['both_wrong']} (both wrong, different answers)")

    # Net improvement
    net_improvement = dis['lora_improvements'] - dis['lora_regressions']
    print(f"\n  Net improvement:       {net_improvement:+d} samples")

    # Load and display improvements
    improvements_file = infer_dir / "lora_improvements.jsonl"
    if improvements_file.exists():
        improvements = load_jsonl(improvements_file)
        print(f"\n{'='*80}")
        print(f"LoRA IMPROVEMENTS ({len(improvements)} total)")
        print(f"Cases where base model was wrong, but LoRA got it right")
        print(f"{'='*80}")

        for i, example in enumerate(improvements[:args.show_examples]):
            display_example(example)

        if len(improvements) > args.show_examples:
            print(f"\n... and {len(improvements) - args.show_examples} more improvements")

    # Load and display regressions
    regressions_file = infer_dir / "lora_regressions.jsonl"
    if regressions_file.exists():
        regressions = load_jsonl(regressions_file)
        print(f"\n\n{'='*80}")
        print(f"LoRA REGRESSIONS ({len(regressions)} total)")
        print(f"Cases where base model was correct, but LoRA got it wrong")
        print(f"{'='*80}")

        for i, example in enumerate(regressions[:args.show_examples]):
            display_example(example)

        if len(regressions) > args.show_examples:
            print(f"\n... and {len(regressions) - args.show_examples} more regressions")

    # Load and display some cases where both were wrong
    disagreements_file = infer_dir / "disagreements_analysis.jsonl"
    if disagreements_file.exists():
        all_disagreements = load_jsonl(disagreements_file)
        both_wrong = [d for d in all_disagreements if not d['base_correct'] and not d['lora_correct']]

        if both_wrong:
            print(f"\n\n{'='*80}")
            print(f"BOTH WRONG BUT DIFFERENT ({len(both_wrong)} total)")
            print(f"Cases where both models were wrong but gave different answers")
            print(f"{'='*80}")

            for i, example in enumerate(both_wrong[:args.show_examples]):
                display_example(example)

            if len(both_wrong) > args.show_examples:
                print(f"\n... and {len(both_wrong) - args.show_examples} more cases")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if imp['f1'] > 0:
        print(f"✓ LoRA fine-tuning IMPROVED the model")
        print(f"  - F1 score increased by {imp['f1']:.4f}")
        print(f"  - Net {net_improvement} more correct predictions")
    elif imp['f1'] < 0:
        print(f"✗ LoRA fine-tuning HURT the model")
        print(f"  - F1 score decreased by {abs(imp['f1']):.4f}")
        print(f"  - Net {abs(net_improvement)} fewer correct predictions")
    else:
        print(f"→ LoRA fine-tuning had NO IMPACT on F1 score")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
