#!/usr/bin/env python3
"""Compare metrics across multiple stage 9 runs.

Usage:
    python scripts/compare_runs.py outputs/data_sg/202601312244 outputs/data_sg/202602010035
    python scripts/compare_runs.py outputs/data_se/2026*  # glob works too
"""
import json
import sys
from pathlib import Path


def extract_metrics(run_dir: Path):
    config_path = run_dir / "runtime_config.yaml"
    results_path = run_dir / "stage9_repeat_results.json"

    # Extract model name from YAML (simple parse)
    model_name = "?"
    enable_thinking = "?"
    enable_thinking_r2 = None
    if config_path.exists():
        for line in config_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("model_name:"):
                model_name = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            if stripped.startswith("enable_thinking:"):
                enable_thinking = stripped.split(":", 1)[1].strip()
            if stripped.startswith("enable_thinking_round2:"):
                enable_thinking_r2 = stripped.split(":", 1)[1].strip()

    thinking_label = ""
    if enable_thinking_r2 == "false":
        thinking_label = " (no R2 thinking)"
    elif enable_thinking == "false":
        thinking_label = " (no thinking)"

    # Extract metrics from results JSON
    if not results_path.exists():
        return None

    with open(results_path) as f:
        r = json.load(f)

    return {
        "dir": run_dir.name,
        "model": model_name + thinking_label,
        "r1_hit3": r.get("round1_hit@3", 0),
        "r1_ndcg3": r.get("round1_ndcg@3", 0),
        "hit3": r.get("hit@3", 0),
        "ndcg3": r.get("ndcg@3", 0),
        "hit1": r.get("hit@1", 0),
        "hit5": r.get("hit@5", 0),
        "mrr": r.get("mrr", 0),
        "gt_found": r.get("ground_truth_found_rate", r.get("gt_found_rate", r.get("gt_in_candidates_rate", 0))),
        "samples": r.get("valid_samples", r.get("total_samples", 0)),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_runs.py <run_dir1> <run_dir2> ...")
        sys.exit(1)

    runs = []
    for arg in sys.argv[1:]:
        run_dir = Path(arg)
        if not run_dir.is_dir():
            print(f"Skipping {arg}: not a directory")
            continue
        m = extract_metrics(run_dir)
        if m:
            runs.append(m)
        else:
            print(f"Skipping {arg}: no stage9_repeat_results.json")

    if not runs:
        print("No valid runs found.")
        sys.exit(1)

    # Print table
    print(f"{'Dir':<16} {'Model':<45} {'R1 Hit@3':>9} {'R1 NDCG@3':>10} {'Hit@3':>7} {'NDCG@3':>8} {'Hit@1':>7} {'Hit@5':>7} {'MRR':>7} {'GT%':>6} {'N':>5}")
    print("-" * 145)
    for m in runs:
        print(
            f"{m['dir']:<16} {m['model']:<45} {m['r1_hit3']:>9.4f} {m['r1_ndcg3']:>10.4f} "
            f"{m['hit3']:>7.4f} {m['ndcg3']:>8.4f} {m['hit1']:>7.4f} {m['hit5']:>7.4f} "
            f"{m['mrr']:>7.4f} {m['gt_found']*100 if m['gt_found'] <= 1 else m['gt_found']:>5.1f}% {m['samples']:>5}"
        )


if __name__ == "__main__":
    main()
