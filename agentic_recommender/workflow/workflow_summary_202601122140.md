# Workflow Runner Summary

**Generated:** 2026-01-12 21:40

## Overview

A configurable workflow system for processing food delivery data and running LLM-based recommendations.

## Created Files

| File | Purpose |
|------|---------|
| `workflow_config.yaml` | YAML configuration for all stages, inputs/outputs |
| `workflow_runner.py` | Main workflow runner with stage control |

## Generated Outputs

```
outputs/
├── stage1_merged_data.parquet     # 151 MB - All merged data
├── stage1_merged_preview.json     # Top 1000 rows as JSON (for easy viewing)
├── stage1_stats.json              # Dataset statistics
├── stage2_enriched_users.json     # User representations
├── stage2_users_summary.json      # User processing summary
├── stage3_cuisine_profiles.json   # 78 cuisine profiles
├── stage4_prompts.json            # Formatted prompts
├── stage4_prompts_readable.txt    # Human-readable prompts
├── stage5_predictions.json        # Mock LLM predictions (first 50)
├── stage5_predictions_summary.json
├── stage6_topk_results.json       # TopK evaluation results
├── stage6_topk_samples.json       # Test samples used
└── workflow.log                   # Full execution log
```

## Pipeline Stages

| Stage | Description | Input | Output |
|-------|-------------|-------|--------|
| `load_data` | Load Singapore food delivery data | Raw CSV files | Parquet + JSON preview |
| `build_users` | Create EnrichedUser representations | Merged data | User JSON |
| `build_cuisines` | Build cuisine temporal profiles | Merged data | Cuisine profiles |
| `generate_prompts` | Format prompts for prediction | Users + data | Prompts JSON |
| `run_predictions` | Run mock predictions (testing) | Prompts | Predictions |
| `run_topk_evaluation` | **Real LLM evaluation with TopK metrics** | Merged data | TopK metrics |

## How to Use

### List all stages
```bash
python workflow_runner.py --list
```

### Run ALL enabled stages
```bash
python workflow_runner.py
```

### Run specific stage(s)
```bash
python workflow_runner.py --stages load_data
python workflow_runner.py --stages build_users generate_prompts
python workflow_runner.py --stages run_topk_evaluation
```

### Use custom config
```bash
python workflow_runner.py --config my_custom_config.yaml
```

## Key Configuration (workflow_config.yaml)

### Process ALL users (for auxiliary matrix)
```yaml
build_users:
  settings:
    max_users: null  # null = ALL users (~476K with >=3 orders)
```

### Control predictions
```yaml
run_predictions:
  settings:
    limit: 50  # First 50 prompts get mock predictions
```

### TopK Evaluation with Real LLM
```yaml
run_topk_evaluation:
  enabled: true
  settings:
    n_samples: 50        # Number of test samples
    min_history: 5       # Min orders per user
    k_values: [1, 3, 5, 10]

llm:
  provider: "openrouter"  # Use real LLM
  openrouter:
    model_name: "google/gemini-2.0-flash-001"
    api_key: "your-api-key"  # Or set OPENROUTER_API_KEY env var
```

## Dataset Statistics

```json
{
  "total_rows": 3431870,
  "unique_customers": 476150,
  "unique_orders": 1709414,
  "unique_vendors": 7203,
  "unique_cuisines": 78,
  "unique_products": 242552,
  "avg_items_per_order": 2.01
}
```

## TopK Evaluation Metrics

The `run_topk_evaluation` stage computes:
- **Hit@K**: % of times ground truth is in top-K predictions
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain

This uses the `SequentialRecommendationEvaluator` from `agentic_recommender/evaluation/topk.py`.

## Example: Run Full Pipeline with Real LLM

1. Edit `workflow_config.yaml`:
   ```yaml
   llm:
     provider: "openrouter"
   ```

2. Set API key:
   ```bash
   export OPENROUTER_API_KEY="your-key"
   ```

3. Run TopK evaluation:
   ```bash
   python workflow_runner.py --stages run_topk_evaluation
   ```

## Notes

- Stage 1 outputs both Parquet (full data) and JSON (preview of 1000 rows)
- Processing all 476K users takes significant time and disk space
- Mock provider is used by default for testing; switch to `openrouter` for real predictions
- The TopK evaluation stage builds test samples from the data and measures actual LLM performance
