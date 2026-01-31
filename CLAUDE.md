# AgenticRecommender

## Quick Start

**Entry point:** `agentic_recommender/workflow/workflow_runner.py`
**Config:** `agentic_recommender/workflow/workflow_config_qwen32_linux.yaml`

```bash
# From project root:
python -m agentic_recommender.workflow.workflow_runner --config workflow_config_qwen32_linux.yaml

# Run specific stages:
python -m agentic_recommender.workflow.workflow_runner --config workflow_config_qwen32_linux.yaml --stages load_data build_users

# List stages:
python -m agentic_recommender.workflow.workflow_runner --config workflow_config_qwen32_linux.yaml --list

# Background run (with nohup):
./agentic_recommender/workflow/run_workflow.sh -c workflow_config_qwen32_linux.yaml
```

**Note:** Config path is resolved relative to the script's directory — pass just the filename, not a full path.

## Pipeline Stages (in order)

| # | Stage | Enabled | Purpose |
|---|-------|---------|---------|
| 1 | `load_data` | yes | Load/merge Singapore food delivery data |
| 2 | `build_users` | yes | Build enriched user representations |
| 3 | `build_cuisines` | yes | Build cuisine profiles |
| 4 | `generate_prompts` | no | Legacy prompt generation |
| 5 | `run_predictions` | no | Legacy LLM predictions |
| 6 | `run_topk_evaluation` | no | Direct LLM ranking |
| 7 | `run_rerank_evaluation` | no | Simple retrieve-rerank |
| 8 | `run_enhanced_rerank_evaluation` | yes | **Main eval**: two-round LLM + LightGCN reflection |
| 9 | `run_repeat_evaluation` | yes | Repeated orders: two-round LLM (cuisine→vendor) |

## Key Files

| What | Where |
|------|-------|
| Workflow runner | `agentic_recommender/workflow/workflow_runner.py` |
| Config (Qwen32) | `agentic_recommender/workflow/workflow_config_qwen32_linux.yaml` |
| Rerank evaluator | `agentic_recommender/evaluation/rerank_eval.py` |
| LightGCN | `agentic_recommender/similarity/lightGCN.py` |
| Swing similarity | `agentic_recommender/similarity/methods.py` |
| LLM providers | `agentic_recommender/models/llm_provider.py` |
| Data loader | `agentic_recommender/data/enriched_loader.py` |
| Prompt templates | `agentic_recommender/core/templates/rerank/` |
| Stage cache | `agentic_recommender/workflow/stage_cache.py` |
| Repeat filter | `agentic_recommender/data/repeat_filter.py` |
| Geohash index | `agentic_recommender/data/geohash_index.py` |
| Repeat evaluator | `agentic_recommender/evaluation/repeat_evaluator.py` |

## Output Structure

Each run creates a timestamped subfolder under `outputs/`:
```
outputs/
├── stage1_merged_data.parquet          # Cached stage outputs (root level)
├── stage2_enriched_users.json
├── stage3_cuisine_profiles.json
├── stage8_enhanced_rerank_detailed.json
└── 202601262250/                       # Per-run timestamped folder
    ├── runtime_config.yaml             # Config snapshot for this run
    ├── workflow.log                    # Run log
    ├── stage1_merged_data.parquet     # Merged orders+vendors+products
    ├── stage1_merged_preview.json     # First N rows preview
    ├── stage1_stats.json              # Row/unique counts
    ├── stage1_test_data.parquet       # Test split (method2)
    ├── stage8_enhanced_rerank_samples.json    # Test samples used
    ├── stage8_enhanced_rerank_results.json    # Aggregate metrics (Hit@K, NDCG, MRR, basket)
    ├── stage8_enhanced_rerank_detailed.json   # Per-sample results
    ├── stage8_enhanced_rerank_detailed_preview.json
    ├── detailed_results.jsonl         # Streaming results (async)
    ├── stage9_repeat_results.json     # Stage 9: Hit@K, NDCG, MRR for repeat orders
    ├── stage9_repeat_samples.json     # Test samples used
    └── stage9_repeat_detailed.json    # Per-sample results
```

## Testing: Validate by Stage

After making changes, test stage-by-stage by checking outputs. Run individual stages to isolate issues.

**Stage 1 — load_data:** Check `stage1_stats.json` for expected counts (617K rows, 109K customers, 39 cuisines).
```bash
python -m agentic_recommender.workflow.workflow_runner --config workflow_config_qwen32_linux.yaml --stages load_data
```

**Stage 2 — build_users:** Check `stage2_users_summary.json` (18K+ users with min 5 orders).
```bash
python -m agentic_recommender.workflow.workflow_runner --config workflow_config_qwen32_linux.yaml --stages build_users
```

**Stage 3 — build_cuisines:** Check `stage3_cuisine_profiles.json` has 39 cuisine entries.
```bash
python -m agentic_recommender.workflow.workflow_runner --config workflow_config_qwen32_linux.yaml --stages build_cuisines
```

**Stage 8 — run_enhanced_rerank_evaluation:** Check `stage8_enhanced_rerank_results.json` for Hit@K, NDCG, MRR metrics.
```bash
python -m agentic_recommender.workflow.workflow_runner --config workflow_config_qwen32_linux.yaml --stages run_enhanced_rerank_evaluation
```

**Stage 9 — run_repeat_evaluation:** Check `stage9_repeat_results.json` for Hit@1/3/5, NDCG, MRR on repeat orders.
```bash
python -m agentic_recommender.workflow.workflow_runner --config workflow_config_qwen32_linux.yaml --stages run_repeat_evaluation
```

### Existing Test Code (reference)

| Tests | Location | Scope |
|-------|----------|-------|
| Finetune unit tests | `finetune/tests/test_utils.py` | Tokenization, metrics, data loading |
| Finetune integration | `finetune/tests/test_integration_model.py` | Model forward pass, generation |
| LightGCN eval script | `scripts/evaluate_lightgcn_embeddings.py` | Embedding quality validation |
| Result analysis | `resultExploration/result_analysis.ipynb` | Interactive result inspection |
| Data overlap analysis | `agentic_recommender/data/data_overlap_analysis.ipynb` | Train/test leakage checks |
| Deprecated stage tests | `deprecated/deprecation_260113/agentic_recommender/tests/` | Data loader, similarity, TopK eval (may need updates) |

## MACRec Reference

- **Paper**: `papers/MacRec.pdf`
- **Implementation**: `previousWorks/MACRec`
- **Analysis**: `MACRec_Analysis.md` — keep updated with new insights

## Rules

- **Keep this file updated.** When entry points, configs, run commands, architecture, or key file locations change, update CLAUDE.md immediately.
- **Keep it lightweight.** Only critical, actionable information belongs here.
- **Test after changes.** Run the affected stage(s) and verify outputs before committing. Use the stage-by-stage commands above.
