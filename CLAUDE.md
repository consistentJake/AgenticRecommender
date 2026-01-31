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

## MACRec Reference

- **Paper**: `papers/MacRec.pdf`
- **Implementation**: `previousWorks/MACRec`
- **Analysis**: `MACRec_Analysis.md` — keep updated with new insights

## Rules

- **Keep this file updated.** When entry points, configs, run commands, architecture, or key file locations change, update CLAUDE.md immediately.
- **Keep it lightweight.** Only critical, actionable information belongs here.
