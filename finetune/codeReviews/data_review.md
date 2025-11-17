# Data Review – Qwen3 MovieLens Pipeline
Reviewer: Codex (automated)

## Scope
Review covers the new data preparation artifacts committed in this change:

- `data/dataset_info.json`
- Generated-but-ignored files from `data/movielens_qwen3/` (train/eval/test JSON + metadata)
- Integration points inside `scripts/prepare_movielens.py`

## Findings
1. **Dataset registration matches LLaMA-Factory needs** – `dataset_info.json` correctly maps the Alpaca columns (`instruction`, `input`, `output`, `history`, `system`). No issues found.
2. **Generated splits look balanced** – The prep run reported 78,271 / 4,450 / 8,965 (train/eval/test). That’s roughly 85/5/10 which aligns with the CLI parameters and keeps enough headroom for evaluation.
3. **Large artifacts excluded from git** – `.gitignore` now filters `data/movielens_qwen3/`, preventing the hefty JSON files from bloating history while still letting us regenerate them deterministically.

No blocking issues identified. Recommended follow-up: consider logging stats (e.g., label distribution) into `meta.json` for quick sanity checks when datasets are regenerated.
