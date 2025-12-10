## Data Directory (`/data`)

### `movielens_qwen3/`
**Purpose:** Processed MovieLens dataset for Qwen3 chat format

**Files:**
- `train.json` - Training examples in Alpaca format
  - Fields: instruction, input, output, system, history
  - Format: User history + candidate movie → Yes/No prediction

- `eval.json` - Validation/evaluation examples
  - Same format as train.json
  - Used for periodic evaluation during training

- `test_raw.jsonl` - Raw test examples (JSONL format)
  - Fields: user_id, history_titles, candidate_title, label
  - Used for final model evaluation

- `meta.json` - Dataset metadata
  - Contains: history_len, rating_threshold, split ratios, counts

- `dataset_info.json` - Dataset registry for LLaMA-Factory compatibility
  - Maps dataset names to file paths

**Example Record Structure:**
```json
{
  "instruction": "Predict whether the user will like the candidate movie...",
  "input": "User's last 15 watched movies:\n1. The Matrix (1999) (rating ≈ 5.0)...",
  "output": "Yes",
  "system": "You are a movie recommendation assistant...",
  "history": []
}
```