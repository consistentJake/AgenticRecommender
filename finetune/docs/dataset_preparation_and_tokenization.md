# Dataset Preparation, Template Formatting, and Tokenization Guide

This document explains the complete pipeline from raw MovieLens data to tokenized inputs for the Qwen model, including concrete examples and analysis of sequence length distribution.

## Table of Contents

1. [Overview](#overview)
2. [Data Preparation Pipeline](#data-preparation-pipeline)
3. [Template Formatting](#template-formatting)
4. [Tokenization Process](#tokenization-process)
5. [Sequence Length Analysis](#sequence-length-analysis)
6. [Cutoff Length and Truncation Strategy](#cutoff-length-and-truncation-strategy)
7. [Complete Example: JSON to Tokens](#complete-example-json-to-tokens)

---

## Overview

The pipeline transforms raw MovieLens ratings into a supervised fine-tuning dataset for movie recommendation. The key stages are:

```
Raw Data → Samples → Alpaca Format → Chat Messages → Formatted String → Tokens → Training
```

---

## Data Preparation Pipeline

### Script: `scripts/prepare_movielens.py`

This script creates the training dataset from MovieLens ratings.

### Step-by-Step Process

#### 1. Load Raw Data
- Reads `ratings.csv` (userId, movieId, rating, timestamp)
- Reads `movies.csv` (movieId, title, genres)
- Creates movie ID to title mapping

#### 2. Build User Timelines
- Groups ratings by user and sorts by timestamp
- Filters users with >15 rated movies
- Uses first 15 movies as "warm-up" context
- Remaining movies become training examples

#### 3. Create Training Samples
For each movie after the first 15:
```python
Sample(
    history_titles="1. Movie A (rating ≈ 4.0)\n2. Movie B (rating ≈ 3.5)\n...",
    candidate_title="Movie X",
    label="Yes" if rating >= 4.0 else "No",
    split="train"  # or "val" or "test"
)
```

#### 4. Convert to Alpaca Format
```json
{
  "instruction": "Predict whether the user will like the candidate movie. Answer only with Yes or No.",
  "input": "User's last 15 watched movies:\n1. Pearl Harbor (2001) (rating ≈ 3.0)\n...\n15. Babe: Pig in the City (1998) (rating ≈ 3.5)\n\nCandidate movie:\nMoney Pit, The (1986)\n\nShould we recommend this movie to the user? Answer Yes or No.",
  "output": "No",
  "system": "You are a movie recommendation assistant. Given a user's recent history and a candidate movie. Please begin your analysis with 'Yes' or 'No'.",
  "history": []
}
```

#### 5. Save Datasets
- `data/movielens_qwen3/train.json`: 78,271 examples
- `data/movielens_qwen3/eval.json`: 150 examples
- `data/movielens_qwen3/test_raw.jsonl`: 8,965 examples

### Configuration Parameters

From `scripts/prepare_movielens.py`:

```python
--history-len: 15              # Number of past movies for context
--rating-threshold: 4.0        # Label threshold (≥ 4.0 is "Yes")
--val-ratio: 0.05              # Validation split ratio
--test-ratio: 0.1              # Test split ratio
--seed: 13                     # Random seed for reproducibility
```

---

## Template Formatting

### Converting Alpaca Format to Chat Messages

**Function**: `scripts/utils.py::to_chat_messages()` (lines 121-147)

This function converts the Alpaca format JSON to chat-style messages:

```python
def to_chat_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = []

    # 1. Add system message
    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})

    # 2. Add history (currently empty but kept for future use)
    for turn in example.get("history") or []:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # 3. Combine instruction and input for user message
    user_content = example.get("instruction", "")
    if example.get("input"):
        user_content = f"{user_content}\n\n{example['input']}"
    messages.append({"role": "user", "content": user_content})

    # 4. Add assistant response (for training)
    messages.append({"role": "assistant", "content": example.get("output", "")})

    return messages
```

**Output**:
```python
[
    {"role": "system", "content": "You are a movie recommendation assistant..."},
    {"role": "user", "content": "Predict whether the user will like...\n\nUser's last 15 watched movies:\n..."},
    {"role": "assistant", "content": "No"}
]
```

### Applying Qwen Chat Template

**Function**: `tokenizer.apply_chat_template()` from HuggingFace Transformers

The Qwen tokenizer has a built-in chat template that formats messages with special tokens:

```python
chat_str = tokenizer.apply_chat_template(
    messages,
    tokenize=False,           # Returns string, not tokens
    add_generation_prompt=False  # Include assistant response (for training)
)
```

**Output**: Formatted string with Qwen's special tokens
```
<|im_start|>system
You are a movie recommendation assistant. Given a user's recent history and a candidate movie. Please begin your analysis with 'Yes' or 'No'.<|im_end|>
<|im_start|>user
Predict whether the user will like the candidate movie. Answer only with Yes or No.

User's last 15 watched movies:
1. Pearl Harbor (2001) (rating ≈ 3.0)
2. Moulin Rouge (2001) (rating ≈ 0.5)
3. Lost Boys, The (1987) (rating ≈ 3.0)
4. Buffy the Vampire Slayer (1992) (rating ≈ 2.5)
5. Mariachi, El (1992) (rating ≈ 3.0)
6. Bodyguard, The (1992) (rating ≈ 2.5)
7. Grumpy Old Men (1993) (rating ≈ 3.0)
8. Animal House (1978) (rating ≈ 3.5)
9. Teenage Mutant Ninja Turtles (1990) (rating ≈ 2.0)
10. 28 Days (2000) (rating ≈ 2.5)
11. World Is Not Enough, The (1999) (rating ≈ 4.0)
12. Thirteenth Floor, The (1999) (rating ≈ 3.5)
13. Wild Wild West (1999) (rating ≈ 2.5)
14. Boys Don't Cry (1999) (rating ≈ 1.0)
15. Babe: Pig in the City (1998) (rating ≈ 3.5)

Candidate movie:
Money Pit, The (1986)

Should we recommend this movie to the user? Answer Yes or No.<|im_end|>
<|im_start|>assistant
No<|im_end|>
```

**Special Tokens**:
- `<|im_start|>`: Marks the start of a message
- `<|im_end|>`: Marks the end of a message
- Role indicators: `system`, `user`, `assistant`

### For Inference (Without Assistant Response)

**Function**: `scripts/utils.py::to_generation_messages()` (lines 150-181)

For inference/evaluation, use this function which excludes the assistant's response:

```python
messages = to_generation_messages(example)  # No assistant message
prompt_str = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  # Adds "<|im_start|>assistant\n" at the end
)
```

This creates a prompt ready for the model to generate a response.

---

## Tokenization Process

### The Critical Function: `tokenize_func()`

**Location**: `scripts/utils.py::tokenize_func()` (lines 202-285)

This function implements **supervised fine-tuning with label masking**. It's crucial for ensuring only the assistant's response contributes to the training loss.

### Key Concept: Label Masking

During training, we want:
- **Prompt tokens**: Ignored in loss calculation (masked with -100)
- **Response tokens**: Used for training (actual token IDs)

PyTorch's `CrossEntropyLoss` ignores labels with value `-100` by default.

### Step-by-Step Tokenization

#### Step 1: Tokenize Full Text
```python
full_tokens = tokenizer(
    example["text"],  # Full formatted chat string
    truncation=False,  # Don't truncate yet
    padding=False
)
full_ids = full_tokens["input_ids"]
```

**Example**: `[151644, 8948, 198, 2610, 525, ..., 2753, 151645, 198]` (427 tokens)

#### Step 2: Find Assistant Response Start
```python
assistant_marker = "<|im_start|>assistant\n"
if assistant_marker in text:
    before_assistant = text.split(assistant_marker)[0] + assistant_marker
    prompt_tokens = tokenizer(before_assistant, truncation=False, padding=False)
    prompt_len = len(prompt_tokens["input_ids"])
```

This identifies how many tokens are in the prompt (everything before the assistant's actual response).

#### Step 3: Create Labels with Masking
```python
labels = [-100] * prompt_len + full_ids[prompt_len:]
```

**Example**:
```python
# If prompt_len = 420 and full_ids has 427 tokens:
labels = [-100, -100, -100, ..., -100,  # 420 masked tokens
          token_421, token_422, ..., token_427]  # 7 response tokens
```

**Masking Ratio**: Typically ~98.4% of tokens are masked, only the "Yes" or "No" response is trained.

#### Step 4: Left-Side Truncation (if needed)
```python
if len(full_ids) > cutoff_len:
    start = len(full_ids) - cutoff_len
    full_ids = full_ids[start:]
    full_attn = full_attn[start:]
    labels = labels[start:]
```

**Important**: Truncation is from the LEFT side to preserve the assistant's response on the right.

#### Step 5: Return Tokenized Example
```python
return {
    "input_ids": full_ids,        # Shape: (seq_len,)
    "attention_mask": full_attn,  # Shape: (seq_len,) - all 1s
    "labels": labels              # Shape: (seq_len,) - mix of -100 and token IDs
}
```

### Parallel Preprocessing

**Function**: `scripts/utils.py::preprocess_datasets_parallel()` (lines 358-491)

This function applies tokenization to the entire dataset using multiprocessing:

```python
tokenize_fn = partial(tokenize_func, tokenizer=tokenizer, cutoff_len=cutoff_len)
train_ds = train_ds.map(
    tokenize_fn,
    num_proc=8,  # Use 8 CPU cores
    remove_columns=train_ds.column_names
)
```

**Caching**: Preprocessed datasets are cached in `.cache/preprocessed/` to avoid re-tokenization.

---

## Sequence Length Analysis

### Running the Analysis Script

The updated `scripts/check_seq_len.py` provides detailed analysis of sequence lengths:

```bash
# Show examples and statistics
python scripts/check_seq_len.py --config configs/qwen3_7b_movielens_qlora.yaml --show-examples 2

# Full dataset analysis without examples
python scripts/check_seq_len.py --config configs/qwen3_7b_movielens_qlora.yaml --show-examples 0
```

### Results from Full Dataset (78,271 samples)

```
Statistics:
  min: 393 tokens
  max: 875 tokens
  mean: 435 tokens
  p50: 432 tokens
  p90: 458 tokens
  p95: 468 tokens
  p99: 493 tokens
  p100: 875 tokens

Truncation Analysis:
  Samples > 1024 tokens: 0 (0.0%)
  Samples <= 1024 tokens: 78,271 (100.0%)
```

### Key Findings

1. **No truncation occurs**: All samples are well below the 1024 token cutoff
2. **Average length**: ~435 tokens (42% of cutoff)
3. **Maximum length**: 875 tokens (85% of cutoff)
4. **Headroom**: 149 tokens (17%) between max and cutoff

---

## Cutoff Length and Truncation Strategy

### Is 1024 the Right Cutoff Length?

**Answer: YES, and it's even conservative.**

**Evidence**:
- ✅ 100% of samples fit within 1024 tokens
- ✅ Maximum sequence length is 875 tokens (15% below cutoff)
- ✅ 99th percentile is only 493 tokens (52% below cutoff)
- ✅ Mean is 435 tokens (57% below cutoff)

**Implications**:
1. **No data loss**: No samples are being truncated
2. **Headroom for variability**: Room for longer movie titles or more history
3. **Could use smaller cutoff**: For memory efficiency, could reduce to 896 or 512 tokens

**Memory vs. Safety Trade-off**:
```
Cutoff   Memory Savings   Risk
512      50%             May truncate outliers (>875 tokens if they exist in full dataset)
768      25%             Safe for current data (max 875)
896      12.5%           Minimal risk
1024     0% (current)    Maximum safety
```

**Recommendation**: Keep 1024 for safety, or reduce to 896 if memory is constrained.

### Is Left-Side Truncation the Right Approach?

**Answer: YES for SFT, but NOT currently happening.**

**Current Status**:
- ✅ No samples are being truncated (all < 1024 tokens)
- ⚠️ Left truncation logic exists but is never triggered

**If Truncation Were Happening**:

**Pros of Left-Side Truncation**:
1. ✅ **Preserves assistant response**: Essential for training
2. ✅ **Maintains labels alignment**: Response tokens stay intact
3. ✅ **Ensures training signal**: Model always sees the target output

**Cons of Left-Side Truncation (for recommendation tasks)**:
1. ❌ **Loses early history**: First movies in user's timeline are removed
2. ❌ **Temporal bias**: Model only sees recent preferences
3. ❌ **Context degradation**: May miss important patterns in early history

**Alternative Approaches (if truncation were needed)**:

#### Option 1: Right-Side Truncation
```python
# Keep early history, risk cutting response
if len(full_ids) > cutoff_len:
    full_ids = full_ids[:cutoff_len]
    labels = labels[:cutoff_len]
```
**Problem**: May cut the assistant's response, breaking training.

#### Option 2: Middle Truncation
```python
# Remove middle movies, keep early + late + response
if len(full_ids) > cutoff_len:
    keep_prefix = 200  # Early history
    keep_suffix = cutoff_len - keep_prefix
    full_ids = full_ids[:keep_prefix] + full_ids[-keep_suffix:]
```
**Complexity**: Requires careful implementation.

#### Option 3: Reduce History Length
```python
# In prepare_movielens.py
--history-len: 10  # Instead of 15
```
**Best solution**: Prevents truncation at the source.

### Recommendations

**For Current Dataset**:
1. ✅ **Keep cutoff_len=1024**: No changes needed
2. ✅ **Keep left-side truncation**: Correct approach for SFT
3. ⚡ **Optional optimization**: Reduce to 896 to save 12.5% memory

**If Adding Longer Sequences in Future**:
1. Monitor sequence lengths with `check_seq_len.py`
2. If truncation >1%, consider reducing `--history-len` in data preparation
3. Avoid increasing `cutoff_len` beyond model's max context (likely 8192 for Qwen3)

---

## Complete Example: JSON to Tokens

### Input JSON
```json
{
  "instruction": "Predict whether the user will like the candidate movie. Answer only with Yes or No.",
  "input": "User's last 15 watched movies:\n1. Pearl Harbor (2001) (rating ≈ 3.0)\n2. Moulin Rouge (2001) (rating ≈ 0.5)\n...\n15. Babe: Pig in the City (1998) (rating ≈ 3.5)\n\nCandidate movie:\nMoney Pit, The (1986)\n\nShould we recommend this movie to the user? Answer Yes or No.",
  "output": "No",
  "system": "You are a movie recommendation assistant...",
  "history": []
}
```

### Step 1: Convert to Chat Messages
```python
messages = [
    {"role": "system", "content": "You are a movie recommendation assistant..."},
    {"role": "user", "content": "Predict whether the user will like...\n\nUser's last 15..."},
    {"role": "assistant", "content": "No"}
]
```

### Step 2: Apply Chat Template
```
<|im_start|>system
You are a movie recommendation assistant. Given a user's recent history and a candidate movie. Please begin your analysis with 'Yes' or 'No'.<|im_end|>
<|im_start|>user
Predict whether the user will like the candidate movie. Answer only with Yes or No.

User's last 15 watched movies:
1. Pearl Harbor (2001) (rating ≈ 3.0)
2. Moulin Rouge (2001) (rating ≈ 0.5)
...
15. Babe: Pig in the City (1998) (rating ≈ 3.5)

Candidate movie:
Money Pit, The (1986)

Should we recommend this movie to the user? Answer Yes or No.<|im_end|>
<|im_start|>assistant
No<|im_end|>
```

### Step 3: Tokenize
```python
input_ids = [151644, 8948, 198, 2610, 525, ..., 2753, 151645, 198]
# Total: 427 tokens
```

### Step 4: Create Labels with Masking
```python
labels = [
    -100, -100, -100, ..., -100,  # First 420 tokens (prompt)
    151667, 271, 151668, 271, 2753, 151645, 198  # Last 7 tokens (response)
]
```

### Step 5: Return Tokenized Example
```python
{
    "input_ids": [427 tokens],
    "attention_mask": [1, 1, 1, ..., 1],  # All 1s
    "labels": [420 x -100, then 7 token IDs]
}
```

### During Training

**Forward Pass**:
```python
logits = model(input_ids)  # Predict next token for all positions
```

**Loss Calculation**:
```python
loss = CrossEntropyLoss(ignore_index=-100)(logits, labels)
# Only computes loss on the 7 non-masked response tokens
```

**Result**: Model learns to predict "No" given the user's history and candidate movie.

---

## Summary

### Pipeline Overview
1. **Data Preparation** (`prepare_movielens.py`): MovieLens → Alpaca JSON
2. **Template Formatting** (`to_chat_messages` + `apply_chat_template`): JSON → Formatted string with special tokens
3. **Tokenization** (`tokenize_func`): String → Token IDs with label masking
4. **Training** (SFTTrainer): Only learns from assistant responses

### Key Insights
- **Cutoff length (1024)**: Appropriate and conservative
- **Sequence lengths**: All samples fit comfortably (mean 435, max 875)
- **Left-side truncation**: Correct approach for SFT, but not triggered in current dataset
- **Label masking**: 98.4% of tokens masked, only "Yes"/"No" response trained
- **No optimization needed**: Current configuration is working well

### Tools for Analysis
```bash
# Check sequence lengths with examples
python scripts/check_seq_len.py --config configs/qwen3_7b_movielens_qlora.yaml --show-examples 2

# Full analysis
python scripts/check_seq_len.py --config configs/qwen3_7b_movielens_qlora.yaml --show-examples 0

# Custom cutoff analysis
python scripts/check_seq_len.py --config configs/qwen3_7b_movielens_qlora.yaml --cutoff-len 512
```

### References
- Data preparation: `scripts/prepare_movielens.py`
- Utilities: `scripts/utils.py`
- Training: `scripts/finetune_lora.py`
- Inference: `scripts/infer_lora.py`
- Config: `configs/qwen3_7b_movielens_qlora.yaml`
