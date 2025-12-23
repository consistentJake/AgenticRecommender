# Understanding Inference Output Fields

This document explains the different fields in the prediction output files and how they relate to each other.

## The Problem You Identified

Previously, the `*_predictions.jsonl` files only contained:
- `instruction`: From the original test data
- `full_response`: The decoded model output (which looked confusing because it included chat template markers)

You couldn't see the **actual formatted input** sent to the model, making it hard to debug what the model actually received.

## The Solution

The script now captures and saves the complete inference pipeline with these fields:

### Output Fields Explained

Each line in `base_predictions.jsonl` and `lora_predictions.jsonl` now contains:

```json
{
  "label": "Yes",                    // Ground truth answer
  "prediction": "No",                // Extracted answer from model
  "user_prompt": "...",              // User's prompt text only
  "model_input": "...",              // ← NEW! The actual formatted input
  "assistant_response": "...",       // Only generated text
  "full_response": "...",            // Complete decoded output
  "instruction": "...",              // From original data (if present)
  "input": "..."                     // From original data (if present)
}
```

### Field Relationships

Here's how these fields relate to each other:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. user_prompt (plain text)                                 │
│    "User's last 15 watched movies:\n..."                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
          [Apply chat template with system prompt]
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. model_input (formatted for model)                        │
│    "<|im_start|>system\n                                    │
│     You are a food delivery recommendation assistant...\n   │
│     <|im_end|>\n                                            │
│     <|im_start|>user\n                                      │
│     User's last 15 watched movies:\n...\n                   │
│     <|im_end|>\n                                            │
│     <|im_start|>assistant\n"                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
                  [Model generates]
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. assistant_response (generated only)                      │
│    "<think>\n...\n</think>\n\nNo"                           │
└─────────────────────────────────────────────────────────────┘
                          ↓
                  [Decode full output]
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. full_response (model_input + assistant_response)         │
│    "system\n                                                │
│     You are a food delivery recommendation assistant...\n   │
│     user\n                                                  │
│     User's last 15 watched movies:\n...\n                   │
│     assistant\n                                             │
│     <think>\n...\n</think>\n\nNo"                           │
└─────────────────────────────────────────────────────────────┘
                          ↓
                [Extract answer (Yes/No)]
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. prediction                                               │
│    "No"                                                     │
└─────────────────────────────────────────────────────────────┘
```

## Why These Fields Exist

### Historical/Original Fields

- **`instruction`**: From your original training data format - this is the instruction template
- **`input`**: From your original training data format - contains the user history and candidate

### User-Facing Fields

- **`user_prompt`**: Clean text that a human would understand (no special tokens)
  - Useful for: Reading and understanding what the model was asked

### Debugging Fields

- **`model_input`**: THE KEY FIELD YOU WANTED! Shows exactly what text was sent to the model
  - Includes: System prompt, chat template markers (`<|im_start|>`, `<|im_end|>`), and user prompt
  - Useful for: Debugging tokenization, seeing exact input, reproducing inference

- **`assistant_response`**: Only the generated part (after the "assistant" marker)
  - Useful for: Seeing what the model generated without the prompt

- **`full_response`**: Complete decoded output from the model
  - This is what you were seeing before that looked confusing
  - It's the tokenizer decoding all the output tokens (input + generated)
  - Useful for: Seeing the complete output as the model tokenizer decodes it

### Evaluation Fields

- **`label`**: Ground truth (from test data)
- **`prediction`**: Extracted Yes/No/Unknown from `assistant_response`

## Code Changes

### 1. Updated `utils.py` generation functions

Added optional parameters to return the formatted input text:

```python
# Single generation
response, input_text = generate(
    prompt, model, tokenizer, device,
    return_input_text=True  # ← NEW parameter
)

# Batch generation
responses, input_texts = generate_batch(
    prompts, model, tokenizer, device,
    return_input_texts=True  # ← NEW parameter
)
```

### 2. Updated `compare_base_vs_lora.py`

- Captures `input_text` from generation functions
- Saves as `model_input` field in output
- Renamed `prompt` → `user_prompt` for clarity
- Added comprehensive documentation

## Example Output

Here's what a typical record looks like now:

```json
{
  "label": "Yes",
  "prediction": "No",
  "user_prompt": "User's last 15 watched movies:\nInception\nThe Matrix\n...\n\nCandidate movie:\nInterstellar\n\nShould we recommend this movie to the user? Answer Yes or No.",
  "model_input": "<|im_start|>system\nYou are a movie recommendation assistant. Given a user's recent history and a candidate movie. Please begin your analysis with 'Yes' or 'No'.<|im_end|>\n<|im_start|>user\nUser's last 15 watched movies:\nInception\nThe Matrix\n...\n\nCandidate movie:\nInterstellar\n\nShould we recommend this movie to the user? Answer Yes or No.<|im_end|>\n<|im_start|>assistant\n",
  "assistant_response": "<think>\nThe user has watched sci-fi movies like Inception and The Matrix. Interstellar is also a sci-fi movie with similar themes.\n</think>\n\nNo",
  "full_response": "system\nYou are a movie recommendation assistant. Given a user's recent history and a candidate movie. Please begin your analysis with 'Yes' or 'No'.\nuser\nUser's last 15 watched movies:\nInception\nThe Matrix\n...\n\nCandidate movie:\nInterstellar\n\nShould we recommend this movie to the user? Answer Yes or No.\nassistant\n<think>\nThe user has watched sci-fi movies like Inception and The Matrix. Interstellar is also a sci-fi movie with similar themes.\n</think>\n\nNo",
  "user_id": "user_123",
  "candidate_title": "Interstellar",
  "instruction": "You are a movie recommendation assistant...",
  "input": "User's last 15 watched movies:\nInception\n..."
}
```

## Summary

**Before**: You could only see `instruction` (from original data) and `full_response` (confusing decoded output)

**Now**: You have the complete pipeline:
1. `user_prompt` - human-readable prompt
2. **`model_input`** - exact text sent to model (THIS IS WHAT YOU WANTED!)
3. `assistant_response` - only generated text
4. `full_response` - complete decoded output
5. `prediction` - extracted answer

The key addition is **`model_input`**, which shows you exactly what the model receives, including all the chat template formatting and special tokens.
