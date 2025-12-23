# Chat Template Format - Complete Explanation

## Overview

When you pass a list of messages with roles to the tokenizer, it converts them into a **formatted string** with special tokens. This formatted string is what actually gets tokenized and sent to the model.

## The Transformation

### Input: Python Dictionary
```python
messages = [
    {"role": "system", "content": "You are a movie recommendation assistant..."},
    {"role": "user", "content": "User's last 15 watched movies:\nInception\n..."},
]
```

### Process: Apply Chat Template
```python
chat_str = tokenizer.apply_chat_template(
    messages,
    tokenize=False,           # Returns string, not tokens
    add_generation_prompt=True # Adds "<|im_start|>assistant\n" at the end
)
```

### Output: Formatted String (what `model_input` contains)
```
<|im_start|>system
You are a movie recommendation assistant...<|im_end|>
<|im_start|>user
User's last 15 watched movies:
Inception
The Matrix
...<|im_end|>
<|im_start|>assistant
```

This formatted string is then:
1. **Tokenized** into token IDs: `[151644, 8948, 198, 2610, 525, ...]`
2. **Fed to the model** for generation
3. **Saved as `model_input`** in your predictions.jsonl file

## Special Tokens Used by Qwen

Qwen models use the following special tokens:

| Token | Token ID | Purpose |
|-------|----------|---------|
| `<|im_start|>` | 151644 | Marks the start of a message |
| `<|im_end|>` | 151645 | Marks the end of a message |
| `<|endoftext|>` | 151643 | End of text (EOS token) |

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: Python Messages                                             │
├─────────────────────────────────────────────────────────────────────┤
│ [                                                                   │
│   {"role": "system", "content": "You are an assistant..."},        │
│   {"role": "user", "content": "Question?"},                        │
│ ]                                                                   │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
                  tokenizer.apply_chat_template()
                  (tokenize=False, add_generation_prompt=True)
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: Formatted String (model_input) ← SAVED IN PREDICTIONS.JSONL│
├─────────────────────────────────────────────────────────────────────┤
│ <|im_start|>system                                                  │
│ You are an assistant...<|im_end|>                                   │
│ <|im_start|>user                                                    │
│ Question?<|im_end|>                                                 │
│ <|im_start|>assistant                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
                        tokenizer(chat_str)
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: Token IDs                                                   │
├─────────────────────────────────────────────────────────────────────┤
│ input_ids: [151644, 8948, 198, 2610, 525, 264, 17847, ...]        │
│ attention_mask: [1, 1, 1, 1, 1, 1, 1, ...]                         │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
                        model.generate()
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: Generated Token IDs (input + new tokens)                   │
├─────────────────────────────────────────────────────────────────────┤
│ [151644, 8948, ..., 77091, 198, 1271, 11, 358, 649, 1492, ...]    │
│  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    │
│      Input tokens (STEP 3)        New generated tokens             │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
            tokenizer.decode(output, skip_special_tokens=True)
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: Decoded String (full_response) ← SAVED IN PREDICTIONS.JSONL│
├─────────────────────────────────────────────────────────────────────┤
│ system                                                              │
│ You are an assistant...                                             │
│ user                                                                │
│ Question?                                                           │
│ assistant                                                           │
│ Yes, I can help you with that.                                      │
│ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                       │
│ This is the generated part (assistant_response)                     │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
                Split by "assistant" and extract
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6: Extract Answer                                              │
├─────────────────────────────────────────────────────────────────────┤
│ assistant_response: "Yes, I can help you with that."               │
│ prediction: "Yes"                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## What Gets Saved in predictions.jsonl

Based on the flow above, here's what each field contains:

```json
{
  "user_prompt": "Question?",

  "model_input": "<|im_start|>system\nYou are an assistant...<|im_end|>\n<|im_start|>user\nQuestion?<|im_end|>\n<|im_start|>assistant\n",

  "full_response": "system\nYou are an assistant...\nuser\nQuestion?\nassistant\nYes, I can help you with that.",

  "assistant_response": "Yes, I can help you with that.",

  "prediction": "Yes"
}
```

## Key Differences Between Fields

### `model_input` vs `full_response`

| Aspect | `model_input` | `full_response` |
|--------|---------------|-----------------|
| **When created** | Before model.generate() | After model.generate() |
| **Special tokens** | Contains `<|im_start|>`, `<|im_end|>` | Special tokens removed (skip_special_tokens=True) |
| **Content** | Input only (prompt) | Input + generated response |
| **Purpose** | Show exact model input | Show complete decoded output |

### Example Comparison

**model_input** (372 chars):
```
<|im_start|>system
You are...<|im_end|>
<|im_start|>user
Question?<|im_end|>
<|im_start|>assistant
```

**full_response** (same input + generation, ~400 chars):
```
system
You are...
user
Question?
assistant
Yes, I can help you with that.
```

Notice that `full_response` doesn't have the special tokens (`<|im_start|>`, `<|im_end|>`) because we use `skip_special_tokens=True` during decoding.

## Why This Matters for Your Use Case

### For Training
- The tokenizer needs to know where to mask tokens (only train on assistant response)
- The `<|im_start|>assistant\n` marker is critical for identifying where the response starts
- See `tokenize_func()` in utils.py:208 - it finds this marker to create proper labels

### For Inference
- The model has been trained on this specific format
- It "knows" to start generating after seeing `<|im_start|>assistant\n`
- Adding `add_generation_prompt=True` is crucial for proper generation

### For Debugging
- **`model_input`** shows you EXACTLY what the model received (with special tokens)
- **`full_response`** shows you the complete decoded output (without special tokens)
- **`assistant_response`** shows you just the generated part
- This helps you debug if the model isn't responding correctly

## Code Example from Your Codebase

Here's exactly what happens in `utils.py`:

```python
# In generate() function (utils.py:619-628)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt},
]

chat_str = tokenizer.apply_chat_template(
    messages,
    tokenize=False,  # Returns the formatted string
    add_generation_prompt=True,  # Adds <|im_start|>assistant\n at the end
)
# chat_str is what gets saved as "model_input"

# Then tokenize it
inputs = tokenizer(chat_str, return_tensors="pt", padding=True)

# Generate
outputs = model.generate(inputs["input_ids"], ...)

# Decode (this becomes "full_response")
full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Running the Demo

To see this in action, run:
```bash
cd /workspace/AgenticRecommender/finetune/scripts
/venv/py311-cu128/bin/python demo_chat_template.py
```

This will show you:
1. The exact formatted strings for different scenarios
2. How tokens are encoded/decoded
3. The difference between inference and training formats
4. What special tokens look like

## Summary

**The Tokenizer's Job:**
```
messages (dict) → apply_chat_template() → formatted string with special tokens
                                         → tokenize() → token IDs
                                         → model.generate() → new token IDs
                                         → decode() → text response
```

**What You See in Output Files:**
- **`model_input`**: The formatted string with special tokens (before tokenization)
- **`full_response`**: The decoded output with special tokens removed (after generation)
- **`assistant_response`**: Just the generated part
- **`prediction`**: The extracted Yes/No answer

The key insight is that `model_input` shows you the **actual formatted text** that gets converted to tokens and fed to the model, including all the special tokens that guide the model's behavior!
