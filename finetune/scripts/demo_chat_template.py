#!/usr/bin/env python
"""Demonstrate how chat template formatting works for Qwen models.

This script shows exactly what text is generated when you pass a list of
messages with roles to tokenizer.apply_chat_template().
"""

from transformers import AutoTokenizer

# Load Qwen tokenizer (same as used in training/inference)
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Using smaller model for demo
print(f"Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("\n" + "="*80)
print("DEMONSTRATION: How Chat Template Works")
print("="*80)

# Example 1: System + User (for inference)
print("\n" + "-"*80)
print("Example 1: Inference Format (System + User)")
print("-"*80)

messages_inference = [
    {"role": "system", "content": "You are a movie recommendation assistant. Given a user's recent history and a candidate movie. Please begin your analysis with 'Yes' or 'No'."},
    {"role": "user", "content": "User's last 15 watched movies:\nInception\nThe Matrix\nInterstellar\n\nCandidate movie:\nTenet\n\nShould we recommend this movie to the user? Answer Yes or No."},
]

# Apply chat template WITHOUT generation prompt (shows complete conversation)
formatted_complete = tokenizer.apply_chat_template(
    messages_inference,
    tokenize=False,
    add_generation_prompt=False
)

print("\nFormatted output (add_generation_prompt=False):")
print("─" * 80)
print(formatted_complete)
print("─" * 80)

# Apply chat template WITH generation prompt (ready for model input)
formatted_for_generation = tokenizer.apply_chat_template(
    messages_inference,
    tokenize=False,
    add_generation_prompt=True
)

print("\nFormatted output (add_generation_prompt=True) - This is what model receives:")
print("─" * 80)
print(formatted_for_generation)
print("─" * 80)
print("\nNote: The generation prompt adds the 'assistant' marker at the end,")
print("      signaling the model to start generating the assistant's response.")

# Example 2: Complete conversation including assistant response (for training)
print("\n" + "-"*80)
print("Example 2: Training Format (System + User + Assistant)")
print("-"*80)

messages_training = [
    {"role": "system", "content": "You are a movie recommendation assistant. Given a user's recent history and a candidate movie. Please begin your analysis with 'Yes' or 'No'."},
    {"role": "user", "content": "User's last 15 watched movies:\nInception\nThe Matrix\nInterstellar\n\nCandidate movie:\nTenet\n\nShould we recommend this movie to the user? Answer Yes or No."},
    {"role": "assistant", "content": "<think>\nThe user likes sci-fi movies with complex plots (Inception, The Matrix, Interstellar). Tenet is also a sci-fi movie by Christopher Nolan with a complex time-inversion plot. This matches their preferences.\n</think>\n\nYes"},
]

formatted_training = tokenizer.apply_chat_template(
    messages_training,
    tokenize=False,
    add_generation_prompt=False
)

print("\nFormatted output for training:")
print("─" * 80)
print(formatted_training)
print("─" * 80)

# Example 3: Show the special tokens used
print("\n" + "-"*80)
print("Example 3: Understanding Special Tokens")
print("-"*80)

print("\nQwen chat template uses these special tokens:")
print("  <|im_start|> - Marks the start of a message")
print("  <|im_end|>   - Marks the end of a message")
print("\nFormat structure:")
print("  <|im_start|>ROLE\\nCONTENT<|im_end|>\\n")
print("\nWhere ROLE can be:")
print("  - system: System instructions/prompt")
print("  - user: User's input")
print("  - assistant: Model's response")

# Example 4: Show how this gets tokenized
print("\n" + "-"*80)
print("Example 4: Tokenization")
print("-"*80)

# Tokenize the formatted text
tokens = tokenizer(formatted_for_generation, return_tensors="pt")
token_ids = tokens["input_ids"][0]
token_count = len(token_ids)

print(f"\nFormatted text length: {len(formatted_for_generation)} characters")
print(f"Number of tokens: {token_count}")
print(f"\nFirst 20 token IDs: {token_ids[:20].tolist()}")
print(f"Last 10 token IDs: {token_ids[-10:].tolist()}")

# Decode individual tokens to see what they represent
print("\nFirst 10 tokens decoded:")
for i, token_id in enumerate(token_ids[:10]):
    decoded = tokenizer.decode([token_id])
    print(f"  Token {i}: {token_id} → '{decoded}'")

# Example 5: Show what happens during generation
print("\n" + "-"*80)
print("Example 5: What Happens During Inference")
print("-"*80)

print("\nStep-by-step process:")
print("1. Create messages with roles:")
print("   messages = [")
print("       {'role': 'system', 'content': '...'},")
print("       {'role': 'user', 'content': '...'},")
print("   ]")
print("\n2. Apply chat template (this is what model_input captures):")
print("   chat_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)")
print("\n3. Tokenize the formatted string:")
print("   inputs = tokenizer(chat_str, return_tensors='pt')")
print("\n4. Generate response:")
print("   outputs = model.generate(inputs['input_ids'])")
print("\n5. Decode the full output:")
print("   full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)")
print("   Note: skip_special_tokens=True removes <|im_start|> and <|im_end|> tokens")

# Example 6: Compare with and without skip_special_tokens
print("\n" + "-"*80)
print("Example 6: Decoding with vs without special tokens")
print("-"*80)

# Tokenize a simple message
simple_tokens = tokenizer(formatted_for_generation, return_tensors="pt")["input_ids"][0]

# Decode WITH special tokens
decoded_with_special = tokenizer.decode(simple_tokens, skip_special_tokens=False)
print("\nDecoded WITH special tokens (skip_special_tokens=False):")
print("─" * 80)
print(decoded_with_special[:500] + "..." if len(decoded_with_special) > 500 else decoded_with_special)
print("─" * 80)

# Decode WITHOUT special tokens
decoded_without_special = tokenizer.decode(simple_tokens, skip_special_tokens=True)
print("\nDecoded WITHOUT special tokens (skip_special_tokens=True):")
print("─" * 80)
print(decoded_without_special[:500] + "..." if len(decoded_without_special) > 500 else decoded_without_special)
print("─" * 80)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
The chat template converts structured messages into a formatted string:

Input (Python dict):
    [
        {"role": "system", "content": "You are..."},
        {"role": "user", "content": "Question?"},
    ]

Output (formatted string with special tokens):
    <|im_start|>system
    You are...<|im_end|>
    <|im_start|>user
    Question?<|im_end|>
    <|im_start|>assistant

This formatted string is what:
- Gets tokenized into token IDs
- Gets fed to the model as input
- Is saved as "model_input" in the predictions.jsonl file

When the model generates a response, it continues from where the last
<|im_start|>assistant\\n marker ends, generating tokens until it produces
the <|im_end|> token or reaches max_new_tokens.

The full decoded output (full_response) includes the input + generated part,
but with skip_special_tokens=True, so the <|im_start|> and <|im_end|> tokens
are removed, leaving just the role names (system, user, assistant) and content.
""")

print("="*80)
