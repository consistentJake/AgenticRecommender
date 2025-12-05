#!/usr/bin/env python
"""
Compare base Qwen vs LoRA-finetuned adapter on a test prompt.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "./qwen2_5_0_5b_lora_simplifier"
MAX_NEW_TOKENS = 100


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def generate(prompt: str, model, tokenizer, device: str):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    chat_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        chat_str,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )

    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Load base model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    ).to(device)

    # Load LoRA adapter and merge into base
    peft_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
        is_trainable=False,
    )
    merged_model = peft_model.merge_and_unload()

    test_prompt = "Rewrite in simpler terms: 'There is a growing consensus on the issue.'"

    print("=== Base model ===")
    print(generate(test_prompt, base_model, tokenizer, device))
    print("\n=== LoRA-finetuned model ===")
    print(generate(test_prompt, merged_model, tokenizer, device))


if __name__ == "__main__":
    main()
