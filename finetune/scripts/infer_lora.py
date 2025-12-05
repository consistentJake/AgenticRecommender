#!/usr/bin/env python
"""Run inference comparing base Qwen3-0.6B vs MovieLens LoRA adapter."""

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER_DIR = "output/qwen3-movielens-qlora"
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.7
TOP_P = 0.9
SYSTEM_PROMPT = (
    "You are a movie recommendation assistant. Given a user's recent history and "
    "a candidate movie, respond with exactly one word: Yes or No."
)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def generate(prompt: str, model, tokenizer, device: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
            temperature=TEMPERATURE,
            top_p=TOP_P,
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

    quantization_config = None
    device_map = None
    torch_dtype = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        device_map = "auto"
        torch_dtype = torch.bfloat16

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if device_map is None:
        base_model = base_model.to(device)

    # Load LoRA adapter and merge into base
    peft_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
        is_trainable=False,
    )
    merged_model = peft_model.merge_and_unload()

    test_prompt = (
        "User's last 15 watched movies:\n"
        "1. The Matrix (1999) (rating ≈ 5.0)\n"
        "2. Terminator 2: Judgment Day (1991) (rating ≈ 4.5)\n"
        "3. Blade Runner (1982) (rating ≈ 4.0)\n"
        "...\n\n"
        "Candidate movie:\n"
        "Aliens (1986)\n\n"
        "Should we recommend this movie to the user? Answer Yes or No."
    )

    print("=== Base model ===")
    print(generate(test_prompt, base_model, tokenizer, device))
    print("\n=== LoRA-finetuned model ===")
    print(generate(test_prompt, merged_model, tokenizer, device))


if __name__ == "__main__":
    main()
