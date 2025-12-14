"""Optional integration test that runs a real model forward + generate.

This test is skipped unless you supply a model via CLI or env:
  pytest tests/test_integration_model.py --integration-model-path Qwen/Qwen3-8B
or let it read model_name_or_path from a config:
  pytest tests/test_integration_model.py --integration-config-path configs/qwen3_7b_movielens_qlora.yaml

It reuses the same data prep (chat template + tokenize_func) and verifies:
 - Forward pass loss respects label masking (-100).
 - Greedy generation returns a parsable answer.
"""

from pathlib import Path

import pytest
import torch

# Add scripts directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from utils import (
    load_yaml_config,
    load_model_and_tokenizer,
    get_device,
    load_json,
    to_chat_messages,
    to_generation_messages,
    tokenize_func,
    generate,
    extract_answer,
    DEFAULT_SYSTEM_PROMPT,
)


def _resolve_model_path(cfg_path: str, direct: str | None) -> str | None:
    if direct:
        return direct
    cfg_file = Path(cfg_path)
    if cfg_file.exists():
        try:
            cfg = load_yaml_config(cfg_file)
            return cfg.get("model_name_or_path")
        except Exception:
            return None
    return None


def _resolve_device() -> str:
    return get_device()


def _log(msg: str) -> None:
    print(f"[integration] {msg}")


def _load_first_record(data_path: str):
    p = Path(data_path)
    if not p.exists():
        return None
    try:
        data = load_json(str(p))
        if isinstance(data, list) and data:
            return data[0]
    except Exception:
        return None
    return None


def test_real_model_forward_and_generate(integration_model_path, integration_config_path, integration_data_path, sample_chat_example):
    """Runs a single forward + generate using the configured chat model."""
    model_path = _resolve_model_path(integration_config_path, integration_model_path)
    if model_path is None:
        pytest.skip("Set --integration-model-path or provide config with model_name_or_path")

    # Load real eval record if available; fall back to synthetic sample.
    example = _load_first_record(integration_data_path) or sample_chat_example
    if example is None:
        pytest.skip("No integration data available")

    # Load config to reuse training-time settings.
    cfg = {}
    cfg_file = Path(integration_config_path)
    if cfg_file.exists():
        cfg = load_yaml_config(cfg_file)
    cfg["model_name_or_path"] = model_path

    device = _resolve_device()
    _log(f"using device={device}, model={model_path}")

    model, tokenizer = load_model_and_tokenizer(cfg, device, disable_cache=True)
    model.eval()

    # Prepare chat text using the same template and tokenize with masking.
    messages = to_chat_messages(example)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ) + tokenizer.eos_token
    _log(f"raw input json: {example}")
    _log(f"templated text (truncated): {text[:200]!r}...")

    encoded = tokenize_func({"text": text}, tokenizer, cutoff_len=256)

    input_ids = torch.tensor([encoded["input_ids"]], device=device)
    attention_mask = torch.tensor([encoded["attention_mask"]], device=device)
    labels = torch.tensor([encoded["labels"]], device=device)

    masked = (labels == -100).sum().item()
    kept = (labels != -100).sum().item()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

    print(f"[integration] seq_len={input_ids.shape[1]}, masked={masked}, kept={kept}, loss={loss.item():.4f}")

    # Sanity checks: shapes align, some labels kept, and loss is finite/reasonable.
    assert input_ids.shape[1] == labels.shape[1]
    assert kept > 0, "Assistant labels should survive cutoff_len"
    assert torch.isfinite(loss)
    assert loss.item() < 50, "Loss unexpectedly large; check masking/tokenization"

    # Greedy generate a short answer and ensure it parses (even if random).
    # Use to_generation_messages to properly format the prompt with instruction + input
    gen_messages = to_generation_messages(example)
    chat_str = tokenizer.apply_chat_template(
        gen_messages,
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
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(out[0], skip_special_tokens=True)
    parsed = extract_answer(response)
    _log(f"model raw response (truncated): {response[:200]!r}")
    _log(f"parsed answer: {parsed}")
    assert isinstance(parsed, str)
    assert parsed.lower() in {"yes", "no", "unknown"}
