"""Comprehensive unit tests for utility functions."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml
from datasets import Dataset

# Add scripts directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from utils import (
    get_device,
    load_yaml_config,
    load_json,
    load_data,
    format_prompt,
    to_chat_messages,
    build_datasets,
    tokenize_func,
    preprocess_datasets_parallel,
    generate,
    extract_answer,
    compute_metrics,
    DEFAULT_SYSTEM_PROMPT,
)


# ==============================================================================
# Priority 1: Test tokenize_func (most critical)
# ==============================================================================

def test_tokenize_func_label_masking(sample_tokenizer, sample_formatted_text):
    """Verify prompt tokens are masked (-100) and assistant tokens preserved."""
    example = {"text": sample_formatted_text}
    result = tokenize_func(example, sample_tokenizer, cutoff_len=1024)

    # Check structure
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result

    # Verify labels contain both -100 (masked) and non--100 (preserved) values
    labels = result["labels"]
    assert -100 in labels, "Prompt tokens should be masked with -100"
    assert any(l != -100 for l in labels), "Some assistant tokens should be preserved"


def test_tokenize_func_length_consistency(sample_tokenizer, sample_formatted_text):
    """Ensure input_ids, attention_mask, and labels have same length."""
    example = {"text": sample_formatted_text}
    result = tokenize_func(example, sample_tokenizer, cutoff_len=1024)

    input_ids_len = len(result["input_ids"])
    attention_mask_len = len(result["attention_mask"])
    labels_len = len(result["labels"])

    assert input_ids_len == attention_mask_len == labels_len, \
        f"Lengths don't match: input_ids={input_ids_len}, attention_mask={attention_mask_len}, labels={labels_len}"


def test_tokenize_func_truncation(sample_tokenizer):
    """Test behavior when text exceeds cutoff_len (assistant should survive)."""
    # Create a very long text
    long_text = "<|im_start|>system\nYou are helpful.<|im_end|>\n"
    long_text += "<|im_start|>user\n" + "word " * 1000 + "<|im_end|>\n"
    long_text += "<|im_start|>assistant\nYes<|im_end|>"

    example = {"text": long_text}
    cutoff = 128
    result = tokenize_func(example, sample_tokenizer, cutoff_len=cutoff)

    # Should be truncated to cutoff_len and still retain assistant labels
    assert len(result["input_ids"]) <= cutoff
    assert len(result["labels"]) <= cutoff
    assert any(l != -100 for l in result["labels"]), "Assistant tokens should survive left truncation"


def test_tokenize_func_missing_marker(sample_tokenizer):
    """Edge case: no assistant marker in text."""
    # Text without proper assistant marker
    example = {"text": "Some random text without proper format"}
    result = tokenize_func(example, sample_tokenizer, cutoff_len=1024)

    # Should fallback to using all tokens (no masking)
    assert result["labels"] == result["input_ids"]


def test_tokenize_func_real_tokenizer(sample_tokenizer):
    """Test with actual Qwen tokenizer and realistic example."""
    messages = [
        {"role": "system", "content": "You are a movie recommendation assistant."},
        {"role": "user", "content": "Should I watch The Matrix?"},
        {"role": "assistant", "content": "Yes"}
    ]

    text = sample_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    ) + sample_tokenizer.eos_token

    example = {"text": text}
    result = tokenize_func(example, sample_tokenizer, cutoff_len=512)

    # Verify all required keys present
    assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}

    # Verify masking occurred
    assert -100 in result["labels"]
    assert any(l != -100 for l in result["labels"])


# ==============================================================================
# Priority 2: Test evaluation functions
# ==============================================================================

def test_extract_answer_yes_no():
    """Test Yes/No extraction from various formats."""
    assert extract_answer("Yes") == "Yes"
    assert extract_answer("No") == "No"
    assert extract_answer("yes, I agree") == "Yes"
    assert extract_answer("no, I don't think so") == "No"
    assert extract_answer("The answer is Yes.") == "Yes"
    assert extract_answer("The answer is No.") == "No"


def test_extract_answer_think_tags():
    """Test removal of <think> tags."""
    response = "<think>Let me consider this...</think> Yes"
    assert extract_answer(response) == "Yes"

    response = "<think>Hmm, not sure</think> No"
    assert extract_answer(response) == "No"


def test_extract_answer_assistant_prefix():
    """Test handling of 'assistant' prefix in response."""
    response = "system: ...\nuser: ...\nassistant: Yes, definitely"
    assert extract_answer(response) == "Yes"

    response = "user: Question?\nassistant: No"
    assert extract_answer(response) == "No"


def test_extract_answer_unknown():
    """Test ambiguous cases return 'Unknown'."""
    assert extract_answer("Maybe") == "Unknown"
    assert extract_answer("I don't know") == "No"  # Contains "no", so extracted as "No"
    assert extract_answer("Both yes and no") == "Unknown"  # Contains both, ambiguous
    assert extract_answer("") == "Unknown"
    assert extract_answer("Uncertain") == "Unknown"


def test_compute_metrics_accuracy(sample_predictions_and_labels):
    """Test accuracy calculation."""
    predictions, labels = sample_predictions_and_labels
    metrics = compute_metrics(predictions, labels)

    # predictions = ["Yes", "No", "Yes", "Unknown", "No"]
    # labels =      ["Yes", "Yes", "Yes", "No", "No"]
    # correct:       Yes    No     Yes    No       Yes
    # So 3/5 = 0.6
    assert metrics["accuracy"] == 0.6
    assert metrics["total"] == 5
    assert metrics["correct"] == 3


def test_compute_metrics_f1_precision_recall():
    """Test F1, precision, and recall calculations."""
    predictions = ["Yes", "No", "Yes", "No"]
    labels = ["Yes", "Yes", "Yes", "No"]

    metrics = compute_metrics(predictions, labels)

    # TP: pred=Yes, label=Yes -> 2
    # FP: pred=Yes, label=No -> 0
    # FN: pred=No, label=Yes -> 1
    # TN: pred=No, label=No -> 1

    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1

    # Precision: TP / (TP + FP) = 2 / 2 = 1.0
    assert metrics["precision"] == 1.0

    # Recall: TP / (TP + FN) = 2 / 3 ≈ 0.667
    assert abs(metrics["recall"] - 0.6666666) < 0.001

    # F1: 2 * (P * R) / (P + R)
    assert abs(metrics["f1"] - 0.8) < 0.001


# ==============================================================================
# Priority 3: Test data functions
# ==============================================================================

def test_load_json():
    """Test JSON loading."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"key": "value"}, f)
        temp_path = f.name

    try:
        data = load_json(temp_path)
        assert data == {"key": "value"}
    finally:
        Path(temp_path).unlink()


def test_load_json_file_not_found():
    """Test JSON loading with non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_json("/nonexistent/file.json")


def test_load_data_json_format():
    """Test load_data with JSON format."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([{"id": 1}, {"id": 2}], f)
        temp_path = f.name

    try:
        data = load_data(temp_path)
        assert data == [{"id": 1}, {"id": 2}]
    finally:
        Path(temp_path).unlink()


def test_load_data_jsonl_format():
    """Test load_data with JSONL format."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write('{"id": 1}\n')
        f.write('{"id": 2}\n')
        temp_path = f.name

    try:
        data = load_data(temp_path)
        assert data == [{"id": 1}, {"id": 2}]
    finally:
        Path(temp_path).unlink()


def test_format_prompt_training_format():
    """Test format_prompt with training data format."""
    example = {"input": "Test prompt"}
    assert format_prompt(example) == "Test prompt"


def test_format_prompt_test_format(sample_test_example):
    """Test format_prompt with test data format."""
    prompt = format_prompt(sample_test_example)

    assert "User's last 15 watched movies:" in prompt
    assert "The Matrix (1999)" in prompt
    assert "Tenet (2020)" in prompt
    assert "Should we recommend this movie to the user?" in prompt


def test_format_prompt_unknown_format():
    """Test format_prompt with unknown format raises error."""
    example = {"unknown_key": "value"}
    with pytest.raises(ValueError, match="Unknown data format"):
        format_prompt(example)


def test_to_chat_messages(sample_chat_example):
    """Test chat message formatting."""
    messages = to_chat_messages(sample_chat_example)

    # Should have system, user, and assistant messages
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[1]["role"] == "user"
    assert "Answer yes or no" in messages[1]["content"]
    assert "Is this a test?" in messages[1]["content"]
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Yes"


def test_to_chat_messages_no_system():
    """Test chat message formatting without system prompt."""
    example = {
        "instruction": "Question",
        "input": "Input text",
        "output": "Answer"
    }
    messages = to_chat_messages(example)

    # Should have only user and assistant messages
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"

    # Should combine instruction and input
    assert messages[0]["content"] == "Question\n\nInput text"


def test_to_chat_messages_empty_input():
    """Test that empty input field only uses instruction."""
    example = {
        "instruction": "Question only",
        "input": "",  # Empty input (common in test data)
        "output": "Answer"
    }
    messages = to_chat_messages(example)

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    # Should only use instruction when input is empty
    assert messages[0]["content"] == "Question only"


def test_to_generation_messages_combines_fields():
    """Test that to_generation_messages also combines instruction + input correctly."""
    example = {
        "instruction": "User orders:",
        "input": "Additional details",
        "system": "You are a recommender"
    }
    messages = to_generation_messages(example)

    assert len(messages) == 2  # system + user (no assistant)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "User orders:\n\nAdditional details"


def test_to_generation_messages_empty_input():
    """Test that to_generation_messages handles empty input correctly."""
    example = {
        "instruction": "User orders:",
        "input": "",
        "system": "You are a recommender"
    }
    messages = to_generation_messages(example)

    assert len(messages) == 2
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "User orders:"


def test_chat_template_and_tokenization_round_trip(sample_tokenizer, sample_chat_example):
    """Ensure chat example -> template -> tokenization keeps assistant label intact."""
    messages = to_chat_messages(sample_chat_example)
    text = sample_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    ) + sample_tokenizer.eos_token

    result = tokenize_func({"text": text}, sample_tokenizer, cutoff_len=512)

    # Assistant tokens (labels != -100) should decode back to the ground-truth output.
    assistant_tokens = [
        tok for tok, label in zip(result["input_ids"], result["labels"]) if label != -100
    ]
    decoded_answer = sample_tokenizer.decode(assistant_tokens, skip_special_tokens=True).strip().lower()

    assert "<|im_start|>assistant" in text
    assert decoded_answer.startswith(sample_chat_example["output"].lower())


# ==============================================================================
# Priority 4: Test other utilities
# ==============================================================================

def test_get_device():
    """Test device detection."""
    device = get_device()
    assert device in ["mps", "cuda", "cpu"]


def test_load_yaml_config():
    """Test YAML config loading."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({"key": "value", "number": 42}, f)
        temp_path = Path(f.name)

    try:
        config = load_yaml_config(temp_path)
        assert config["key"] == "value"
        assert config["number"] == 42
    finally:
        temp_path.unlink()


def test_load_yaml_config_not_found():
    """Test YAML loading with non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_yaml_config(Path("/nonexistent/config.yaml"))


def test_build_datasets():
    """Test Dataset creation from JSON files."""
    # Create temporary JSON files
    train_data = [{"id": 1, "text": "train1"}, {"id": 2, "text": "train2"}]
    eval_data = [{"id": 3, "text": "eval1"}]

    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = Path(tmpdir) / "train.json"
        eval_path = Path(tmpdir) / "eval.json"

        with open(train_path, 'w') as f:
            json.dump(train_data, f)
        with open(eval_path, 'w') as f:
            json.dump(eval_data, f)

        train_ds, eval_ds = build_datasets(str(train_path), str(eval_path))

        assert isinstance(train_ds, Dataset)
        assert isinstance(eval_ds, Dataset)
        assert len(train_ds) == 2
        assert len(eval_ds) == 1
        assert train_ds[0]["text"] == "train1"
        assert eval_ds[0]["text"] == "eval1"


# ==============================================================================
# Integration Test: preprocess_datasets_parallel
# ==============================================================================

def test_preprocess_datasets_parallel(sample_tokenizer, sample_chat_example):
    """Test complete preprocessing pipeline."""
    # Create small datasets
    train_data = [sample_chat_example, sample_chat_example]
    eval_data = [sample_chat_example]

    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data)

    # Formatting function
    def formatting_func(example):
        messages = to_chat_messages(example)
        return sample_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    # Preprocess without caching (num_proc=1 for simplicity)
    train_processed, eval_processed = preprocess_datasets_parallel(
        train_ds,
        eval_ds,
        sample_tokenizer,
        formatting_func,
        cutoff_len=512,
        cache_dir=None,
        num_proc=1,
        clear_cache=False
    )

    # Verify datasets have required columns
    assert "input_ids" in train_processed.column_names
    assert "attention_mask" in train_processed.column_names
    assert "labels" in train_processed.column_names

    # Verify sizes
    assert len(train_processed) == 2
    assert len(eval_processed) == 1

    # Verify tokenization worked
    first_example = train_processed[0]
    assert len(first_example["input_ids"]) > 0
    assert len(first_example["labels"]) == len(first_example["input_ids"])
    assert -100 in first_example["labels"]  # Prompt masking occurred


# ==============================================================================
# Integration-style checks for truncation and loss masking
# ==============================================================================

def test_cutoff_len_respects_mask_when_answer_fits(sample_tokenizer, sample_chat_example):
    """When the answer fits under cutoff_len, labels should include assistant tokens."""
    text = sample_tokenizer.apply_chat_template(
        to_chat_messages(sample_chat_example),
        tokenize=False,
        add_generation_prompt=False
    ) + sample_tokenizer.eos_token

    result = tokenize_func({"text": text}, sample_tokenizer, cutoff_len=128)

    assert -100 in result["labels"]
    assert any(l != -100 for l in result["labels"]), "Assistant tokens should be kept when within cutoff"


def test_loss_ignores_prompt_tokens():
    """Cross entropy should only use positions where labels != -100."""
    import torch

    # Toy example: first two positions are prompt (-100), last two are assistant labels.
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 3, 4]])

    vocab_size = 10
    logits = torch.zeros(1, 4, vocab_size)
    logits[0, 2, 3] = 5.0  # correct token 3
    logits[0, 3, 4] = 5.0  # correct token 4

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))

    # Loss should be finite and only depend on the last two positions.
    assert torch.isfinite(loss)
    assert loss.item() < 0.1


def test_inference_pipeline_with_mock_model(sample_tokenizer, sample_test_example):
    """End-to-end smoke test: prompt -> generate -> extract -> metrics."""
    import torch

    class MockModel:
        def generate(self, input_ids, attention_mask=None, max_new_tokens=8, **kwargs):
            # Always append tokens for " Yes"
            yes_tokens = sample_tokenizer.encode(" Yes", add_special_tokens=False)
            combined = torch.cat(
                [input_ids, torch.tensor([yes_tokens], device=input_ids.device)], dim=1
            )
            return combined

    prompt = format_prompt(sample_test_example)
    response = generate(
        prompt,
        MockModel(),
        sample_tokenizer,
        device="cpu",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        do_sample=False,
        max_new_tokens=4,
        temperature=0.0,
        top_p=1.0,
    )

    pred = extract_answer(response)
    metrics = compute_metrics([pred], [sample_test_example["label"]])

    assert pred == "Yes"
    assert metrics["accuracy"] == 1.0
