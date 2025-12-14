"""Pytest fixtures for utility function tests."""

import pytest
from transformers import AutoTokenizer


@pytest.fixture(scope="session")
def sample_tokenizer():
    """Load real Qwen tokenizer for testing."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


@pytest.fixture
def sample_chat_example():
    """Sample training example in standard format."""
    return {
        "system": "You are a helpful assistant.",
        "instruction": "Answer yes or no.",
        "input": "Is this a test?",
        "output": "Yes",
        "history": []
    }


@pytest.fixture
def sample_formatted_text(sample_tokenizer):
    """Pre-formatted text with Qwen chat template."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Answer yes or no.\n\nIs this a test?"},
        {"role": "assistant", "content": "Yes"}
    ]
    text = sample_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return text + sample_tokenizer.eos_token


@pytest.fixture
def sample_predictions_and_labels():
    """Sample predictions and labels for metrics testing."""
    predictions = ["Yes", "No", "Yes", "Unknown", "No"]
    labels = ["Yes", "Yes", "Yes", "No", "No"]
    return predictions, labels


@pytest.fixture
def sample_test_example():
    """Sample test data in JSONL format."""
    return {
        "user_id": "123",
        "history_titles": [
            "1. The Matrix (1999) (rating ≈ 5.0)",
            "2. Inception (2010) (rating ≈ 4.5)",
            "3. Interstellar (2014) (rating ≈ 4.0)"
        ],
        "candidate_title": "Tenet (2020)",
        "label": "Yes"
    }


# ---------------------------------------------------------------------------
# Integration options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--integration-model-path",
        action="store",
        default=None,
        help="Model ID or local path for integration test (overrides env/config)",
    )
    parser.addoption(
        "--integration-config-path",
        action="store",
        default="configs/qwen3_7b_movielens_qlora.yaml",
        help="Config file to read model_name_or_path when no model path is provided",
    )
    parser.addoption(
        "--integration-data-path",
        action="store",
        default="data/movielens_qwen3/eval.json",
        help="Path to eval/test data to feed into integration test (first record used)",
    )


@pytest.fixture(scope="session")
def integration_model_path(request):
    """Resolve model path from CLI option > env > config (if exists)."""
    opt_model = request.config.getoption("--integration-model-path")
    if opt_model:
        return opt_model

    import os

    env_model = os.getenv("INTEGRATION_MODEL_PATH")
    if env_model:
        return env_model

    return None


@pytest.fixture(scope="session")
def integration_config_path(request):
    """Config path from CLI option (defaults to qwen3_7b config)."""
    return request.config.getoption("--integration-config-path")


@pytest.fixture(scope="session")
def integration_data_path(request):
    """Data path from CLI option (defaults to first eval record)."""
    return request.config.getoption("--integration-data-path")
