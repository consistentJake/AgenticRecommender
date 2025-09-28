"""Utility helpers for loading prompt templates used by multi-agent components."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=None)
def load_prompt_config(name: str) -> Dict[str, Any]:
    """Load a prompt template by name from the prompts directory."""
    prompt_path = PROMPT_DIR / f"{name}.json"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt config '{name}' not found at {prompt_path}")

    with prompt_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


__all__ = ["load_prompt_config"]
