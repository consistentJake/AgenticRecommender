"""Enhanced logging utilities for the Agentic Recommender system.

This module centralises logging behaviour across the project, providing:
- Config-driven log directory, console/file toggles, colours, and line width
- Structured JSONL logging for downstream analysis
- Colourised console output with configurable agent/request styling
- Special handling for LLM requests/responses including prompt rendering
- Backwards-compatible helpers for component loggers used across the codebase
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "config"


def _load_raw_config() -> Dict[str, Any]:
    """Load the raw JSON config file if present."""
    if not CONFIG_PATH.exists():
        return {}

    try:
        raw_text = CONFIG_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return {}

    if not raw_text:
        return {}

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # If config is invalid JSON we fail gracefully by returning empty.
        return {}


ANSI_COLOR_CODES: Dict[str, str] = {
    "reset": "\033[0m",
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_black": "\033[90m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
}


DEFAULT_AGENT_COLOURS: Dict[str, str] = {
    "Manager": "magenta",
    "Analyst": "blue",
    "Reflector": "yellow",
    "Searcher": "green",
    "Interpreter": "cyan",
    "Orchestrator": "white",
    "System": "white",
}


DEFAULT_GENERAL_COLOURS: Dict[str, str] = {
    "default": "white",
    "prompt_template": "bright_black",
    "prompt_variable": "bright_cyan",
    "llm_request": "bright_blue",
    "llm_response": "bright_green",
    "error": "bright_red",
}


@dataclass
class LoggingSettings:
    """Resolved logging settings derived from configuration."""

    base_log_dir: Path
    console_enabled: bool = True
    file_enabled: bool = True
    console_level: str = "INFO"
    max_line_width: int = 100
    colours: Dict[str, str] = field(default_factory=dict)
    agent_colours: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, raw_config: Dict[str, Any]) -> "LoggingSettings":
        logging_section = raw_config.get("logging", {}) if isinstance(raw_config, dict) else {}

        base_dir_setting = logging_section.get("base_log_dir", "logs")
        if isinstance(base_dir_setting, str) and base_dir_setting.strip():
            candidate = Path(base_dir_setting.strip()).expanduser()
            base_log_dir = candidate if candidate.is_absolute() else (ROOT_DIR / candidate)
        else:
            base_log_dir = ROOT_DIR / "logs"

        console_enabled = bool(logging_section.get("console_enabled", True))
        file_enabled = bool(logging_section.get("file_enabled", True))
        console_level = str(logging_section.get("console_level", "INFO")).upper()

        width_candidate = logging_section.get("max_line_width", 100)
        try:
            max_line_width = int(width_candidate)
        except (TypeError, ValueError):
            max_line_width = 100
        if max_line_width <= 20:
            max_line_width = 20

        colours = DEFAULT_GENERAL_COLOURS.copy()
        user_colour_overrides = logging_section.get("colors") or logging_section.get("colours")
        if isinstance(user_colour_overrides, dict):
            colours.update({k: str(v) for k, v in user_colour_overrides.items()})

        agent_colours = DEFAULT_AGENT_COLOURS.copy()
        user_agent_overrides = logging_section.get("agent_colors") or logging_section.get("agent_colours")
        if isinstance(user_agent_overrides, dict):
            agent_colours.update({k: str(v) for k, v in user_agent_overrides.items()})

        return cls(
            base_log_dir=base_log_dir,
            console_enabled=console_enabled,
            file_enabled=file_enabled,
            console_level=console_level,
            max_line_width=max_line_width,
            colours=colours,
            agent_colours=agent_colours,
        )


RAW_CONFIG = _load_raw_config()
SETTINGS = LoggingSettings.from_config(RAW_CONFIG)
SETTINGS.base_log_dir.mkdir(parents=True, exist_ok=True)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
GENERAL_LOG_FILE = SETTINGS.base_log_dir / f"agentic_recommender_{RUN_TIMESTAMP}.log"
SESSION_JSON_LOG_FILE = SETTINGS.base_log_dir / f"session_{RUN_TIMESTAMP}.jsonl"


# ---------------------------------------------------------------------------
# Component logging helpers (backwards compatibility)
# ---------------------------------------------------------------------------

_COMPONENT_LOGGING_INITIALISED = False


def _ensure_component_logging() -> None:
    """Initialise the shared component logger with console/file handlers."""
    global _COMPONENT_LOGGING_INITIALISED
    if _COMPONENT_LOGGING_INITIALISED:
        return

    base_logger = logging.getLogger("agentic_recommender")
    base_logger.setLevel(getattr(logging, SETTINGS.console_level, logging.INFO))
    base_logger.propagate = False

    if not base_logger.handlers:
        if SETTINGS.file_enabled:
            file_handler = logging.FileHandler(GENERAL_LOG_FILE, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
            )
            base_logger.addHandler(file_handler)

        if SETTINGS.console_enabled:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, SETTINGS.console_level, logging.INFO))
            console_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            base_logger.addHandler(console_handler)

    _COMPONENT_LOGGING_INITIALISED = True


def get_component_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a namespaced logger sharing the global configuration."""
    _ensure_component_logging()
    logger = logging.getLogger(f"agentic_recommender.{name}")
    logger.setLevel(level)
    return logger


def get_general_log_file() -> Path:
    """Expose the plain-text general log file path for the current run."""
    _ensure_component_logging()
    return GENERAL_LOG_FILE


# ---------------------------------------------------------------------------
# Rich console rendering helpers
# ---------------------------------------------------------------------------


class LogRenderer:
    """Render structured log entries with colour and width controls."""

    def __init__(self, settings: LoggingSettings):
        self.settings = settings

    # Public renderers -----------------------------------------------------

    def render_agent_action(self, entry: Dict[str, Any]) -> str:
        agent_label = self._colour_agent(entry.get("agent"))
        icon = self._icon_for_action(entry.get("subtype"))
        duration_ms = entry.get("performance", {}).get("duration_ms")
        duration_suffix = f" ({duration_ms:.1f}ms)" if isinstance(duration_ms, (int, float)) else ""
        message_lines = self._wrap_text(entry.get("message", ""))
        if not message_lines:
            message_lines = [""]

        prefix = f"{agent_label} {icon} "
        rendered_lines = [prefix + message_lines[0] + duration_suffix]
        indent = " " * len(prefix)
        for line in message_lines[1:]:
            rendered_lines.append(indent + line)
        return "\n".join(rendered_lines)

    def render_llm_request(self, entry: Dict[str, Any]) -> str:
        header_colour = self.settings.colours.get("llm_request", "bright_blue")
        header_text = self._colour_text("LLM Request", header_colour)
        agent_label = self._colour_agent(entry.get("agent"))
        provider = entry.get("provider", "unknown")
        model = entry.get("model_name", "")
        stage = entry.get("stage")
        request_id = entry.get("request_id")

        header_parts = [agent_label, header_text, f"→ {provider}"]
        if model:
            header_parts[-1] += f" ({model})"
        if stage:
            header_parts.append(f"stage={stage}")
        if request_id:
            header_parts.append(f"id={request_id}")

        prompt_repr = self._format_prompt(
            entry.get("prompt", ""),
            entry.get("template_variables", {}),
        )
        prompt_block = self._indent_block(prompt_repr)

        sections = [" ".join(part for part in header_parts if part)]
        if prompt_block.strip():
            sections.append("Prompt:\n" + prompt_block)
        return "\n".join(sections)

    def render_llm_response(self, entry: Dict[str, Any]) -> str:
        header_colour = self.settings.colours.get("llm_response", "bright_green")
        header_text = self._colour_text("LLM Response", header_colour)
        agent_label = self._colour_agent(entry.get("agent"))
        duration_ms = entry.get("performance", {}).get("duration_ms")
        duration_suffix = f" in {duration_ms:.1f}ms" if isinstance(duration_ms, (int, float)) else ""
        tokens_used = entry.get("performance", {}).get("tokens_used")
        tokens_suffix = f", tokens={tokens_used}" if isinstance(tokens_used, (int, float)) else ""
        request_id = entry.get("request_id")

        header_parts = [agent_label, header_text + duration_suffix + tokens_suffix]
        if request_id:
            header_parts.append(f"id={request_id}")

        response_lines = self._wrap_text(entry.get("response", ""))
        if not response_lines:
            response_lines = [""]
        response_block = self._indent_block("\n".join(response_lines))

        sections = [" ".join(part for part in header_parts if part)]
        if response_block.strip():
            sections.append("Response:\n" + response_block)
        return "\n".join(sections)

    # Internal helpers -----------------------------------------------------

    def _colour_agent(self, agent_name: Optional[str]) -> str:
        if not agent_name:
            return self._colour_text("Unknown", self.settings.colours.get("default", "white"))
        colour_name = self.settings.agent_colours.get(agent_name, self.settings.colours.get("default", "white"))
        return self._colour_text(f"[{agent_name}]", colour_name)

    @staticmethod
    def _icon_for_action(action_type: Optional[str]) -> str:
        icon_map = {
            "thought": "💭",
            "action": "⚡",
            "observation": "👁️",
            "error": "❌",
            "communication": "🔄",
            "tool_usage": "🛠️",
            "tool_result": "📎",
            "reflection": "🪞",
            "user_analysis": "🔍",
            "item_analysis": "🛍️",
            "performance_summary": "📊",
            "request": "📨",
            "success": "✅",
        }
        return icon_map.get(action_type or "", "ℹ️")

    def _colour_text(self, text: str, colour_name: Optional[str]) -> str:
        if not text:
            return ""
        if not colour_name:
            return text
        code = ANSI_COLOR_CODES.get(colour_name.lower())
        if not code:
            return text
        reset = ANSI_COLOR_CODES["reset"]
        return f"{code}{text}{reset}"

    def _wrap_text(self, text: str) -> List[str]:
        if not text:
            return []
        width = max(self.settings.max_line_width, 20)
        wrapped: List[str] = []
        for segment in text.splitlines() or [""]:
            if not segment:
                wrapped.append("")
                continue
            wrapped.extend(textwrap.wrap(segment, width=width))
        return wrapped

    def _indent_block(self, text: str, indent: int = 2) -> str:
        prefix = " " * indent
        return "\n".join(f"{prefix}{line}" if line else "" for line in text.splitlines())

    def _format_prompt(self, prompt: str, variables: Dict[str, Any]) -> str:
        if not prompt:
            return ""

        base_colour = self.settings.colours.get("prompt_template", "bright_black")
        variable_colour = self.settings.colours.get("prompt_variable", "bright_cyan")
        base_code = ANSI_COLOR_CODES.get(base_colour.lower(), "")
        var_code = ANSI_COLOR_CODES.get(variable_colour.lower(), "")
        reset = ANSI_COLOR_CODES["reset"]

        wrapped_lines = self._wrap_text(prompt)
        if not wrapped_lines:
            wrapped_lines = [prompt]

        # Pre-compute replacement strings sorted by length to avoid partial replacements.
        replacements: List[str] = []
        for value in variables.values():
            if value is None:
                continue
            value_str = str(value)
            if value_str:
                replacements.append(value_str)
        replacements.sort(key=len, reverse=True)

        coloured_lines: List[str] = []
        for line in wrapped_lines:
            coloured_line = line
            for value_str in replacements:
                coloured_value = value_str
                if var_code:
                    coloured_value = f"{var_code}{value_str}{base_code if base_code else reset}"
                coloured_line = coloured_line.replace(value_str, coloured_value)
            if base_code:
                coloured_line = f"{base_code}{coloured_line}{reset}"
            coloured_lines.append(coloured_line)

        return "\n".join(coloured_lines)


# ---------------------------------------------------------------------------
# Agentic logger with structured file + colour console output
# ---------------------------------------------------------------------------


class AgenticLogger:
    """Central logging hub for agent actions, LLM calls, and session metrics."""

    def __init__(self, settings: Optional[LoggingSettings] = None, session_id: Optional[str] = None):
        self.settings = settings or SETTINGS
        self.renderer = LogRenderer(self.settings)
        self.session_start = time.time()
        self.session_id = session_id or f"session_{RUN_TIMESTAMP}"
        self.json_log_path = SESSION_JSON_LOG_FILE
        self.runtime_logger = get_component_logger("logging.session")

        self.metrics: Dict[str, Any] = {
            "think_times": [],
            "action_times": [],
            "agent_call_times": defaultdict(list),
            "total_tokens": 0,
            "session_start": self.session_start,
        }

    # Public API -----------------------------------------------------------

    def log_agent_action(
        self,
        agent_name: str,
        action_type: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        step_number: Optional[int] = None,
        duration_ms: Optional[float] = None,
        tokens_used: Optional[int] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "type": "agent_action",
            "agent": agent_name,
            "subtype": action_type,
            "message": message or "",
            "context": context or {},
            "step_number": step_number,
            "performance": {
                "duration_ms": duration_ms,
                "tokens_used": tokens_used,
            },
        }

        self._write_structured_entry(entry)
        self._emit_console(self.renderer.render_agent_action(entry))
        self._update_metrics_for_action(action_type, agent_name, duration_ms, tokens_used)

    def log_llm_request(
        self,
        provider: str,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        meta = dict(metadata or {})
        request_id = meta.get("request_id") or str(uuid4())
        meta.setdefault("request_id", request_id)

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "type": "llm_request",
            "agent": meta.get("agent"),
            "provider": provider,
            "model_name": meta.get("model_name"),
            "stage": meta.get("stage"),
            "request_id": request_id,
            "prompt": prompt,
            "template_variables": meta.get("template_variables", {}),
            "context": meta,
        }

        self._write_structured_entry(entry)
        self._emit_console(self.renderer.render_llm_request(entry))
        return request_id

    def log_llm_response(
        self,
        provider: str,
        request_id: str,
        response_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        tokens_used: Optional[int] = None,
    ) -> None:
        meta = dict(metadata or {})
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "type": "llm_response",
            "agent": meta.get("agent"),
            "provider": provider,
            "model_name": meta.get("model_name"),
            "stage": meta.get("stage"),
            "request_id": request_id,
            "response": response_text,
            "context": meta,
            "performance": {
                "duration_ms": duration_ms,
                "tokens_used": tokens_used,
            },
        }

        self._write_structured_entry(entry)
        self._emit_console(self.renderer.render_llm_response(entry))

        if isinstance(tokens_used, (int, float)):
            self.metrics["total_tokens"] += int(tokens_used)

    def log_communication(
        self,
        from_agent: str,
        action: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log_agent_action(
            agent_name=from_agent,
            action_type="communication",
            message=f"{action}: {message}",
            context=context,
        )

    def log_performance_summary(self) -> Dict[str, Any]:
        total_time = time.time() - self.metrics["session_start"]
        avg_think = self._average(self.metrics["think_times"])
        avg_action = self._average(self.metrics["action_times"])
        agent_summary = {k: len(v) for k, v in self.metrics["agent_call_times"].items()}

        summary = {
            "total_session_time": total_time,
            "average_think_time": avg_think,
            "average_action_time": avg_action,
            "total_tokens_used": self.metrics["total_tokens"],
            "agent_call_counts": agent_summary,
        }

        self.log_agent_action(
            agent_name="System",
            action_type="performance_summary",
            message=f"Session completed in {total_time:.2f}s, tokens={self.metrics['total_tokens']}",
            context=summary,
        )
        return summary

    def get_performance_summary(self) -> Dict[str, Any]:
        total_time = time.time() - self.metrics["session_start"]
        return {
            "total_session_time": total_time,
            "average_think_time": self._average(self.metrics["think_times"]),
            "average_action_time": self._average(self.metrics["action_times"]),
            "total_tokens_used": self.metrics["total_tokens"],
            "agent_call_counts": {k: len(v) for k, v in self.metrics["agent_call_times"].items()},
        }

    # Internal utilities ---------------------------------------------------

    def _write_structured_entry(self, entry: Dict[str, Any]) -> None:
        if not self.settings.file_enabled:
            return
        try:
            with self.json_log_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(entry) + os.linesep)
        except OSError:
            # Failing to write logs should not interrupt the core workflow.
            pass

    def _emit_console(self, rendered: str) -> None:
        if not rendered:
            return
        self.runtime_logger.info(rendered)

    def _update_metrics_for_action(
        self,
        action_type: str,
        agent_name: str,
        duration_ms: Optional[float],
        tokens_used: Optional[int],
    ) -> None:
        if isinstance(duration_ms, (int, float)):
            if action_type == "thought":
                self.metrics["think_times"].append(duration_ms)
            elif action_type == "action":
                self.metrics["action_times"].append(duration_ms)
            else:
                self.metrics["agent_call_times"][agent_name].append(duration_ms)

        if isinstance(tokens_used, (int, float)):
            self.metrics["total_tokens"] += int(tokens_used)

    @staticmethod
    def _average(values: Iterable[float]) -> float:
        values = list(values)
        if not values:
            return 0.0
        return sum(values) / len(values)


# Global singleton instance -------------------------------------------------

_GLOBAL_LOGGER: Optional[AgenticLogger] = None


def get_logger(log_dir: Optional[str] = None, web_demo: bool = False) -> AgenticLogger:
    """Return the singleton AgenticLogger instance."""

    # Parameters maintained for backwards compatibility; `web_demo` is unused
    # here but retained to avoid breaking existing callers.
    del web_demo

    global _GLOBAL_LOGGER
    if _GLOBAL_LOGGER is None:
        if log_dir:
            custom_settings = LoggingSettings.from_config({
                "logging": {
                    "base_log_dir": log_dir,
                    "console_enabled": SETTINGS.console_enabled,
                    "file_enabled": SETTINGS.file_enabled,
                    "console_level": SETTINGS.console_level,
                    "max_line_width": SETTINGS.max_line_width,
                    "colors": SETTINGS.colours,
                    "agent_colors": SETTINGS.agent_colours,
                }
            })
            custom_settings.base_log_dir.mkdir(parents=True, exist_ok=True)
            _GLOBAL_LOGGER = AgenticLogger(custom_settings)
        else:
            _GLOBAL_LOGGER = AgenticLogger(SETTINGS)
    return _GLOBAL_LOGGER


def log_agent_action(agent_name: str, action_type: str, message: str, **kwargs) -> None:
    """Convenience shim mirroring previous public API."""
    logger = get_logger()
    logger.log_agent_action(agent_name, action_type, message, **kwargs)
