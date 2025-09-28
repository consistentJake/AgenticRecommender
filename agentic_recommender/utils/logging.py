"""Logging utilities for the agentic recommendation system."""

import json
import logging
import os
import time
import textwrap
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "config"


def _resolve_base_log_dir() -> Path:
    """Resolve base log directory, optionally from configs/config."""
    default_dir = ROOT_DIR / "logs"

    try:
        raw_config = CONFIG_PATH.read_text().strip()
    except OSError:
        return default_dir

    if not raw_config:
        return default_dir

    try:
        parsed = json.loads(raw_config)
    except json.JSONDecodeError:
        return default_dir

    logging_section = None
    if isinstance(parsed, dict):
        candidate = parsed.get("logging")
        logging_section = candidate if isinstance(candidate, dict) else parsed

    if not logging_section:
        return default_dir

    base_log_dir = logging_section.get("base_log_dir") or logging_section.get("log_dir")
    if not isinstance(base_log_dir, str) or not base_log_dir.strip():
        return default_dir

    path = Path(base_log_dir.strip()).expanduser()
    return path if path.is_absolute() else (ROOT_DIR / path)


BASE_LOG_DIR = _resolve_base_log_dir()
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
GENERAL_LOG_FILE = BASE_LOG_DIR / f"agentic_recommender_{RUN_TIMESTAMP}.log"
CONSOLE_LOG_ENABLED = os.getenv("AGENTIC_CONSOLE_LOG", "1").lower() not in {"0", "false", "no"}
_COMPONENT_LOGGING_INITIALISED = False


def _ensure_component_logging() -> None:
    """Initialise shared logging handlers for the project."""
    global _COMPONENT_LOGGING_INITIALISED
    if _COMPONENT_LOGGING_INITIALISED:
        return

    BASE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    base_logger = logging.getLogger("agentic_recommender")
    base_logger.setLevel(logging.INFO)
    base_logger.propagate = False

    file_handler = logging.FileHandler(GENERAL_LOG_FILE)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    base_logger.addHandler(file_handler)

    if CONSOLE_LOG_ENABLED:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        base_logger.addHandler(console_handler)

    _COMPONENT_LOGGING_INITIALISED = True


def get_component_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that shares the global component logging configuration."""
    _ensure_component_logging()
    logger = logging.getLogger(f"agentic_recommender.{name}")
    logger.setLevel(level)
    return logger


def get_general_log_file() -> Path:
    """Expose the shared log file path for the current run."""
    _ensure_component_logging()
    return GENERAL_LOG_FILE


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class AgenticLogger:
    """
    Comprehensive logging system for multi-agent recommendation system.
    
    Based on MACRec's logging implementation:
    - File logging for debugging
    - Console logging with color coding
    - Structured JSON format
    - Performance metrics tracking
    """
    
    def __init__(self, log_dir: Optional[str] = None, web_demo: bool = False):
        self.log_dir = Path(log_dir) if log_dir else BASE_LOG_DIR
        self.web_demo = web_demo
        self.web_log = []
        
        # Performance tracking
        self.metrics = {
            'think_times': [],
            'action_times': [],
            'agent_call_times': defaultdict(list),
            'total_tokens': 0,
            'session_start': time.time()
        }

        # Setup logging handlers
        self._setup_file_logging()
        self._setup_console_logging()

        # Pretty-print configuration for context payloads
        self.wrap_width = int(os.getenv("AGENTIC_LOG_WRAP", "120"))
    
    def _setup_file_logging(self):
        """Setup structured file logging"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.file_logger = logging.getLogger('agentic_rec_file')
        self.file_logger.setLevel(logging.DEBUG)
        
        # File handler
        log_path = self.log_dir / f"session_{int(time.time())}.jsonl"
        handler = logging.FileHandler(log_path)
        handler.setLevel(logging.DEBUG)
        
        # JSON formatter
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.file_logger.addHandler(handler)
    
    def _setup_console_logging(self):
        """Setup console logging with color coding"""
        self.console_logger = logging.getLogger('agentic_rec_console')
        self.console_logger.setLevel(logging.INFO)
        
        # Console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        
        # Simple formatter for console
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.console_logger.addHandler(handler)
    
    def log_agent_action(self, agent_name: str, action_type: str, 
                        message: str, context: Dict[str, Any] = None,
                        step_number: int = None, duration_ms: float = None,
                        tokens_used: int = None):
        """
        Log an agent action with full context and performance metrics.
        
        Args:
            agent_name: Name of the agent (Manager, Analyst, Reflector, etc.)
            action_type: Type of action (thought, action, observation, error)
            message: Human-readable message
            context: Additional context data
            step_number: Current step in the workflow
            duration_ms: Time taken for this action
            tokens_used: Number of tokens consumed
        """
        
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': f"rec_session_{int(self.metrics['session_start'])}",
            'agent': agent_name,
            'type': action_type,
            'step_number': step_number,
            'content': {
                'message': message,
                'context': context or {}
            },
            'performance': {
                'duration_ms': duration_ms,
                'tokens_used': tokens_used
            }
        }
        
        # Log to file as JSON
        self.file_logger.info(json.dumps(log_entry, ensure_ascii=False))

        # Log to console with formatting
        self._console_format(log_entry)

        # Mirror to general component logger so plaintext log captures full context
        try:
            component_logger = get_component_logger("agent_actions")
            component_logger.info(self._format_plain_log_entry(log_entry))
        except Exception as exc:  # pragma: no cover - logging fallback
            self.console_logger.warning(
                f"AgenticLogger mirror logging failed: {exc}"
            )
        
        # Update performance metrics
        if duration_ms:
            # Only track specific action types we care about
            if action_type in ['thought', 'action']:
                key = f'{action_type}_times'
            else:
                key = 'agent_call_times'
            
            if key in self.metrics:
                if isinstance(self.metrics[key], list):
                    self.metrics[key].append(duration_ms)
                else:
                    self.metrics[key][agent_name].append(duration_ms)
        
        if tokens_used:
            self.metrics['total_tokens'] += tokens_used

    def _format_plain_log_entry(self, log_entry: Dict[str, Any]) -> str:
        """Render a multiline plaintext representation for general logs."""
        timestamp = log_entry.get('timestamp')
        agent = log_entry.get('agent')
        action_type = log_entry.get('type')
        content = log_entry.get('content', {})
        message = content.get('message', '')
        ctx = content.get('context') or {}
        performance = log_entry.get('performance') or {}

        lines = [f"[{timestamp}] {agent}::{action_type}"]
        lines.append("Message:")
        lines.append(str(message))

        if ctx:
            lines.append("Context:")
            lines.append(self._format_context_block(ctx, indent="  "))

        perf_bits = []
        if performance.get('duration_ms') is not None:
            perf_bits.append(f"duration_ms={performance['duration_ms']:.2f}")
        if performance.get('tokens_used') is not None:
            perf_bits.append(f"tokens_used={performance['tokens_used']}")
        if perf_bits:
            lines.append("Performance: " + ", ".join(perf_bits))

        lines.append("-")
        return "\n".join(lines)

    def _format_context_block(self, value: Any, indent: str = "") -> str:
        """Pretty-format context payloads with controlled line width."""
        if isinstance(value, dict):
            lines: List[str] = []
            for key in sorted(value.keys()):
                formatted = self._format_context_block(value[key], indent=indent + "  ")
                if "\n" in formatted:
                    lines.append(f"{indent}{key}:")
                    lines.append(formatted)
                else:
                    lines.append(f"{indent}{key}: {formatted}")
            return "\n".join(lines) if lines else f"{indent}{{}}"

        if isinstance(value, list):
            if not value:
                return f"{indent}[]"
            lines: List[str] = []
            for idx, item in enumerate(value):
                formatted = self._format_context_block(item, indent=indent + "  ")
                if "\n" in formatted:
                    lines.append(f"{indent}- item_{idx}:")
                    lines.append(formatted)
                else:
                    lines.append(f"{indent}- {formatted}")
            return "\n".join(lines)

        if isinstance(value, (tuple, set)):
            return self._format_context_block(list(value), indent=indent)

        if isinstance(value, str):
            wrapped = self._wrap_text(value)
            if "\n" in wrapped:
                return textwrap.indent(wrapped, indent)
            return f"{indent}{wrapped}" if indent else wrapped

        return f"{indent}{value}" if indent else str(value)

    def _wrap_text(self, text: str) -> str:
        """Soft-wrap text to configured width while preserving blank lines."""
        if not text:
            return text

        width = max(self.wrap_width, 40)
        wrapped_lines: List[str] = []
        for raw_line in str(text).splitlines():
            if not raw_line.strip():
                wrapped_lines.append("")
                continue

            wrapped_lines.extend(
                textwrap.wrap(
                    raw_line,
                    width=width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                ) or [raw_line]
            )

        return "\n".join(wrapped_lines)
    
    def _console_format(self, entry: Dict[str, Any]):
        """Format log entry for console with color coding"""
        # Color mapping based on MACRec
        colors = {
            'Manager': '\033[95m',     # Purple
            'Analyst': '\033[94m',     # Blue
            'Reflector': '\033[93m',   # Yellow
            'Searcher': '\033[92m',    # Green
            'System': '\033[96m'       # Cyan
        }
        
        color = colors.get(entry['agent'], '\033[0m')
        reset = '\033[0m'
        
        # Format based on action type
        if entry['type'] == 'thought':
            prefix = "💭"
        elif entry['type'] == 'action':
            prefix = "⚡"
        elif entry['type'] == 'observation':
            prefix = "👁️"
        elif entry['type'] == 'error':
            prefix = "❌"
        else:
            prefix = "ℹ️"
        
        # Create formatted message
        step_info = f"Step {entry['step_number']}: " if entry['step_number'] else ""
        perf_info = ""
        if entry['performance']['duration_ms']:
            perf_info = f" ({entry['performance']['duration_ms']:.1f}ms)"
        
        formatted_msg = f"{color}[{entry['agent']}]{reset} {prefix} {step_info}{entry['content']['message']}{perf_info}"
        
        # Add indentation for non-manager agents (following MACRec style)
        if entry['agent'].lower() not in ['manager', 'system']:
            formatted_msg = f"  {formatted_msg}"
        
        self.console_logger.info(formatted_msg)
    
    def log_communication(self, from_agent: str, to_agent: str, 
                         action: str, argument: Any, response: str = None):
        """Log inter-agent communication"""
        if response is None:
            # Calling another agent
            self.log_agent_action(
                agent_name=from_agent,
                action_type="communication",
                message=f"Calling {to_agent} with {action}[{argument}]",
                context={'target_agent': to_agent, 'action': action, 'argument': argument}
            )
        else:
            # Receiving response
            self.log_agent_action(
                agent_name=from_agent,
                action_type="communication",
                message=f"Response from {to_agent}: {response[:100]}...",
                context={'source_agent': to_agent, 'response_length': len(response)}
            )
    
    def log_performance_summary(self):
        """Log performance summary at end of session"""
        total_time = time.time() - self.metrics['session_start']
        avg_think_time = sum(self.metrics['think_times']) / len(self.metrics['think_times']) if self.metrics['think_times'] else 0
        avg_action_time = sum(self.metrics['action_times']) / len(self.metrics['action_times']) if self.metrics['action_times'] else 0
        
        summary = {
            'total_session_time': total_time,
            'average_think_time': avg_think_time,
            'average_action_time': avg_action_time,
            'total_tokens_used': self.metrics['total_tokens'],
            'agent_call_counts': {k: len(v) for k, v in self.metrics['agent_call_times'].items()}
        }
        
        self.log_agent_action(
            agent_name="System",
            action_type="performance_summary",
            message=f"Session completed in {total_time:.2f}s, {self.metrics['total_tokens']} tokens",
            context=summary
        )
        
        return summary
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        total_time = time.time() - self.metrics['session_start']
        avg_think_time = sum(self.metrics['think_times']) / len(self.metrics['think_times']) if self.metrics['think_times'] else 0
        avg_action_time = sum(self.metrics['action_times']) / len(self.metrics['action_times']) if self.metrics['action_times'] else 0
        
        return {
            'total_session_time': total_time,
            'average_think_time': avg_think_time,
            'average_action_time': avg_action_time,
            'total_tokens_used': self.metrics['total_tokens'],
            'agent_call_counts': {k: len(v) for k, v in self.metrics['agent_call_times'].items()}
        }


# Global logger instance
_global_logger = None


def get_logger(log_dir: Optional[str] = None, web_demo: bool = False) -> AgenticLogger:
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = AgenticLogger(log_dir, web_demo)
    return _global_logger

def log_agent_action(agent_name: str, action_type: str, message: str, **kwargs):
    """Convenience function for logging"""
    logger = get_logger()
    logger.log_agent_action(agent_name, action_type, message, **kwargs)
