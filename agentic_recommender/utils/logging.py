"""
Logging system for agentic recommendation system.
Based on MACRec's logging implementation.
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any
from collections import defaultdict
from enum import Enum


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
    
    def __init__(self, log_dir: str = "logs", web_demo: bool = False):
        self.log_dir = log_dir
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
    
    def _setup_file_logging(self):
        """Setup structured file logging"""
        import os
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create logger
        self.file_logger = logging.getLogger('agentic_rec_file')
        self.file_logger.setLevel(logging.DEBUG)
        
        # File handler
        handler = logging.FileHandler(f"{self.log_dir}/session_{int(time.time())}.jsonl")
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
        self.file_logger.info(json.dumps(log_entry))
        
        # Log to console with formatting
        self._console_format(log_entry)
        
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


# Global logger instance
_global_logger = None

def get_logger(log_dir: str = "logs", web_demo: bool = False) -> AgenticLogger:
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = AgenticLogger(log_dir, web_demo)
    return _global_logger

def log_agent_action(agent_name: str, action_type: str, message: str, **kwargs):
    """Convenience function for logging"""
    logger = get_logger()
    logger.log_agent_action(agent_name, action_type, message, **kwargs)