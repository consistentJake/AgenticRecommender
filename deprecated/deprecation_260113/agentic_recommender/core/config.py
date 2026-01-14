"""
Centralized configuration management for agentic recommendation system.

This module provides a unified configuration system that:
1. Loads config from YAML/JSON files
2. Provides type-safe access to configuration values
3. Supports environment variable overrides
4. Validates configuration on load
"""

import os
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class SwingConfig:
    """Configuration for Swing similarity."""
    alpha1: float = 5.0
    alpha2: float = 1.0
    beta: float = 0.3
    similarity_threshold: float = 0.1
    top_k: int = 5


@dataclass
class ReflectorConfig:
    """Configuration for Enhanced Reflector."""
    enable_two_stage: bool = True
    first_round_temperature: float = 0.3
    second_round_temperature: float = 0.3
    min_similar_users: int = 1
    confidence_threshold: float = 0.5


@dataclass
class ManagerConfig:
    """Configuration for Manager agent."""
    thought_temperature: float = 0.8
    action_temperature: float = 0.3
    max_steps: int = 10
    thought_max_tokens: int = 256
    action_max_tokens: int = 128


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider_type: str = "gemini"  # gemini, claude, openrouter, mock
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    fallback_enabled: bool = True
    retry_count: int = 2
    timeout: int = 30


@dataclass
class AgentConfig:
    """Master configuration for agent system."""
    # LLM settings
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Agent-specific settings
    manager: ManagerConfig = field(default_factory=ManagerConfig)
    reflector: ReflectorConfig = field(default_factory=ReflectorConfig)

    # Similarity settings
    swing: SwingConfig = field(default_factory=SwingConfig)

    # Paths
    data_dir: str = "datasets"
    prompts_dir: str = "agentic_recommender/prompts"
    cache_dir: str = ".cache/agents"

    # Logging
    log_level: str = "INFO"
    enable_detailed_logging: bool = True


class ConfigManager:
    """
    Centralized configuration manager.

    Usage:
        # Load from file
        config_mgr = ConfigManager.from_file("configs/agent_config.yaml")

        # Access configuration
        swing_config = config_mgr.get_swing_config()
        llm_config = config_mgr.get_llm_config()

        # Get specific values
        threshold = config_mgr.get("swing.similarity_threshold", default=0.1)
    """

    def __init__(self, config: AgentConfig = None):
        """
        Initialize configuration manager.

        Args:
            config: AgentConfig instance. If None, uses defaults.
        """
        self.config = config or AgentConfig()
        self._apply_env_overrides()

    @classmethod
    def from_file(cls, config_path: str) -> 'ConfigManager':
        """
        Load configuration from YAML or JSON file.

        Args:
            config_path: Path to configuration file

        Returns:
            ConfigManager instance
        """
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load file
        with path.open('r') as f:
            if path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")

        # Build config from dict
        config = cls._dict_to_config(data)
        return cls(config)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigManager':
        """
        Create ConfigManager from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            ConfigManager instance
        """
        config = cls._dict_to_config(data)
        return cls(config)

    @staticmethod
    def _dict_to_config(data: Dict[str, Any]) -> AgentConfig:
        """Convert dictionary to AgentConfig."""
        config = AgentConfig()

        # LLM config
        if 'llm' in data:
            llm_data = data['llm']
            config.llm = LLMConfig(
                provider_type=llm_data.get('provider_type', config.llm.provider_type),
                model_name=llm_data.get('model_name'),
                api_key=llm_data.get('api_key'),
                fallback_enabled=llm_data.get('fallback_enabled', config.llm.fallback_enabled),
                retry_count=llm_data.get('retry_count', config.llm.retry_count),
                timeout=llm_data.get('timeout', config.llm.timeout)
            )

        # Manager config
        if 'manager' in data:
            mgr_data = data['manager']
            config.manager = ManagerConfig(
                thought_temperature=mgr_data.get('thought_temperature', config.manager.thought_temperature),
                action_temperature=mgr_data.get('action_temperature', config.manager.action_temperature),
                max_steps=mgr_data.get('max_steps', config.manager.max_steps),
                thought_max_tokens=mgr_data.get('thought_max_tokens', config.manager.thought_max_tokens),
                action_max_tokens=mgr_data.get('action_max_tokens', config.manager.action_max_tokens)
            )

        # Reflector config
        if 'reflector' in data:
            ref_data = data['reflector']
            config.reflector = ReflectorConfig(
                enable_two_stage=ref_data.get('enable_two_stage', config.reflector.enable_two_stage),
                first_round_temperature=ref_data.get('first_round_temperature', config.reflector.first_round_temperature),
                second_round_temperature=ref_data.get('second_round_temperature', config.reflector.second_round_temperature),
                min_similar_users=ref_data.get('min_similar_users', config.reflector.min_similar_users),
                confidence_threshold=ref_data.get('confidence_threshold', config.reflector.confidence_threshold)
            )

        # Swing config
        if 'swing' in data:
            swing_data = data['swing']
            config.swing = SwingConfig(
                alpha1=swing_data.get('alpha1', config.swing.alpha1),
                alpha2=swing_data.get('alpha2', config.swing.alpha2),
                beta=swing_data.get('beta', config.swing.beta),
                similarity_threshold=swing_data.get('similarity_threshold', config.swing.similarity_threshold),
                top_k=swing_data.get('top_k', config.swing.top_k)
            )

        # Paths
        if 'data_dir' in data:
            config.data_dir = data['data_dir']
        if 'prompts_dir' in data:
            config.prompts_dir = data['prompts_dir']
        if 'cache_dir' in data:
            config.cache_dir = data['cache_dir']

        # Logging
        if 'log_level' in data:
            config.log_level = data['log_level']
        if 'enable_detailed_logging' in data:
            config.enable_detailed_logging = data['enable_detailed_logging']

        return config

    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # LLM provider override
        if 'LLM_PROVIDER' in os.environ:
            self.config.llm.provider_type = os.environ['LLM_PROVIDER']

        # API key override
        if 'LLM_API_KEY' in os.environ:
            self.config.llm.api_key = os.environ['LLM_API_KEY']

        # Log level override
        if 'LOG_LEVEL' in os.environ:
            self.config.log_level = os.environ['LOG_LEVEL']

    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        return self.config.llm

    def get_manager_config(self) -> ManagerConfig:
        """Get Manager configuration."""
        return self.config.manager

    def get_reflector_config(self) -> ReflectorConfig:
        """Get Reflector configuration."""
        return self.config.reflector

    def get_swing_config(self) -> SwingConfig:
        """Get Swing similarity configuration."""
        return self.config.swing

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., "swing.alpha1")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        parts = key.split('.')
        value = self.config

        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return default

        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'llm': {
                'provider_type': self.config.llm.provider_type,
                'model_name': self.config.llm.model_name,
                'fallback_enabled': self.config.llm.fallback_enabled,
                'retry_count': self.config.llm.retry_count,
                'timeout': self.config.llm.timeout
            },
            'manager': {
                'thought_temperature': self.config.manager.thought_temperature,
                'action_temperature': self.config.manager.action_temperature,
                'max_steps': self.config.manager.max_steps,
                'thought_max_tokens': self.config.manager.thought_max_tokens,
                'action_max_tokens': self.config.manager.action_max_tokens
            },
            'reflector': {
                'enable_two_stage': self.config.reflector.enable_two_stage,
                'first_round_temperature': self.config.reflector.first_round_temperature,
                'second_round_temperature': self.config.reflector.second_round_temperature,
                'min_similar_users': self.config.reflector.min_similar_users,
                'confidence_threshold': self.config.reflector.confidence_threshold
            },
            'swing': {
                'alpha1': self.config.swing.alpha1,
                'alpha2': self.config.swing.alpha2,
                'beta': self.config.swing.beta,
                'similarity_threshold': self.config.swing.similarity_threshold,
                'top_k': self.config.swing.top_k
            },
            'data_dir': self.config.data_dir,
            'prompts_dir': self.config.prompts_dir,
            'cache_dir': self.config.cache_dir,
            'log_level': self.config.log_level,
            'enable_detailed_logging': self.config.enable_detailed_logging
        }

    def save(self, path: str):
        """
        Save configuration to file.

        Args:
            path: Output file path (YAML or JSON)
        """
        output_path = Path(path)
        data = self.to_dict()

        with output_path.open('w') as f:
            if output_path.suffix in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            elif output_path.suffix == '.json':
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {output_path.suffix}")


# Global singleton
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: str = None) -> ConfigManager:
    """
    Get global configuration manager.

    Args:
        config_path: Optional path to config file

    Returns:
        Global ConfigManager instance
    """
    global _global_config_manager

    if _global_config_manager is None:
        if config_path and Path(config_path).exists():
            _global_config_manager = ConfigManager.from_file(config_path)
        else:
            _global_config_manager = ConfigManager()

    return _global_config_manager


def set_config_manager(config_mgr: ConfigManager):
    """Set global configuration manager."""
    global _global_config_manager
    _global_config_manager = config_mgr
