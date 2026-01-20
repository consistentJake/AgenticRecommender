"""
Centralized prompt template management.

This module provides a unified system for managing agent prompts that:
1. Stores all prompts in one place
2. Supports variable substitution
3. Enables easy prompt versioning and A/B testing
4. Validates prompts on load
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum


class PromptType(Enum):
    """Types of prompts in the system."""
    # Manager prompts
    MANAGER_THINK = "manager_think"
    MANAGER_ACT = "manager_act"

    # Analyst prompts
    ANALYST_USER = "analyst_user"
    ANALYST_ITEM = "analyst_item"
    ANALYST_SEQUENCE = "analyst_sequence"

    # Reflector prompts
    REFLECTOR_FIRST_ROUND = "reflector_first_round"
    REFLECTOR_SECOND_ROUND = "reflector_second_round"

    # System prompts
    SYSTEM_RECOMMENDATION = "system_recommendation"


# Mapping from PromptType to template file paths (relative to templates directory)
TEMPLATE_FILES: Dict[PromptType, str] = {
    PromptType.MANAGER_THINK: "manager/think.txt",
    PromptType.MANAGER_ACT: "manager/act.txt",
    PromptType.ANALYST_USER: "analyst/user.txt",
    PromptType.REFLECTOR_FIRST_ROUND: "reflector/first_round.txt",
    PromptType.REFLECTOR_SECOND_ROUND: "reflector/second_round.txt",
    PromptType.SYSTEM_RECOMMENDATION: "system/recommendation.txt",
}

# Required variables for each prompt type
REQUIRED_VARS: Dict[PromptType, List[str]] = {
    PromptType.MANAGER_THINK: ['context', 'scratchpad'],
    PromptType.MANAGER_ACT: ['context', 'scratchpad'],
    PromptType.ANALYST_USER: ['user_info', 'user_history', 'instruction'],
    PromptType.REFLECTOR_FIRST_ROUND: ['order_history', 'candidate_product'],
    PromptType.REFLECTOR_SECOND_ROUND: [
        'order_history', 'candidate_product', 'first_prediction',
        'first_confidence', 'first_reasoning', 'similar_user_evidence'
    ],
    PromptType.SYSTEM_RECOMMENDATION: [],
}


class PromptTemplate:
    """
    Template with variable substitution.

    Usage:
        template = PromptTemplate("Hello {name}!")
        result = template.render(name="Alice")
        # "Hello Alice!"
    """

    def __init__(self, template: str, required_vars: List[str] = None):
        """
        Initialize prompt template.

        Args:
            template: Template string with {variable} placeholders
            required_vars: List of required variable names
        """
        self.template = template
        self.required_vars = required_vars or []

        # Extract all variables from template
        self.variables = self._extract_variables()

    def _extract_variables(self) -> List[str]:
        """Extract variable names from template."""
        return re.findall(r'\{(\w+)\}', self.template)

    def render(self, **kwargs) -> str:
        """
        Render template with variables.

        Args:
            **kwargs: Variable values

        Returns:
            Rendered prompt

        Raises:
            ValueError: If required variables are missing
        """
        # Check required variables
        missing = [var for var in self.required_vars if var not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        # Render template
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Template variable not provided: {e}")

    def get_variables(self) -> List[str]:
        """Get all variable names in template."""
        return self.variables.copy()


def _get_templates_dir() -> Path:
    """Get the templates directory path."""
    return Path(__file__).parent / "templates"


class PromptManager:
    """
    Centralized prompt manager.

    Usage:
        # Initialize
        prompt_mgr = PromptManager()

        # Render a prompt
        prompt = prompt_mgr.render(
            PromptType.REFLECTOR_FIRST_ROUND,
            order_history="...",
            candidate_product="..."
        )
    """

    def __init__(self, prompts_dir: Optional[str] = None):
        """
        Initialize prompt manager.

        Args:
            prompts_dir: Optional directory for loading prompts from files.
                         If not provided, uses the default templates directory.
        """
        self.prompts_dir = Path(prompts_dir) if prompts_dir else _get_templates_dir()
        self.templates: Dict[PromptType, PromptTemplate] = {}

        # Load prompts from template files
        self._load_templates()

    def _load_templates(self):
        """Load prompt templates from files."""
        for prompt_type, file_path in TEMPLATE_FILES.items():
            full_path = self.prompts_dir / file_path
            if full_path.exists():
                with full_path.open('r') as f:
                    content = f.read()
                required_vars = REQUIRED_VARS.get(prompt_type, [])
                self.templates[prompt_type] = PromptTemplate(content, required_vars)

    def register_template(self, prompt_type: PromptType, template: PromptTemplate):
        """
        Register a custom template.

        Args:
            prompt_type: Type of prompt
            template: PromptTemplate instance
        """
        self.templates[prompt_type] = template

    def render(self, prompt_type: PromptType, **kwargs) -> str:
        """
        Render a prompt with variables.

        Args:
            prompt_type: Type of prompt to render
            **kwargs: Variable values

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If prompt type not found or variables missing
        """
        if prompt_type not in self.templates:
            raise ValueError(f"Prompt type not found: {prompt_type}")

        template = self.templates[prompt_type]
        return template.render(**kwargs)

    def get_template(self, prompt_type: PromptType) -> PromptTemplate:
        """
        Get template for a prompt type.

        Args:
            prompt_type: Type of prompt

        Returns:
            PromptTemplate instance
        """
        if prompt_type not in self.templates:
            raise ValueError(f"Prompt type not found: {prompt_type}")

        return self.templates[prompt_type]

    def load_from_file(self, prompt_type: PromptType, file_path: str):
        """
        Load prompt template from file.

        Args:
            prompt_type: Type of prompt
            file_path: Path to prompt file
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")

        with path.open('r') as f:
            content = f.read()

        required_vars = REQUIRED_VARS.get(prompt_type, [])
        self.templates[prompt_type] = PromptTemplate(content, required_vars)

    def save_to_file(self, prompt_type: PromptType, file_path: str):
        """
        Save prompt template to file.

        Args:
            prompt_type: Type of prompt
            file_path: Output file path
        """
        if prompt_type not in self.templates:
            raise ValueError(f"Prompt type not found: {prompt_type}")

        template = self.templates[prompt_type]
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('w') as f:
            f.write(template.template)

    def reload_templates(self):
        """Reload all templates from files."""
        self.templates.clear()
        self._load_templates()


# Global singleton
_global_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager(prompts_dir: str = None) -> PromptManager:
    """
    Get global prompt manager.

    Args:
        prompts_dir: Optional directory for loading prompts

    Returns:
        Global PromptManager instance
    """
    global _global_prompt_manager

    if _global_prompt_manager is None:
        _global_prompt_manager = PromptManager(prompts_dir)

    return _global_prompt_manager


def set_prompt_manager(prompt_mgr: PromptManager):
    """Set global prompt manager."""
    global _global_prompt_manager
    _global_prompt_manager = prompt_mgr
