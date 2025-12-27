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
            prompts_dir: Optional directory for loading prompts from files
        """
        self.prompts_dir = Path(prompts_dir) if prompts_dir else None
        self.templates: Dict[PromptType, PromptTemplate] = {}

        # Initialize default prompts
        self._initialize_default_prompts()

    def _initialize_default_prompts(self):
        """Initialize default prompt templates."""

        # Manager: Think prompt
        self.templates[PromptType.MANAGER_THINK] = PromptTemplate(
            """You are a Manager agent in a sequential recommendation system.

CURRENT SITUATION:
{context}

SCRATCHPAD (previous thoughts and actions):
{scratchpad}

TASK: Analyze the current situation and think about what information you need to make a good sequential recommendation.

Consider:
1. What do we know about the user's sequential behavior?
2. What additional information might be helpful?
3. What should be our next step?

Think step by step about the reasoning process.""",
            required_vars=['context', 'scratchpad']
        )

        # Manager: Act prompt
        self.templates[PromptType.MANAGER_ACT] = PromptTemplate(
            """Based on your thinking, choose the most appropriate action:

AVAILABLE ACTIONS:
- Analyse[user, user_id] - analyze user preferences and sequential patterns
- Analyse[item, item_id] - analyze specific item characteristics
- Reflect[analysis, candidate] - perform reflection with similar user evidence
- Finish[result] - return final recommendation result

CURRENT CONTEXT:
{context}

SCRATCHPAD:
{scratchpad}

Choose ONE action and return it in the exact format shown above.
If you have enough information to make a recommendation, use Finish[result].
Otherwise, choose the most useful analysis or reflection action.""",
            required_vars=['context', 'scratchpad']
        )

        # Reflector: First Round
        self.templates[PromptType.REFLECTOR_FIRST_ROUND] = PromptTemplate(
            """You are a food delivery recommendation system. Analyze the user's order history and predict whether they will purchase the candidate product.

## User's Recent Orders
{order_history}

## Candidate Product
{candidate_product}

## Task
Predict whether the user is likely to purchase this product. Consider:
1. Cuisine preferences (do they order this type of food?)
2. Price sensitivity (is this within their typical spending?)
3. Time patterns (when do they typically order?)
4. Sequential behavior (what do they order after certain cuisines?)

Return JSON: {{"prediction": true/false, "confidence": 0.0-1.0, "reasoning": "explanation"}}""",
            required_vars=['order_history', 'candidate_product']
        )

        # Reflector: Second Round
        self.templates[PromptType.REFLECTOR_SECOND_ROUND] = PromptTemplate(
            """You are a food delivery recommendation system performing a REFINED prediction.

## User's Recent Orders
{order_history}

## Candidate Product
{candidate_product}

## First Round Analysis
- Initial Prediction: {first_prediction}
- Confidence: {first_confidence:.1%}
- Reasoning: {first_reasoning}

## Evidence from Similar Users
{similar_user_evidence}

## Task
Review your initial prediction considering the behavior of similar users.
- If similar users (high similarity score) consistently bought/rejected similar items, weigh that heavily
- If similar user behavior contradicts your initial prediction, reconsider
- Provide your final decision with reasoning

Return JSON: {{"prediction": true/false, "confidence": 0.0-1.0, "reasoning": "explanation including how similar user evidence influenced your decision"}}""",
            required_vars=['order_history', 'candidate_product', 'first_prediction',
                          'first_confidence', 'first_reasoning', 'similar_user_evidence']
        )

        # Analyst: User Analysis
        self.templates[PromptType.ANALYST_USER] = PromptTemplate(
            """Analyze this user's preferences for sequential recommendation:

USER INFORMATION:
{user_info}

RECENT INTERACTION HISTORY:
{user_history}

TASK: Provide insights about:
1. User's preference patterns
2. Sequential behavior (what they tend to buy after what)
3. Recent trends or shifts in preferences
4. Recommendations for next item

{instruction}""",
            required_vars=['user_info', 'user_history', 'instruction']
        )

        # System: Recommendation
        self.templates[PromptType.SYSTEM_RECOMMENDATION] = PromptTemplate(
            """You are a food delivery recommendation assistant. Given a user's recent purchase history and a candidate product, decide whether the user is likely to purchase this product next. You must answer with 'Yes' or 'No' and provide reasoning.""",
            required_vars=[]
        )

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

        self.templates[prompt_type] = PromptTemplate(content)

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
