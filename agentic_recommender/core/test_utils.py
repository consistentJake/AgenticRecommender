"""
Testing utilities for agentic recommendation system.

This module provides mock/pseudo implementations for testing:
1. PseudoLLMAgent - Simulates LLM responses based on rules
2. Test data generators
3. Assertion helpers
"""

import json
import re
from typing import Dict, Any, Optional, List
from ..models.llm_provider import MockLLMProvider


class PseudoLLMAgent(MockLLMProvider):
    """
    Pseudo LLM agent for testing with intelligent response generation.

    This mock LLM can:
    1. Parse prompts and understand context
    2. Generate appropriate JSON responses
    3. Simulate realistic agent behavior
    4. Track calls for assertions
    """

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        """
        Initialize pseudo LLM agent.

        Args:
            responses: Optional predefined response mappings
        """
        super().__init__(responses)
        self.calls: List[Dict[str, Any]] = []

    def _generate_mock_response(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate intelligent mock response based on prompt."""

        # Track call
        self.calls.append({
            'prompt': prompt,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'kwargs': kwargs
        })

        # Check predefined responses first
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response

        # Intelligent response generation based on prompt patterns
        prompt_lower = prompt.lower()

        # Manager Think responses
        if 'think step by step' in prompt_lower or 'analyze the current situation' in prompt_lower:
            return self._generate_manager_think_response(prompt)

        # Manager Act responses
        if 'available actions' in prompt_lower and ('analyse' in prompt_lower or 'finish' in prompt_lower):
            return self._generate_manager_act_response(prompt)

        # Reflector First Round responses
        if 'predict whether the user' in prompt_lower and 'order history' in prompt_lower:
            return self._generate_reflector_first_round_response(prompt)

        # Reflector Second Round responses
        if 'refined prediction' in prompt_lower and 'similar users' in prompt_lower:
            return self._generate_reflector_second_round_response(prompt)

        # Analyst responses
        if 'analyze this user' in prompt_lower or 'user information' in prompt_lower:
            return self._generate_analyst_response(prompt)

        # Default fallback
        return "Mock LLM response"

    def _generate_manager_think_response(self, prompt: str) -> str:
        """Generate Manager think response."""
        # Extract context hints from prompt
        if 'pizza' in prompt.lower():
            return "The user shows a preference for pizza. I should analyze their ordering patterns to understand their typical price range and ordering times. Next, I'll request user analysis."
        elif 'user' in prompt.lower():
            return "I need to gather information about the user's preferences and order history before making a recommendation. I should analyze the user data."
        else:
            return "I should first analyze the user's behavior patterns to make an informed recommendation."

    def _generate_manager_act_response(self, prompt: str) -> str:
        """Generate Manager act response."""
        # Extract user_id if present
        user_match = re.search(r'user[_\s]*id["\s:]*(["\w]+)', prompt, re.IGNORECASE)

        if 'finish' in prompt.lower() and ('enough information' in prompt.lower() or 'recommendation' in prompt.lower()):
            return "Finish[Yes]"
        elif user_match:
            user_id = user_match.group(1).strip('"')
            return f"Analyse[user, {user_id}]"
        elif 'reflect' in prompt.lower():
            return "Reflect[analysis, candidate]"
        else:
            return "Analyse[user, test_user]"

    def _generate_reflector_first_round_response(self, prompt: str) -> str:
        """Generate Reflector first round JSON response."""
        # Analyze prompt for cuisine and price patterns
        has_pizza = 'pizza' in prompt.lower()
        has_mexikanskt = 'mexikanskt' in prompt.lower()

        # Extract candidate cuisine if possible
        candidate_match = re.search(r'cuisine[:\s]+(\w+)', prompt, re.IGNORECASE)
        candidate_cuisine = candidate_match.group(1).lower() if candidate_match else None

        # Simple heuristic: predict Yes if user has ordered similar cuisine
        if candidate_cuisine:
            if (candidate_cuisine == 'pizza' and has_pizza) or \
               (candidate_cuisine == 'mexikanskt' and has_mexikanskt):
                prediction = True
                confidence = 0.75
                reasoning = f"User has previously ordered {candidate_cuisine} cuisine, showing preference for this type"
            else:
                prediction = False
                confidence = 0.65
                reasoning = f"User has not shown strong preference for {candidate_cuisine} in recent orders"
        else:
            # Default prediction
            prediction = has_pizza  # Use pizza as default indicator
            confidence = 0.6
            reasoning = "Based on order history patterns"

        return json.dumps({
            "prediction": prediction,
            "confidence": confidence,
            "reasoning": reasoning
        })

    def _generate_reflector_second_round_response(self, prompt: str) -> str:
        """Generate Reflector second round JSON response."""
        # Extract first round prediction
        initial_pred_match = re.search(r'Initial Prediction:\s*(Yes|No)', prompt, re.IGNORECASE)
        initial_pred = initial_pred_match.group(1).lower() == 'yes' if initial_pred_match else True

        # Check similar user evidence
        has_similar_users = 'similar user' in prompt.lower()
        bought_similar = 'bought similar' in prompt.lower()
        did_not_buy = 'did not buy similar' in prompt.lower()

        # Adjust prediction based on similar user evidence
        if has_similar_users:
            if bought_similar and not did_not_buy:
                # Strong evidence for purchase
                prediction = True
                confidence = 0.85
                reasoning = "Similar users consistently purchased similar items, strengthening initial prediction"
            elif did_not_buy and not bought_similar:
                # Strong evidence against purchase
                prediction = False
                confidence = 0.85
                reasoning = "Similar users did not purchase similar items, contradicting initial prediction"
            else:
                # Mixed evidence, keep initial prediction but lower confidence
                prediction = initial_pred
                confidence = 0.65
                reasoning = "Mixed evidence from similar users, maintaining initial prediction with moderate confidence"
        else:
            # No similar user data, keep initial prediction
            prediction = initial_pred
            confidence = 0.55
            reasoning = "No similar user data available, relying on order history analysis"

        return json.dumps({
            "prediction": prediction,
            "confidence": confidence,
            "reasoning": reasoning
        })

    def _generate_analyst_response(self, prompt: str) -> str:
        """Generate Analyst response."""
        # Extract cuisine patterns from prompt
        cuisines = []
        for cuisine in ['pizza', 'burgare', 'mexikanskt', 'kebab', 'sushi', 'indiskt']:
            if cuisine in prompt.lower():
                cuisines.append(cuisine)

        if cuisines:
            return f"User shows preference for {', '.join(cuisines)}. Recent order patterns suggest they order frequently on weekends and prefer mid-price range items."
        else:
            return "User shows varied cuisine preferences with no strong pattern. Recent orders suggest exploration behavior."

    def get_calls(self) -> List[Dict[str, Any]]:
        """Get list of all calls made to the agent."""
        return self.calls.copy()

    def reset_calls(self):
        """Reset call tracking."""
        self.calls = []

    def get_last_call(self) -> Optional[Dict[str, Any]]:
        """Get the last call made to the agent."""
        return self.calls[-1] if self.calls else None


def create_test_order_history(cuisines: List[str], prices: List[float] = None) -> str:
    """
    Generate test order history markdown table.

    Args:
        cuisines: List of cuisines for orders
        prices: Optional list of prices (defaults to 0.5)

    Returns:
        Markdown table string
    """
    if prices is None:
        prices = [0.5] * len(cuisines)

    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hours = list(range(12, 21))  # 12-20

    lines = ["| idx | day | hour | cuisine | price |", "|-----|-----|------|---------|-------|"]

    for i, (cuisine, price) in enumerate(zip(cuisines, prices), 1):
        day = days[i % len(days)]
        hour = hours[i % len(hours)]
        lines.append(f"| {i} | {day} | {hour} | {cuisine} | {price:.2f} |")

    return "\n".join(lines)


def create_test_candidate(name: str, cuisine: str, price: float) -> str:
    """
    Generate test candidate product string.

    Args:
        name: Product name
        cuisine: Cuisine type
        price: Price

    Returns:
        Formatted candidate string
    """
    return f"- {name} (cuisine: {cuisine}, price: ${price:.2f})"


def create_test_similar_user_evidence(
    user_ids: List[str],
    similarities: List[float],
    cuisines_list: List[List[str]],
    bought_similar: List[bool]
) -> str:
    """
    Generate similar user evidence string.

    Args:
        user_ids: List of user IDs
        similarities: List of similarity scores
        cuisines_list: List of cuisine lists for each user
        bought_similar: List of booleans indicating if they bought similar

    Returns:
        Formatted evidence string
    """
    lines = []

    for i, (user_id, sim, cuisines, bought) in enumerate(
        zip(user_ids, similarities, cuisines_list, bought_similar), 1
    ):
        decision = "Bought similar" if bought else "Did not buy similar"
        cuisine_str = ', '.join(cuisines)
        lines.append(f"""**Similar User {i}** (Similarity: {sim:.1%})
- Recent cuisines: {cuisine_str}
- Action for similar products: {decision}""")

    return "\n\n".join(lines)


def assert_json_response(response: str, expected_keys: List[str]):
    """
    Assert that response is valid JSON with expected keys.

    Args:
        response: Response string
        expected_keys: List of expected keys in JSON

    Raises:
        AssertionError: If response is not valid JSON or missing keys
    """
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Response is not valid JSON: {e}")

    for key in expected_keys:
        if key not in data:
            raise AssertionError(f"Response missing key: {key}")


def assert_action_format(action: str):
    """
    Assert that action follows the correct format.

    Args:
        action: Action string

    Raises:
        AssertionError: If action format is invalid
    """
    pattern = r'(Analyse|Reflect|Finish)\[.+\]'
    if not re.match(pattern, action):
        raise AssertionError(f"Invalid action format: {action}")
