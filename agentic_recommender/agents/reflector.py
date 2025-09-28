"""
Reflector agent implementation.
Improves recommendations through iterative refinement.

Reference: previousWorks/MACRec/macrec/agents/reflector.py
"""

import json
import time
from typing import Dict, Any, Optional, List, Tuple

from .base import Agent, AgentType, ReflectionStrategy
from ..models.llm_provider import LLMProvider
from ..utils.logging import get_component_logger
from ..configs.prompt_loader import load_prompt_config


class Reflector(Agent):
    """
    Reflector agent for iterative improvement of recommendations.
    
    Responsibilities:
    - Self-assessment of prediction quality
    - Generate improvement insights
    - Support multiple reflection strategies
    - Create reflection-based training data
    
    Reference: MACRec_Analysis.md:114-176
    """
    
    def __init__(self, llm_provider: LLMProvider, 
                 reflection_strategy: ReflectionStrategy = ReflectionStrategy.REFLEXION,
                 config: Dict[str, Any] = None):
        super().__init__(AgentType.REFLECTOR, llm_provider, config)
        self.component_logger = get_component_logger("agents.reflector")
        
        self.reflection_strategy = reflection_strategy
        self.reflections = []
        self.reflections_str = ""
        self.reflection_input = ""
        self.reflection_output = ""
        
        self.component_logger.info(
            "🪞 Reflector initialized with strategy: %s",
            reflection_strategy.value,
        )

        self.prompt_config = load_prompt_config("reflector")
    
    def forward(self, input_task: str = None, scratchpad: str = None, 
               first_attempt: Any = None, ground_truth: Any = None, **kwargs) -> str:
        """
        Main reflection method.
        
        Args:
            input_task: Original task description
            scratchpad: History of thoughts and actions
            first_attempt: Initial recommendation attempt
            ground_truth: Correct answer (if available)
            **kwargs: Additional context
            
        Returns:
            Reflection insights
        """
        # Store inputs for training data generation
        self.reflection_input = input_task or ""
        
        if self.reflection_strategy == ReflectionStrategy.NONE:
            return "No reflection enabled"
        
        elif self.reflection_strategy == ReflectionStrategy.LAST_ATTEMPT:
            return self._reflect_last_attempt(input_task, scratchpad)
        
        elif self.reflection_strategy == ReflectionStrategy.REFLEXION:
            return self._reflect_with_llm(input_task, scratchpad, first_attempt, ground_truth)
        
        elif self.reflection_strategy == ReflectionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            return self._reflect_combined(input_task, scratchpad, first_attempt, ground_truth)
        
        else:
            return "Unknown reflection strategy"
    
    def _reflect_last_attempt(self, input_task: str, scratchpad: str) -> str:
        """
        Simply store the last attempt as context.
        
        Reference: MACRec_Analysis.md:130-134
        """
        self.reflections = [scratchpad]
        
        reflection = f"""Previous attempt context:
Task: {input_task}
Attempt: {scratchpad}

Use this context to improve your next attempt."""
        
        self.reflections_str = reflection
        self.reflection_output = reflection
        
        return reflection
    
    def _reflect_with_llm(self, input_task: str, scratchpad: str, 
                         first_attempt: Any = None, ground_truth: Any = None) -> str:
        """
        Use LLM to analyze and generate reflection.
        
        Reference: MACRec_Analysis.md:136-140
        """
        start_time = time.time()
        
        attempt = 0
        max_retries = 2
        reflection_data: Dict[str, Any] = {}
        formatted_reflection = ""
        prompt = ""
        reflection = ""

        while attempt <= max_retries:
            prompt = self._build_reflection_prompt(input_task, scratchpad, first_attempt, ground_truth)

            reflection = self.llm_provider.generate(
                prompt,
                temperature=0.4,
                max_tokens=256,
                json_mode=True,
            )

            try:
                reflection_data = json.loads(reflection)
            except json.JSONDecodeError:
                reflection_data = {}

            if self._validate_reflection_schema(reflection_data):
                formatted_reflection = self._format_reflection_summary(reflection_data)
                break

            attempt += 1
            if attempt > max_retries:
                formatted_reflection = f"Reflection (unstructured): {reflection}"
                reflection_data = {'correctness': 'unknown', 'reason': reflection, 'improvement': 'Parsing failed', 'next_action': 'Retry with adjusted prompt'}
                break

            retry_prompt = self.prompt_config.get('retry_message', 'Invalid reflection output. Try again with the required keys.')
            scratchpad = f"{scratchpad}\nNOTE: {retry_prompt}"
        
        # Store reflection
        self.reflections.append(reflection)
        self.reflections_str = formatted_reflection
        self.reflection_output = reflection
        
        # Log reflection
        duration = time.time() - start_time
        self.logger.log_agent_action(
            agent_name=self.agent_type.value,
            action_type="reflection",
            message=f"Generated reflection: {reflection_data.get('reason', 'See full reflection')}",
            context={
                'input_task': input_task,
                'first_attempt': first_attempt,
                'reflection_data': reflection_data,
                'prompt': prompt,
                'raw_response': self.reflection_output,
            },
            duration_ms=duration * 1000
        )
        
        return formatted_reflection
    
    def _reflect_combined(self, input_task: str, scratchpad: str,
                         first_attempt: Any = None, ground_truth: Any = None) -> str:
        """
        Combine last attempt storage with LLM reflection.

        Reference: MACRec_Analysis.md:142-149
        """
        # First, store last attempt
        last_attempt_reflection = self._reflect_last_attempt(input_task, scratchpad)
        
        # Then, generate LLM reflection
        llm_reflection = self._reflect_with_llm(input_task, scratchpad, first_attempt, ground_truth)
        
        # Combine both
        combined_reflection = f"""{last_attempt_reflection}

{llm_reflection}"""
        
        self.reflections_str = combined_reflection
        return combined_reflection

    def _build_reflection_prompt(self, input_task: str, scratchpad: str,
                               first_attempt: Any = None, ground_truth: Any = None) -> str:
        """
        Build prompt for LLM-based reflection.
        
        Template reference: previousWorks/MACRec/config/prompts/agent_prompt/reflector.json
        Uses MACRec reflect_prompt_json template for structured self-reflection.
        Reference: MACRec_Analysis.md:153-163
        """
        template = self.prompt_config.get('reflection_template', '')

        conversation_table = self._format_conversation_table(scratchpad)
        task_summary = input_task or "Sequential recommendation"
        formatted = template.format(
            task_summary=task_summary,
            conversation_table=conversation_table,
            first_attempt=first_attempt or "unknown",
            ground_truth=ground_truth or "unknown",
        )

        return formatted

    def _format_conversation_table(self, scratchpad: str) -> str:
        if not scratchpad:
            return "No conversation captured."

        rows = []
        for line in scratchpad.split('\n'):
            if not line.strip():
                continue
            rows.append(f"- {line.strip()}")
        return '\n'.join(rows)

    def _validate_reflection_schema(self, data: Dict[str, Any]) -> bool:
        required_keys = {"correctness", "reason", "improvement", "next_action"}
        return bool(data) and required_keys.issubset(set(data.keys()))

    def _format_reflection_summary(self, data: Dict[str, Any]) -> str:
        return (
            "Reflection Analysis:\n"
            f"Correctness: {data.get('correctness', 'unknown')}\n"
            f"Reason: {data.get('reason', 'No reason provided')}\n"
            f"Improvement: {data.get('improvement', 'No improvement suggested')}\n"
            f"Next action: {data.get('next_action', 'Not specified')}"
        )
    
    def reflect_and_retry(self, task_context: Dict[str, Any], 
                         first_attempt: Any) -> Tuple[str, Dict[str, Any]]:
        """
        Complete reflection cycle: analyze attempt and provide improvement guidance.
        
        Args:
            task_context: Original task context
            first_attempt: First recommendation attempt
            
        Returns:
            Tuple of (improvement_guidance, reflection_insights)
        """
        # Generate reflection
        reflection = self.forward(
            input_task=task_context.get('task', ''),
            scratchpad=task_context.get('scratchpad', ''),
            first_attempt=first_attempt,
            ground_truth=task_context.get('ground_truth')
        )
        
        # Extract insights
        try:
            insights = json.loads(self.reflection_output)
        except json.JSONDecodeError:
            insights = {
                'correctness': False,
                'reason': self.reflection_output,
                'improvement': 'Try a different approach'
            }
        
        # Create improvement guidance
        guidance = f"""Previous attempt analysis:
{insights.get('reason', 'Unknown issue')}

Improvement strategy:
{insights.get('improvement', 'No specific improvement suggested')}

Now try again with this insight in mind."""
        
        return guidance, insights
    
    def set_strategy(self, strategy: ReflectionStrategy):
        """Change reflection strategy"""
        self.reflection_strategy = strategy
        self.logger.log_agent_action(
            agent_name=self.agent_type.value,
            action_type="configuration",
            message=f"Reflection strategy changed to: {strategy.value}"
        )
    
    def get_reflection_history(self) -> List[str]:
        """Get history of all reflections"""
        return self.reflections.copy()
    
    def clear_reflections(self):
        """Clear reflection history"""
        self.reflections = []
        self.reflections_str = ""
        self.reflection_input = ""
        self.reflection_output = ""
