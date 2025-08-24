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
        
        self.reflection_strategy = reflection_strategy
        self.reflections = []
        self.reflections_str = ""
        self.reflection_input = ""
        self.reflection_output = ""
        
        print(f"🪞 Reflector initialized with strategy: {reflection_strategy.value}")
    
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
        
        # Build reflection prompt
        prompt = self._build_reflection_prompt(input_task, scratchpad, first_attempt, ground_truth)
        
        # Generate reflection
        reflection = self.llm_provider.generate(
            prompt, 
            temperature=0.7,
            json_mode=True  # Encourage JSON output for structured feedback
        )
        
        # Try to parse as JSON for structured feedback
        try:
            reflection_data = json.loads(reflection)
            formatted_reflection = f"""Reflection Analysis:
Correctness: {reflection_data.get('correctness', 'unknown')}
Reason: {reflection_data.get('reason', 'No reason provided')}
Improvement: {reflection_data.get('improvement', 'No improvement suggested')}"""
        except json.JSONDecodeError:
            # Fallback to text format
            formatted_reflection = f"Reflection: {reflection}"
            reflection_data = {'correctness': False, 'reason': reflection}
        
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
                'reflection_data': reflection_data
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
        
        Reference: MACRec_Analysis.md:153-163
        """
        base_prompt = """You are an advanced reasoning agent that can improve based on self reflection.
You will be given a previous reasoning trial in which you were given a task to complete.

Firstly, you should determine if the given answer is correct.
Then, provide reasons for your judgement.
Possible reasons for failure may be:
- Guessing the wrong answer
- Using wrong format for action
- Insufficient analysis of user sequential patterns
- Ignoring important contextual information

In a few sentences, discover the potential problems in your previous reasoning trial
and devise a new, concise, high level plan that aims to mitigate the same failure.

Return JSON format: {"correctness": boolean, "reason": "detailed explanation", "improvement": "suggested improvement plan"}
"""
        
        task_info = f"""
ORIGINAL TASK:
{input_task}

PREVIOUS ATTEMPT (scratchpad):
{scratchpad}
"""
        
        if first_attempt is not None:
            task_info += f"""
FIRST PREDICTION:
{first_attempt}
"""
        
        if ground_truth is not None:
            task_info += f"""
CORRECT ANSWER:
{ground_truth}
"""
        
        return base_prompt + task_info
    
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