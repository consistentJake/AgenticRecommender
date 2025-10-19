"""
Manager agent implementation.
Central orchestrator using two-stage LLM architecture.

Reference: previousWorks/MACRec/macrec/agents/manager.py
"""

import time
import json
import re
from typing import Dict, Any, Optional, Tuple

from .base import Agent, AgentType, create_agent_prompt
from ..models.llm_provider import LLMProvider
from ..utils.logging import get_component_logger


class Manager(Agent):
    """
    Manager agent with two-stage LLM architecture (think + act).
    
    Responsibilities:
    - Central orchestration of recommendation process
    - Thinking phase: analyze situation and plan
    - Action phase: execute decisions with structured output
    - Token limit management
    - Agent coordination
    
    Reference: MACRec_Analysis.md:28-52
    """
    
    def __init__(self, thought_llm: LLMProvider, action_llm: LLMProvider, 
                 config: Dict[str, Any] = None):
        super().__init__(AgentType.MANAGER, thought_llm, config)
        self.component_logger = get_component_logger("agents.manager")
        
        # Two separate LLMs for specialization
        self.thought_llm = thought_llm  # Optimized for reasoning
        self.action_llm = action_llm    # Optimized for structured output
        
        # State management
        self.scratchpad = ""
        self.current_task = None
        self.max_steps = config.get('max_steps', 10) if config else 10
        
        self.component_logger.info(
            "🧠 Manager initialized with thought_llm: %s",
            thought_llm.get_model_info()['model_name'],
        )
        self.component_logger.info(
            "⚡ Manager initialized with action_llm: %s",
            action_llm.get_model_info()['model_name'],
        )
    
    def think(self, task_context: Dict[str, Any]) -> str:
        """
        Manager thinking phase - analyze situation and plan next steps.
        
        Args:
            task_context: Current task context including user history, candidates, etc.
            
        Returns:
            Thought process as string
        """
        start_time = time.time()
        
        # Build thinking prompt
        prompt_text, prompt_template, template_vars = self._build_thinking_prompt(task_context)
        model_info = self.thought_llm.get_model_info()
        # LLM_LOG: capture metadata for thought-stage request logging with colour-coded template output
        log_metadata = {
            'agent': self.agent_type.value,
            'stage': 'thought',
            'step_number': self.step_count + 1,
            'template_variables': template_vars,
            'prompt_template': prompt_template,
            'model_name': model_info.get('model_name'),
            'provider': model_info.get('provider'),
        }
        
        # Generate thought using thought LLM
        thought = self.thought_llm.generate(
            prompt_text,
            temperature=0.8,  # Higher temperature for creative reasoning
            max_tokens=256,
            log_metadata=log_metadata,
        )
        
        # Update scratchpad
        self.scratchpad += f"\nThought {self.step_count + 1}: {thought}"
        
        # Log thinking
        duration = time.time() - start_time
        self.logger.log_agent_action(
            agent_name=self.agent_type.value,
            action_type="thought",
            message=thought,
            context=task_context,
            step_number=self.step_count + 1,
            duration_ms=duration * 1000
        )
        
        return thought
    
    def act(self, task_context: Dict[str, Any] = None) -> Tuple[str, Any]:
        """
        Manager action phase - decide and execute structured action.
        
        Args:
            task_context: Current task context
            
        Returns:
            Tuple of (action_type, argument)
        """
        start_time = time.time()
        
        build_context = task_context or {}
        prompt_text, prompt_template, template_vars = self._build_action_prompt(build_context)
        model_info = self.action_llm.get_model_info()
        # LLM_LOG: capture metadata for action-stage request logging with colour-coded template output
        log_metadata = {
            'agent': self.agent_type.value,
            'stage': 'action',
            'step_number': self.step_count + 1,
            'template_variables': template_vars,
            'prompt_template': prompt_template,
            'context': build_context,
            'model_name': model_info.get('model_name'),
            'provider': model_info.get('provider'),
        }
        
        # Generate action using action LLM
        action_text = self.action_llm.generate(
            prompt_text,
            temperature=0.3,  # Lower temperature for structured output
            max_tokens=128,
            json_mode=False,  # Start with text format
            log_metadata=log_metadata,
        )
        
        # Parse action
        action_type, argument = self._parse_action(action_text)
        
        # Update scratchpad
        self.scratchpad += f"\nAction {self.step_count + 1}: {action_type}[{argument}]"
        
        # Log action
        duration = time.time() - start_time
        self.logger.log_agent_action(
            agent_name=self.agent_type.value,
            action_type="action",
            message=f"{action_type}[{argument}]",
            context={'parsed_action': action_type, 'argument': argument},
            step_number=self.step_count + 1,
            duration_ms=duration * 1000
        )
        
        return action_type, argument
    
    def forward(self, stage: str = None, **kwargs) -> str:
        """
        Main forward method supporting both think and act stages.
        
        Args:
            stage: 'thought' or 'action'
            **kwargs: Task context and arguments
            
        Returns:
            Response based on stage
        """
        if stage == 'thought':
            return self.think(kwargs)
        elif stage == 'action':
            action_type, argument = self.act(kwargs)
            return f"{action_type}[{argument}]"
        else:
            # Default: full think-act cycle
            thought = self.think(kwargs)
            action_type, argument = self.act(kwargs)
            return f"Thought: {thought}\nAction: {action_type}[{argument}]"
    
    def _build_thinking_prompt(self, task_context: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """Build prompt text, template, and variables for the thinking phase."""

        context_str = (
            json.dumps(task_context, indent=2, default=str)
            if task_context
            else "No context provided."
        )
        scratchpad_text = self.scratchpad.strip() or "No prior thoughts recorded."

        template = (
            "You are a Manager agent in a sequential recommendation system.\n\n"
            "CURRENT SITUATION:\n"
            "{context}\n\n"
            "SCRATCHPAD (previous thoughts and actions):\n"
            "{scratchpad}\n\n"
            "TASK: Analyze the current situation and think about what information you need to make a good sequential recommendation.\n\n"
            "Consider:\n"
            "1. What do we know about the user's sequential behavior?\n"
            "2. What additional information might be helpful?\n"
            "3. What should be our next step?\n\n"
            "Think step by step about the reasoning process."
        )

        variables = {
            'context': context_str,
            'scratchpad': scratchpad_text,
        }

        prompt_text = template.format(**variables)
        return prompt_text, template, variables
    
    def _build_action_prompt(self, task_context: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """Build prompt text, template, and variables for the action phase."""

        context_str = (
            json.dumps(task_context, indent=2, default=str)
            if task_context
            else "No context provided."
        )
        scratchpad_text = self.scratchpad.strip() or "No prior thoughts recorded."

        template = (
            "Based on your thinking, choose the most appropriate action:\n\n"
            "AVAILABLE ACTIONS:\n"
            "- Analyse[user, user_id] - analyze user preferences and sequential patterns\n"
            "- Analyse[item, item_id] - analyze specific item characteristics\n"
            "- Search[query] - search for external information\n"
            "- Finish[result] - return final recommendation result\n\n"
            "CURRENT CONTEXT:\n"
            "{context}\n\n"
            "SCRATCHPAD:\n"
            "{scratchpad}\n\n"
            "Choose ONE action and return it in the exact format shown above.\n"
            "If you have enough information to make a recommendation, use Finish[result].\n"
            "Otherwise, choose the most useful analysis or search action."
        )

        variables = {
            'context': context_str,
            'scratchpad': scratchpad_text,
        }

        prompt_text = template.format(**variables)
        return prompt_text, template, variables
    
    def _parse_action(self, action_text: str) -> Tuple[str, Any]:
        """
        Parse action text into type and argument.
        
        Args:
            action_text: Raw action text from LLM
            
        Returns:
            Tuple of (action_type, argument)
        """
        # Clean the action text
        action_text = action_text.strip()
        
        # Try to extract action using regex
        patterns = [
            r'(Analyse|Search|Finish)\[([^\]]+)\]',  # Standard format
            r'(Analyse|Search|Finish)\s*\[([^\]]+)\]',  # With spaces
            r'Action:\s*(Analyse|Search|Finish)\[([^\]]+)\]',  # With "Action:" prefix
        ]
        
        for pattern in patterns:
            match = re.search(pattern, action_text, re.IGNORECASE)
            if match:
                action_type = match.group(1).lower().capitalize()
                argument = match.group(2).strip()
                
                # Parse argument based on action type
                if action_type == "Analyse":
                    # Parse "user, 524" or "item, 181"
                    if ',' in argument:
                        parts = [p.strip() for p in argument.split(',')]
                        return action_type, parts
                    else:
                        return action_type, [argument]
                else:
                    return action_type, argument
        
        # Fallback: couldn't parse, default to Finish
        self.logger.log_agent_action(
            agent_name=self.agent_type.value,
            action_type="warning",
            message=f"Could not parse action: {action_text}. Defaulting to Finish.",
            context={'raw_action': action_text}
        )
        
        return "Finish", "Unable to parse action"
    
    def should_continue(self) -> bool:
        """Check if manager should continue processing"""
        return self.step_count < self.max_steps and not self.is_finished
    
    def finish(self, result: Any):
        """Mark manager as finished with result"""
        self.is_finished = True
        self.logger.log_agent_action(
            agent_name=self.agent_type.value,
            action_type="completion",
            message=f"Manager finished with result: {result}",
            context={'final_result': result}
        )


class MockAgent(Agent):
    """Mock agent for testing"""
    
    def __init__(self, agent_type: AgentType, responses: Dict[str, str] = None):
        from ..models.llm_provider import MockLLMProvider
        mock_llm = MockLLMProvider(responses)
        super().__init__(agent_type, mock_llm)
        self.predefined_responses = responses or {}
    
    def forward(self, **kwargs) -> str:
        """Return mock response"""
        argument = kwargs.get('argument', '')
        
        # Check for predefined responses
        for key, response in self.predefined_responses.items():
            if str(key) in str(argument):
                return response
        
        # Default mock responses by agent type
        if self.agent_type == AgentType.ANALYST:
            return "Mock analysis: User prefers category X based on recent interactions"
        elif self.agent_type == AgentType.REFLECTOR:
            return '{"correctness": false, "reason": "Mock reflection: Need more user context"}'
        else:
            return f"Mock response from {self.agent_type.value}"
