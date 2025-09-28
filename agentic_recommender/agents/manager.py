"""
Manager agent implementation.
Central orchestrator using two-stage LLM architecture.

Reference: previousWorks/MACRec/macrec/agents/manager.py
"""

import time
import json
import re
from typing import Dict, Any, Optional, Tuple

from .base import Agent, AgentType
from ..models.llm_provider import LLMProvider
from ..utils.logging import get_component_logger
from ..configs.prompt_loader import load_prompt_config


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

        # Prompt configuration
        self.prompt_config = load_prompt_config("manager")
        self.max_thought_tokens = self.prompt_config.get("max_thought_tokens", 256)
        self.max_action_tokens = self.prompt_config.get("max_action_tokens", 128)
        self.stop_sequences = self.prompt_config.get("stop", [])
        
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
        prompt = self._build_thinking_prompt(task_context)

        # Generate thought using thought LLM
        thought = self.thought_llm.generate(
            prompt, 
            temperature=0.8,  # Higher temperature for creative reasoning
            max_tokens=self.max_thought_tokens
        )
        
        # Update scratchpad
        self.scratchpad += f"\nThought {self.step_count + 1}: {thought}"
        
        # Log thinking
        duration = time.time() - start_time
        self.logger.log_agent_action(
            agent_name=self.agent_type.value,
            action_type="thought",
            message=thought,
            context={
                'prompt': prompt,
                'task_context': self._compact_task_context(task_context),
            },
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
        
        # Build action prompt
        prompt = self._build_action_prompt(task_context or {})
        
        # Generate action using action LLM
        action_text = self.action_llm.generate(
            prompt,
            temperature=0.2,  # Lower temperature for structured output
            max_tokens=self.max_action_tokens,
            json_mode=True
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
            context={
                'parsed_action': action_type,
                'argument': argument,
                'prompt': prompt,
                'raw_response': action_text,
            },
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
    
    def _build_thinking_prompt(self, task_context: Dict[str, Any]) -> str:
        """
        Build prompt for thinking phase.
        
        Template reference: previousWorks/MACRec/config/prompts/manager_prompt/analyse.json
        Based on MACRec manager_prompt template for thought generation.
        """
        template = self.prompt_config.get("thinking_template", "")

        user_context = self._format_user_context(task_context)
        candidate_table = self._format_candidate_table(task_context)
        analyst_summary = self._summarise_text(task_context.get('last_analysis'))
        search_summary = self._summarise_text(task_context.get('last_search'))
        reflection_notes = self._format_reflection_notes(task_context)
        scratchpad = self.scratchpad or "No previous thoughts yet."

        filled = template.format(
            user_context=user_context,
            candidate_table=candidate_table,
            analyst_summary=analyst_summary,
            search_summary=search_summary,
            reflection_notes=reflection_notes,
            scratchpad=scratchpad,
        )

        return filled
    
    def _build_action_prompt(self, task_context: Dict[str, Any]) -> str:
        """
        Build prompt for action phase.
        
        Template reference: previousWorks/MACRec/config/prompts/manager_prompt/analyse.json
        Uses MACRec manager_prompt template with AVAILABLE ACTIONS format.
        """
        template = self.prompt_config.get("action_template", "")

        action_context = self._format_action_context(task_context)
        scratchpad = self.scratchpad or "No previous thoughts yet."

        filled = template.format(
            action_context=action_context,
            scratchpad=scratchpad,
        )

        return filled
    
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

        # Try JSON parsing first
        try:
            action_obj = json.loads(action_text)
            action_type = str(action_obj.get('type', '')).strip().lower().capitalize()
            argument = action_obj.get('content')

            if action_type == "Analyse" and isinstance(argument, list):
                return action_type, [str(part) for part in argument]
            if action_type in {"Search", "Finish"}:
                return action_type, argument
        except json.JSONDecodeError:
            pass

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

    def _format_user_context(self, task_context: Dict[str, Any]) -> str:
        """Build a readable snapshot of the user context."""
        if not task_context:
            return "No context provided."

        user_id = task_context.get('user_id', 'unknown')
        sequence = task_context.get('user_sequence', [])
        ground_truth = task_context.get('ground_truth')
        extra_context = task_context.get('context', {})
        
        lines = [f"user_id: {user_id}"]
        if sequence:
            lines.append(f"recent_sequence: {' → '.join(sequence[-10:])}")
        if ground_truth:
            lines.append(f"ground_truth: {ground_truth}")
        user_profile = extra_context.get('user_profile') if isinstance(extra_context, dict) else None
        if user_profile:
            lines.append(f"user_profile: {json.dumps(user_profile, ensure_ascii=False)}")

        return "\n".join(lines)

    def _format_candidate_table(self, task_context: Dict[str, Any]) -> str:
        """Format candidate items into a compact table representation."""
        candidates = task_context.get('candidates') or []
        if not candidates:
            return "No candidates provided."

        candidate_names = {}
        context_block = task_context.get('context') or {}
        if isinstance(context_block, dict):
            candidate_names = context_block.get('candidate_names') or {}

        lines = []
        for idx, item in enumerate(candidates):
            display = candidate_names.get(item, item)
            lines.append(f"- {idx+1}. {display} ({item})")
        return "\n".join(lines)

    def _summarise_text(self, text: Any) -> str:
        """Trim and normalise optional text snippets for prompt injection."""
        if not text:
            return "None yet."

        if isinstance(text, (dict, list)):
            text = json.dumps(text, ensure_ascii=False)

        text = str(text).strip()
        return text[:400] + ('…' if len(text) > 400 else '')

    def _format_reflection_notes(self, task_context: Dict[str, Any]) -> str:
        """Prepare reflection information for the thinking prompt."""
        reflection = task_context.get('reflection') or task_context.get('reflection_notes')
        if not reflection:
            return "No prior reflections."

        if isinstance(reflection, dict):
            return json.dumps(reflection, ensure_ascii=False)

        return str(reflection)

    def _format_action_context(self, task_context: Dict[str, Any]) -> str:
        """Compact context block for the action prompt."""
        context = {
            'user_id': task_context.get('user_id'),
            'sequence_tail': task_context.get('user_sequence', [])[-5:],
            'candidates': task_context.get('candidates'),
            'last_analysis': task_context.get('last_analysis'),
            'last_search': task_context.get('last_search'),
        }

        return json.dumps(context, ensure_ascii=False, indent=2)

    def _compact_task_context(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        if not task_context:
            return {}

        compact = {
            'user_id': task_context.get('user_id'),
            'sequence_tail': task_context.get('user_sequence', [])[-5:],
            'candidates': task_context.get('candidates'),
            'ground_truth': task_context.get('ground_truth'),
        }

        if task_context.get('context'):
            compact['context_keys'] = list(task_context['context'].keys())

        if task_context.get('last_analysis'):
            compact['last_analysis'] = self._summarise_text(task_context['last_analysis'])

        if task_context.get('last_search'):
            compact['last_search'] = self._summarise_text(task_context['last_search'])

        return compact
    
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
