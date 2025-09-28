"""
Analyst agent implementation.
Specializes in analyzing user and item characteristics.

Reference: previousWorks/MACRec/macrec/agents/analyst.py
"""

import json
from typing import Dict, Any, List, Optional

from .base import ToolAgent, AgentType
from ..models.llm_provider import LLMProvider
from ..utils.logging import get_component_logger
from ..configs.prompt_loader import load_prompt_config


class Analyst(ToolAgent):
    """
    Analyst agent for user and item analysis.
    
    Responsibilities:
    - Analyze user preferences from historical sequences
    - Extract item features and patterns
    - Provide insights for sequential recommendations
    
    Reference: MACRec_Analysis.md:81-198
    """
    
    def __init__(self, llm_provider: LLMProvider, user_data: Dict[str, Any] = None,
                 item_data: Dict[str, Any] = None, config: Dict[str, Any] = None):
        
        # Initialize tools for data access
        tools = {
            'UserInfo': self._get_user_info,
            'ItemInfo': self._get_item_info,
            'UserHistory': self._get_user_history,
            'ItemHistory': self._get_item_history,
            'SequenceAnalysis': self._analyze_sequence,
            'Finish': self._finish_analysis
        }
        
        super().__init__(AgentType.ANALYST, llm_provider, tools, config=config)
        self.component_logger = get_component_logger("agents.analyst")

        # Data sources
        self.user_data = user_data or {}
        self.item_data = item_data or {}
        self.user_histories = {}  # Will be populated with session data
        self.item_histories = {}

        # Prompt configuration
        self.prompt_config = load_prompt_config("analyst")
        self.max_analysis_steps = (config or {}).get('max_steps', 6)
        self.command_temperature = (config or {}).get('command_temperature', 0.3)
        
        self.component_logger.info(
            "🔍 Analyst initialized with %s users, %s items",
            len(self.user_data),
            len(self.item_data),
        )
    
    def forward(self, argument: Any, json_mode: bool = False, **kwargs) -> str:
        """Iteratively gather evidence using tool commands before finishing."""
        target_type, target_id = self._resolve_target(argument)
        history: List[Dict[str, str]] = []

        for step in range(self.max_analysis_steps):
            prompt = self._build_command_prompt(
                target_type=target_type,
                target_id=target_id,
                history=history,
                request_context=kwargs.get('request_context'),
                manager_notes=kwargs.get('manager_notes'),
            )

            raw_command = self.llm_provider.generate(
                prompt,
                temperature=self.command_temperature,
                max_tokens=196,
                json_mode=True,
            )

            self.logger.log_agent_action(
                agent_name=self.agent_type.value,
                action_type="command_generation",
                message=f"Step {step + 1} command",
                context={
                    'prompt': prompt,
                    'raw_response': raw_command,
                    'target_type': target_type,
                    'target_id': target_id,
                },
                step_number=step + 1,
            )

            command_obj = self._parse_command_response(raw_command)
            command_type = command_obj.get('type')
            content = command_obj.get('content')

            if not command_type:
                # Unable to parse; bail with fallback summary
                return self._fallback_summary(target_id, history, json_mode)

            command_type_lower = str(command_type).lower()

            if command_type_lower == 'finish':
                summary = self._ensure_report_schema(content)
                self.logger.log_agent_action(
                    agent_name=self.agent_type.value,
                    action_type="analysis_summary",
                    message=f"Completed analysis for {target_id}",
                    context={'summary': summary},
                    step_number=step + 1,
                )
                return json.dumps(summary, ensure_ascii=False) if json_mode else self._format_text_summary(summary)

            command_repr, observation = self._execute_analysis_command(command_type_lower, content)
            history.append({
                'command': command_repr,
                'observation': observation,
            })

        # Max steps reached without explicit finish – synthesise summary.
        self.logger.log_agent_action(
            agent_name=self.agent_type.value,
            action_type="analysis_fallback",
            message=f"Max steps reached for {target_id}",
            context={'history': history},
            step_number=self.max_analysis_steps,
        )
        return self._fallback_summary(target_id, history, json_mode)
    
    def _resolve_target(self, argument: Any) -> tuple[str, str]:
        """Determine analysis target type and id from the manager request."""
        if isinstance(argument, list) and len(argument) >= 2:
            return str(argument[0]).lower(), str(argument[1])
        if isinstance(argument, dict):
            target_type = argument.get('type') or argument.get('target') or 'general'
            target_id = argument.get('id') or argument.get('target_id') or 'unknown'
            return str(target_type).lower(), str(target_id)

        return 'general', str(argument)

    def _build_command_prompt(
        self,
        *,
        target_type: str,
        target_id: str,
        history: List[Dict[str, str]],
        request_context: Any = None,
        manager_notes: Any = None,
    ) -> str:
        template = self.prompt_config.get('command_template', '')
        history_block = self._format_history(history)
        request_block = json.dumps(request_context, ensure_ascii=False, indent=2) if request_context else "{}"
        manager_block = self._summarise_text(manager_notes)

        return template.format(
            analysis_target=target_type,
            target_id=target_id,
            request_context=request_block,
            manager_notes=manager_block,
            history=history_block,
        )

    def _parse_command_response(self, raw_command: str) -> Dict[str, Any]:
        try:
            return json.loads(raw_command)
        except json.JSONDecodeError:
            self.logger.log_agent_action(
                agent_name=self.agent_type.value,
                action_type="warning",
                message=f"Analyst command not JSON: {raw_command[:200]}",
            )
            return {}

    def _execute_analysis_command(self, command_type: str, content: Any) -> tuple[str, str]:
        command_repr = self._format_command_repr(command_type, content)
        observation = self.execute_command(command_repr)
        return command_repr, observation

    def _format_command_repr(self, command_type: str, content: Any) -> str:
        if command_type == 'finish':
            return 'Finish[analysis]'

        if isinstance(content, list):
            args = ", ".join(str(item) for item in content)
        elif content is None:
            args = ""
        else:
            args = str(content)

        command_name = command_type[0].upper() + command_type[1:]
        return f"{command_name}[{args}]"

    def _ensure_report_schema(self, content: Any) -> Dict[str, str]:
        schema = self.prompt_config.get('report_schema', {})
        summary: Dict[str, str] = {}
        if isinstance(content, dict):
            for key in schema:
                value = content.get(key)
                summary[key] = self._summarise_text(value) if value else ""
        else:
            fallback_text = self._summarise_text(content)
            for key in schema:
                summary[key] = fallback_text

        for key, description in schema.items():
            summary.setdefault(key, description)

        return summary

    def _format_text_summary(self, summary: Dict[str, str]) -> str:
        lines = []
        for key, value in summary.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _fallback_summary(self, target_id: str, history: List[Dict[str, str]], json_mode: bool) -> str:
        summary = {
            'patterns': f"Limited evidence gathered for {target_id}.",
            'sequential_behavior': self._summarise_text(history[-1]['observation']) if history else "No sequential cues collected.",
            'recent_trends': "Insufficient data to assess trends.",
            'recommendations': "Request additional analyst steps or expand history window.",
        }

        return json.dumps(summary, ensure_ascii=False) if json_mode else self._format_text_summary(summary)

    def _summarise_text(self, value: Any) -> str:
        if not value:
            return ""
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        text = str(value).strip()
        return text[:400] + ('…' if len(text) > 400 else '')

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "No previous commands."

        lines = []
        for idx, entry in enumerate(history, start=1):
            observation = entry.get('observation', '')
            observation = observation.replace('\n', ' ')[:400]
            lines.append(f"Step {idx}: {entry.get('command')} -> {observation}")
        return "\n".join(lines)
    
    def _get_user_info(self, args: List[str]) -> str:
        """Get user profile information"""
        if not args:
            return "No user ID provided"
        
        user_id = args[0]
        
        if user_id in self.user_data:
            user_profile = self.user_data[user_id]
            return f"User {user_id} profile: {user_profile}"
        else:
            return f"User {user_id}: Profile not available"
    
    def _get_item_info(self, args: List[str]) -> str:
        """Get item attributes and details"""
        if not args:
            return "No item ID provided"
        
        item_id = args[0]
        
        if item_id in self.item_data:
            item_info = self.item_data[item_id]
            return f"Item {item_id}: {item_info}"
        else:
            return f"Item {item_id}: Information not available"
    
    def _get_user_history(self, args: List[str]) -> str:
        """Get user's interaction history"""
        if len(args) < 2:
            return "Invalid arguments for UserHistory"
        
        user_id = args[0]
        k = int(args[1]) if args[1].isdigit() else 5
        
        if user_id in self.user_histories:
            history = self.user_histories[user_id][-k:]  # Last k interactions
            history_str = "\n".join([f"- {item}" for item in history])
            return f"User {user_id} recent {k} interactions:\n{history_str}"
        else:
            return f"User {user_id}: No interaction history available"
    
    def _get_item_history(self, args: List[str]) -> str:
        """Get item's interaction history"""
        if len(args) < 2:
            return "Invalid arguments for ItemHistory"
        
        item_id = args[0]
        k = int(args[1]) if args[1].isdigit() else 5
        
        if item_id in self.item_histories:
            history = self.item_histories[item_id][-k:]
            history_str = "\n".join([f"- User {user}" for user in history])
            return f"Item {item_id} recent {k} interactions:\n{history_str}"
        else:
            return f"Item {item_id}: No interaction history available"
    
    def _analyze_sequence(self, args: List[str]) -> str:
        """Analyze a sequence of items for patterns"""
        if not args:
            return "No sequence provided"
        
        sequence = args[0] if isinstance(args[0], list) else args
        
        # Build sequence analysis prompt
        prompt = f"""Analyze this sequence of items for patterns:

SEQUENCE: {sequence}

Identify:
1. Category patterns (are items from similar categories?)
2. Temporal patterns (buying frequency, seasonality)
3. Progression patterns (starter items → advanced items)
4. Complementary patterns (items that go together)

Provide insights about what the user might want next.
"""
        
        return self.llm_provider.generate(prompt, temperature=0.7)
    
    def _finish_analysis(self, args: List[str]) -> str:
        """Finish analysis and return result"""
        result = args[0] if args else "Analysis complete"
        self.is_finished = True
        return result
    
    def update_data_sources(self, user_data: Dict[str, Any] = None, 
                           item_data: Dict[str, Any] = None,
                           user_histories: Dict[str, List] = None,
                           item_histories: Dict[str, List] = None):
        """Update analyst's data sources"""
        if user_data:
            self.user_data.update(user_data)
        if item_data:
            self.item_data.update(item_data)
        if user_histories:
            self.user_histories.update(user_histories)
        if item_histories:
            self.item_histories.update(item_histories)
        
        self.component_logger.info(
            "📊 Analyst data updated: %s users, %s items",
            len(self.user_data),
            len(self.item_data),
        )
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of analyses performed"""
        return {
            'total_analyses': self.total_calls,
            'commands_executed': len(self.command_history),
            'avg_analysis_time': self.total_time / self.total_calls if self.total_calls > 0 else 0,
            'data_sources': {
                'users': len(self.user_data),
                'items': len(self.item_data),
                'user_histories': len(self.user_histories),
                'item_histories': len(self.item_histories)
            }
        }
