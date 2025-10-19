"""
Analyst agent implementation.
Specializes in analyzing user and item characteristics.

Reference: previousWorks/MACRec/macrec/agents/analyst.py
"""

import json
from typing import Dict, Any, List, Optional

from .base import ToolAgent, AgentType, create_agent_prompt
from ..models.llm_provider import LLMProvider
from ..utils.logging import get_component_logger


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
        
        self.component_logger.info(
            "🔍 Analyst initialized with %s users, %s items",
            len(self.user_data),
            len(self.item_data),
        )
    
    def forward(self, argument: Any, json_mode: bool = False, **kwargs) -> str:
        """
        Main forward method for analyst.
        
        Args:
            argument: Analysis request (e.g., ["user", "524"] or ["item", "181"])
            json_mode: Whether to return JSON format
            **kwargs: Additional context
            
        Returns:
            Analysis result
        """
        if isinstance(argument, list) and len(argument) >= 2:
            analysis_type = argument[0].lower()
            target_id = argument[1]
            
            if analysis_type == "user":
                return self._analyze_user(target_id, json_mode, **kwargs)
            elif analysis_type == "item":
                return self._analyze_item(target_id, json_mode, **kwargs)
            else:
                return self._analyze_general(argument, json_mode, **kwargs)
        else:
            return self._analyze_general(argument, json_mode, **kwargs)
    
    def _analyze_user(self, user_id: str, json_mode: bool = False, **kwargs) -> str:
        """
        Analyze user preferences and sequential patterns.
        
        Template reference: previousWorks/MACRec/config/prompts/agent_prompt/analyst.json
        Uses MACRec analyst_prompt template for user analysis.
        """
        
        # Get user information
        user_info = self._get_user_info([user_id])
        user_history = self._get_user_history([user_id, "10"])  # Last 10 interactions
        
        instruction = (
            "Return analysis in JSON format with keys: patterns, sequential_behavior, recent_trends, recommendations"
            if json_mode
            else "Provide detailed analysis."
        )

        template = (
            "Analyze this user's preferences for sequential recommendation:\n\n"
            "USER INFORMATION:\n"
            "{user_info}\n\n"
            "RECENT INTERACTION HISTORY:\n"
            "{user_history}\n\n"
            "TASK: Provide insights about:\n"
            "1. User's preference patterns\n"
            "2. Sequential behavior (what they tend to buy after what)\n"
            "3. Recent trends or shifts in preferences\n"
            "4. Recommendations for next item\n\n"
            "{instruction}"
        )

        variables = {
            'user_info': user_info,
            'user_history': user_history,
            'instruction': instruction,
        }

        prompt_text = template.format(**variables)

        model_info = self.llm_provider.get_model_info()
        # LLM_LOG: user-analysis request metadata ensures prompt template logging stays colour-coded and structured
        log_metadata = {
            'agent': self.agent_type.value,
            'stage': 'user_analysis',
            'target_id': user_id,
            'template_variables': variables,
            'prompt_template': template,
            'model_name': model_info.get('model_name'),
            'provider': model_info.get('provider'),
        }

        analysis = self.llm_provider.generate(
            prompt_text,
            temperature=0.7,
            json_mode=json_mode,
            log_metadata=log_metadata,
        )
        
        # Log the analysis
        self.logger.log_agent_action(
            agent_name=self.agent_type.value,
            action_type="user_analysis",
            message=f"Analyzed user {user_id}",
            context={'user_id': user_id, 'analysis_length': len(analysis)}
        )
        
        return analysis
    
    def _analyze_item(self, item_id: str, json_mode: bool = False, **kwargs) -> str:
        """
        Analyze item characteristics and popularity.
        
        Template reference: previousWorks/MACRec/config/prompts/agent_prompt/analyst.json
        Uses MACRec analyst_prompt template for item analysis.
        """
        
        # Get item information
        item_info = self._get_item_info([item_id])
        item_history = self._get_item_history([item_id, "5"])  # Recent interactions
        
        instruction = (
            "Return analysis in JSON format with keys: characteristics, typical_users, related_items, popularity"
            if json_mode
            else "Provide detailed analysis."
        )

        template = (
            "Analyze this item for recommendation purposes:\n\n"
            "ITEM INFORMATION:\n"
            "{item_info}\n\n"
            "INTERACTION HISTORY:\n"
            "{item_history}\n\n"
            "TASK: Provide insights about:\n"
            "1. Item characteristics and features\n"
            "2. Who typically interacts with this item\n"
            "3. What items are commonly purchased with/after this item\n"
            "4. Item's popularity and trends\n\n"
            "{instruction}"
        )

        variables = {
            'item_info': item_info,
            'item_history': item_history,
            'instruction': instruction,
        }

        prompt_text = template.format(**variables)

        model_info = self.llm_provider.get_model_info()
        # LLM_LOG: item-analysis request metadata ensures prompt template logging stays colour-coded and structured
        log_metadata = {
            'agent': self.agent_type.value,
            'stage': 'item_analysis',
            'target_id': item_id,
            'template_variables': variables,
            'prompt_template': template,
            'model_name': model_info.get('model_name'),
            'provider': model_info.get('provider'),
        }

        analysis = self.llm_provider.generate(
            prompt_text,
            temperature=0.7,
            json_mode=json_mode,
            log_metadata=log_metadata,
        )
        
        # Log the analysis
        self.logger.log_agent_action(
            agent_name=self.agent_type.value,
            action_type="item_analysis",
            message=f"Analyzed item {item_id}",
            context={'item_id': item_id, 'analysis_length': len(analysis)}
        )
        
        return analysis
    
    def _analyze_general(self, argument: Any, json_mode: bool = False, **kwargs) -> str:
        """Handle general analysis requests"""
        request_repr = (
            json.dumps(argument, indent=2, default=str)
            if not isinstance(argument, str)
            else str(argument)
        )
        context_repr = json.dumps(kwargs, indent=2, default=str) if kwargs else "{}"
        instruction = "Return in JSON format." if json_mode else "Provide thorough analysis based on the request."

        template = (
            "You are an Analyst agent. Analyze the following:\n\n"
            "REQUEST: {request}\n\n"
            "CONTEXT: {context}\n\n"
            "{instruction}"
        )

        variables = {
            'request': request_repr,
            'context': context_repr,
            'instruction': instruction,
        }

        prompt_text = template.format(**variables)

        model_info = self.llm_provider.get_model_info()
        # LLM_LOG: general-analysis request metadata for downstream coloured prompt logging
        log_metadata = {
            'agent': self.agent_type.value,
            'stage': 'general_analysis',
            'template_variables': variables,
            'prompt_template': template,
            'model_name': model_info.get('model_name'),
            'provider': model_info.get('provider'),
        }

        return self.llm_provider.generate(
            prompt_text,
            temperature=0.7,
            json_mode=json_mode,
            log_metadata=log_metadata,
        )
    
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
        
        sequence_repr = (
            json.dumps(sequence, indent=2, default=str)
            if not isinstance(sequence, str)
            else str(sequence)
        )

        template = (
            "Analyze this sequence of items for patterns:\n\n"
            "SEQUENCE: {sequence}\n\n"
            "Identify:\n"
            "1. Category patterns (are items from similar categories?)\n"
            "2. Temporal patterns (buying frequency, seasonality)\n"
            "3. Progression patterns (starter items → advanced items)\n"
            "4. Complementary patterns (items that go together)\n\n"
            "Provide insights about what the user might want next."
        )

        variables = {
            'sequence': sequence_repr,
        }

        prompt_text = template.format(**variables)

        model_info = self.llm_provider.get_model_info()
        # LLM_LOG: sequence-analysis request metadata for coloured prompt logging
        log_metadata = {
            'agent': self.agent_type.value,
            'stage': 'sequence_analysis',
            'template_variables': variables,
            'prompt_template': template,
            'model_name': model_info.get('model_name'),
            'provider': model_info.get('provider'),
        }

        return self.llm_provider.generate(
            prompt_text,
            temperature=0.7,
            log_metadata=log_metadata,
        )
    
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
