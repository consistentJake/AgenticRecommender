# Agentic Recommender Agents - Design Document

## Overview

This document outlines the redesigned architecture for the agentic recommendation system, adapted to work with the Singapore food delivery dataset and incorporating enhanced collaborative filtering using Swing similarity.

## Table of Contents

1. [Current Architecture Analysis](#current-architecture-analysis)
2. [Dataset Adaptation](#dataset-adaptation)
3. [Swing Similarity Module](#swing-similarity-module)
4. [Enhanced Reflector Design](#enhanced-reflector-design)
5. [Unified LLM Provider](#unified-llm-provider)
6. [Implementation Plan](#implementation-plan)
7. [Research Findings & Recommendations](#research-findings--recommendations)

---

## Current Architecture Analysis

### Existing Components

```
agents/
├── base.py       # Agent, ToolAgent, ReflectionStrategy base classes
├── analyst.py    # User/item analysis with tool access
├── reflector.py  # Self-reflection for recommendation improvement
└── manager.py    # Two-stage LLM orchestrator (think + act)
```

### Key Observations

1. **Manager Agent**: Uses dual-LLM architecture (thought_llm for reasoning, action_llm for structured output)
2. **Analyst Agent**: Implements tool-based data access (UserInfo, ItemInfo, etc.)
3. **Reflector Agent**: Supports multiple reflection strategies (NONE, LAST_ATTEMPT, REFLEXION, LAST_ATTEMPT_AND_REFLEXION)

### Current Gaps

- No integration with actual dataset format (food delivery vs movies)
- Missing collaborative filtering signals from similar users
- No multi-round LLM judgment with evidence aggregation
- Single LLM provider type (Gemini/OpenRouter only)

---

## Dataset Adaptation

### Singapore Food Delivery Dataset Format

```json
{
  "instruction": "User recent orders (oldest → newest):\n\n| idx | day | hour | cuisine | price |\n...\n\nCandidate product:\n- Buffalo Wings (cuisine: mexikanskt, price: $0.32)",
  "input": "",
  "output": "Yes",
  "system": "You are a food delivery recommendation assistant...",
  "history": []
}
```

### Data Fields

| Field | Description |
|-------|-------------|
| `instruction` | User order history table + candidate product |
| `input` | Empty (combined with instruction) |
| `output` | Ground truth: "Yes" or "No" |
| `system` | System prompt for the recommendation task |
| `history` | Conversation history (currently empty) |

### Proposed Data Adapter

```python
# agentic_recommender/data/food_delivery_adapter.py

@dataclass
class OrderRecord:
    idx: int
    day: str  # Mon, Tue, Wed, Thu, Fri, Sat, Sun
    hour: int
    cuisine: str
    price: float

@dataclass
class CandidateProduct:
    name: str
    cuisine: str
    price: float

@dataclass
class RecommendationSample:
    user_id: str  # Will need to assign during preprocessing
    orders: List[OrderRecord]
    candidate: CandidateProduct
    ground_truth: bool
    system_prompt: str

def parse_instruction(instruction: str) -> Tuple[List[OrderRecord], CandidateProduct]:
    """Parse instruction text into structured data."""
    # Extract order table and candidate product
    pass
```

### Chat Template Integration

Following the pattern from `finetune/scripts/utils.py`:

```python
def to_agent_messages(sample: RecommendationSample) -> List[Dict[str, str]]:
    """Convert sample to chat messages for agent processing."""
    messages = []

    if sample.system_prompt:
        messages.append({"role": "system", "content": sample.system_prompt})

    # Build user message with order history and candidate
    user_content = format_order_history(sample.orders)
    user_content += f"\n\nCandidate: {sample.candidate.name} ({sample.candidate.cuisine}, ${sample.candidate.price})"
    messages.append({"role": "user", "content": user_content})

    return messages
```

---

## Swing Similarity Module

### Algorithm Background

Swing similarity is an item-based collaborative filtering algorithm developed by Alibaba. Unlike traditional methods (Jaccard, Cosine), Swing considers the graph structure and user pairs, making it anti-noise and more accurate for recommendation tasks.

### Mathematical Formula

For user-to-user similarity:

```
sim(u1, u2) = Σ(i ∈ I(u1) ∩ I(u2)) w(u1, u2, i)

where:
w(u1, u2, i) = 1 / (|I(u1)| + α1)^β × (|I(u2)| + α1)^β × (|U(i)| + α2)
```

For item-to-item similarity (used for finding similar users via their item interactions):

```
sim(i, j) = Σ(u ∈ U(i) ∩ U(j)) Σ(v ∈ U(i) ∩ U(j), v≠u)
            [1 / ((|I(u)| + α1)^β × (|I(v)| + α1)^β)] × [1 / (|I(u) ∩ I(v)| + α2)]
```

### Recommended Parameters

- `α1 = 5` (smoothing for user activity)
- `α2 = 1` (smoothing for intersection term)
- `β = 0.3` (power weight for user activity)
- `similarity_threshold = 0.1` (minimum similarity to consider)
- `top_k = 5` (number of similar users to retrieve)

### Implementation Design

```python
# agentic_recommender/similarity/swing.py

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import numpy as np

@dataclass
class SwingConfig:
    alpha1: float = 5.0
    alpha2: float = 1.0
    beta: float = 0.3
    similarity_threshold: float = 0.1
    top_k: int = 5

class SwingSimilarity:
    """Swing similarity calculator for collaborative filtering."""

    def __init__(self, config: SwingConfig = None):
        self.config = config or SwingConfig()
        self.user_items: Dict[str, Set[str]] = {}  # user_id -> set of item_ids
        self.item_users: Dict[str, Set[str]] = {}  # item_id -> set of user_ids
        self._similarity_cache: Dict[Tuple[str, str], float] = {}

    def fit(self, interactions: List[Tuple[str, str]]):
        """Build user-item and item-user mappings from interactions."""
        for user_id, item_id in interactions:
            if user_id not in self.user_items:
                self.user_items[user_id] = set()
            if item_id not in self.item_users:
                self.item_users[item_id] = set()
            self.user_items[user_id].add(item_id)
            self.item_users[item_id].add(user_id)

    def compute_user_similarity(self, user1: str, user2: str) -> float:
        """Compute Swing similarity between two users."""
        cache_key = (min(user1, user2), max(user1, user2))
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        items1 = self.user_items.get(user1, set())
        items2 = self.user_items.get(user2, set())
        common_items = items1 & items2

        if not common_items:
            self._similarity_cache[cache_key] = 0.0
            return 0.0

        α1, α2, β = self.config.alpha1, self.config.alpha2, self.config.beta

        similarity = 0.0
        for item in common_items:
            item_users = self.item_users.get(item, set())
            user_weight = 1.0 / ((len(items1) + α1) ** β * (len(items2) + α1) ** β)
            item_weight = 1.0 / (len(item_users) + α2)
            similarity += user_weight * item_weight

        self._similarity_cache[cache_key] = similarity
        return similarity

    def get_similar_users(
        self,
        user_id: str,
        exclude_users: Set[str] = None
    ) -> List[Tuple[str, float]]:
        """Get top-k similar users above threshold."""
        exclude = exclude_users or set()
        exclude.add(user_id)

        similarities = []
        for other_user in self.user_items.keys():
            if other_user in exclude:
                continue
            sim = self.compute_user_similarity(user_id, other_user)
            if sim >= self.config.similarity_threshold:
                similarities.append((other_user, sim))

        # Sort by similarity descending, take top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:self.config.top_k]
```

---

## Enhanced Reflector Design

### Architecture

The enhanced Reflector uses a two-stage judgment process:

1. **First Stage**: Initial LLM prediction based on user history
2. **Second Stage**: Refined prediction using similar user evidence + reflection

```
┌─────────────────────────────────────────────────────────────────┐
│                     Enhanced Reflector                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐    ┌──────────────────┐    ┌──────────────┐ │
│  │ First Round   │───>│ Swing Similarity │───>│ Second Round │ │
│  │ LLM Judgment  │    │ User Retrieval   │    │ LLM Judgment │ │
│  └───────────────┘    └──────────────────┘    └──────────────┘ │
│         │                     │                      │          │
│         v                     v                      v          │
│  ┌───────────────┐    ┌──────────────────┐    ┌──────────────┐ │
│  │ Initial       │    │ Top-5 Similar    │    │ Final        │ │
│  │ Prediction    │    │ User Evidence    │    │ Decision +   │ │
│  │ + Confidence  │    │ + Their Choices  │    │ Reasoning    │ │
│  └───────────────┘    └──────────────────┘    └──────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
# agentic_recommender/agents/enhanced_reflector.py

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from .base import Agent, AgentType, ReflectionStrategy
from ..similarity.swing import SwingSimilarity, SwingConfig
from ..models.llm_provider import LLMProvider

@dataclass
class SimilarUserEvidence:
    user_id: str
    similarity_score: float
    recent_orders: List[Dict[str, Any]]
    decision_for_candidate: Optional[bool]  # Did they buy similar items?

@dataclass
class ReflectionResult:
    first_round_prediction: bool
    first_round_confidence: float
    first_round_reasoning: str
    similar_users: List[SimilarUserEvidence]
    final_prediction: bool
    final_confidence: float
    final_reasoning: str

class EnhancedReflector(Agent):
    """
    Enhanced Reflector with Swing similarity and two-stage LLM judgment.

    Stage 1: Initial LLM prediction based on user's order history
    Stage 2: Refined prediction using similar user evidence
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        swing_similarity: SwingSimilarity,
        config: Dict[str, Any] = None
    ):
        super().__init__(AgentType.REFLECTOR, llm_provider, config)
        self.swing = swing_similarity
        self.config = config or {}

        # Thresholds
        self.similarity_threshold = self.config.get('similarity_threshold', 0.1)
        self.top_k = self.config.get('top_k', 5)

    def forward(
        self,
        user_id: str,
        order_history: List[Dict[str, Any]],
        candidate_product: Dict[str, Any],
        user_database: Dict[str, Any] = None,
        **kwargs
    ) -> ReflectionResult:
        """
        Execute two-stage reflection process.

        Args:
            user_id: Current user identifier
            order_history: User's recent orders
            candidate_product: Product being considered
            user_database: Database of all users for similarity lookup
        """
        # Stage 1: First round LLM judgment
        first_round = self._first_round_judgment(
            order_history, candidate_product
        )

        # Get similar users using Swing
        similar_users = self._get_similar_user_evidence(
            user_id, candidate_product, user_database
        )

        # Stage 2: Second round with similar user evidence
        final_result = self._second_round_judgment(
            order_history,
            candidate_product,
            first_round,
            similar_users
        )

        return final_result

    def _first_round_judgment(
        self,
        order_history: List[Dict[str, Any]],
        candidate_product: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Initial prediction based on user history alone."""
        prompt = self._build_first_round_prompt(order_history, candidate_product)

        response = self.llm_provider.generate(
            prompt,
            temperature=0.3,
            json_mode=True
        )

        return self._parse_judgment_response(response)

    def _get_similar_user_evidence(
        self,
        user_id: str,
        candidate_product: Dict[str, Any],
        user_database: Dict[str, Any]
    ) -> List[SimilarUserEvidence]:
        """Retrieve evidence from similar users."""
        similar_users = self.swing.get_similar_users(user_id)

        evidence_list = []
        for sim_user_id, similarity_score in similar_users:
            if similarity_score < self.similarity_threshold:
                continue

            user_data = user_database.get(sim_user_id, {})
            evidence = SimilarUserEvidence(
                user_id=sim_user_id,
                similarity_score=similarity_score,
                recent_orders=user_data.get('orders', []),
                decision_for_candidate=self._check_user_bought_similar(
                    user_data, candidate_product
                )
            )
            evidence_list.append(evidence)

        return evidence_list

    def _second_round_judgment(
        self,
        order_history: List[Dict[str, Any]],
        candidate_product: Dict[str, Any],
        first_round: Dict[str, Any],
        similar_users: List[SimilarUserEvidence]
    ) -> ReflectionResult:
        """Refined prediction with similar user evidence."""
        prompt = self._build_second_round_prompt(
            order_history,
            candidate_product,
            first_round,
            similar_users
        )

        response = self.llm_provider.generate(
            prompt,
            temperature=0.3,
            json_mode=True
        )

        final = self._parse_judgment_response(response)

        return ReflectionResult(
            first_round_prediction=first_round.get('prediction', False),
            first_round_confidence=first_round.get('confidence', 0.5),
            first_round_reasoning=first_round.get('reasoning', ''),
            similar_users=similar_users,
            final_prediction=final.get('prediction', False),
            final_confidence=final.get('confidence', 0.5),
            final_reasoning=final.get('reasoning', '')
        )

    def _build_first_round_prompt(
        self,
        order_history: List[Dict[str, Any]],
        candidate_product: Dict[str, Any]
    ) -> str:
        """Build prompt for first round judgment."""
        history_str = self._format_order_history(order_history)

        return f"""You are a food delivery recommendation system. Analyze the user's order history and predict whether they will purchase the candidate product.

## User's Recent Orders
{history_str}

## Candidate Product
- Name: {candidate_product.get('name', 'Unknown')}
- Cuisine: {candidate_product.get('cuisine', 'Unknown')}
- Price: ${candidate_product.get('price', 0):.2f}

## Task
Predict whether the user is likely to purchase this product. Consider:
1. Cuisine preferences (do they order this type of food?)
2. Price sensitivity (is this within their typical spending?)
3. Time patterns (when do they typically order?)
4. Sequential behavior (what do they order after certain cuisines?)

Return JSON: {{"prediction": true/false, "confidence": 0.0-1.0, "reasoning": "explanation"}}"""

    def _build_second_round_prompt(
        self,
        order_history: List[Dict[str, Any]],
        candidate_product: Dict[str, Any],
        first_round: Dict[str, Any],
        similar_users: List[SimilarUserEvidence]
    ) -> str:
        """Build prompt for second round with similar user evidence."""
        history_str = self._format_order_history(order_history)
        similar_evidence = self._format_similar_user_evidence(similar_users)

        return f"""You are a food delivery recommendation system performing a REFINED prediction.

## User's Recent Orders
{history_str}

## Candidate Product
- Name: {candidate_product.get('name', 'Unknown')}
- Cuisine: {candidate_product.get('cuisine', 'Unknown')}
- Price: ${candidate_product.get('price', 0):.2f}

## First Round Analysis
- Initial Prediction: {"Yes" if first_round.get('prediction') else "No"}
- Confidence: {first_round.get('confidence', 0.5):.1%}
- Reasoning: {first_round.get('reasoning', 'N/A')}

## Evidence from Similar Users
{similar_evidence}

## Task
Review your initial prediction considering the behavior of similar users.
- If similar users (high similarity score) consistently bought/rejected similar items, weigh that heavily
- If similar user behavior contradicts your initial prediction, reconsider
- Provide your final decision with reasoning

Return JSON: {{"prediction": true/false, "confidence": 0.0-1.0, "reasoning": "explanation including how similar user evidence influenced your decision"}}"""

    def _format_order_history(self, orders: List[Dict[str, Any]]) -> str:
        """Format orders as markdown table."""
        if not orders:
            return "No order history available."

        lines = ["| # | Day | Hour | Cuisine | Price |", "|---|-----|------|---------|-------|"]
        for i, order in enumerate(orders, 1):
            lines.append(f"| {i} | {order.get('day', '?')} | {order.get('hour', '?')} | {order.get('cuisine', '?')} | ${order.get('price', 0):.2f} |")
        return "\n".join(lines)

    def _format_similar_user_evidence(self, similar_users: List[SimilarUserEvidence]) -> str:
        """Format similar user evidence."""
        if not similar_users:
            return "No similar users found above threshold."

        lines = []
        for i, user in enumerate(similar_users, 1):
            decision = "Bought similar" if user.decision_for_candidate else "Did not buy similar"
            cuisines = set(o.get('cuisine', 'unknown') for o in user.recent_orders[:5])
            lines.append(f"""
**Similar User {i}** (Similarity: {user.similarity_score:.2%})
- Recent cuisines: {', '.join(cuisines)}
- Action for similar products: {decision}""")

        return "\n".join(lines)

    def _check_user_bought_similar(
        self,
        user_data: Dict[str, Any],
        candidate: Dict[str, Any]
    ) -> bool:
        """Check if user bought items similar to candidate."""
        orders = user_data.get('orders', [])
        candidate_cuisine = candidate.get('cuisine', '').lower()

        for order in orders:
            if order.get('cuisine', '').lower() == candidate_cuisine:
                return True
        return False

    def _parse_judgment_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response."""
        import json
        try:
            data = json.loads(response)
            return {
                'prediction': bool(data.get('prediction', False)),
                'confidence': float(data.get('confidence', 0.5)),
                'reasoning': str(data.get('reasoning', ''))
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: try to extract Yes/No from text
            response_lower = response.lower()
            prediction = 'yes' in response_lower and 'no' not in response_lower
            return {
                'prediction': prediction,
                'confidence': 0.5,
                'reasoning': response
            }
```

---

## Unified LLM Provider

### Design Goals

1. **Multi-provider support**: Claude API (priority), Gemini, OpenRouter, Local models
2. **Consistent interface**: Same API across all providers
3. **Fallback chain**: Automatic failover between providers
4. **Cost optimization**: Route to cheaper models for simple tasks

### Provider Hierarchy

```
Priority 1: Claude API (anthropic)
    └─ claude-sonnet-4-20250514 (default)
    └─ claude-3-haiku-20240307 (fast/cheap)

Priority 2: Gemini Direct
    └─ gemini-2.0-flash-exp

Priority 3: OpenRouter
    └─ google/gemini-flash-1.5
    └─ anthropic/claude-3-sonnet

Priority 4: Local (Ollama/vLLM)
    └─ qwen3:0.6b
    └─ llama3:8b
```

### Implementation

```python
# agentic_recommender/models/unified_llm_provider.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import logging

class ProviderType(Enum):
    CLAUDE = "claude"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    LOCAL = "local"
    MOCK = "mock"

@dataclass
class ProviderConfig:
    provider_type: ProviderType
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    priority: int = 0
    max_tokens: int = 4096
    timeout: int = 30

@dataclass
class UnifiedLLMConfig:
    providers: List[ProviderConfig] = field(default_factory=list)
    fallback_enabled: bool = True
    retry_count: int = 2

    @classmethod
    def default_config(cls) -> 'UnifiedLLMConfig':
        """Default configuration with Claude as primary."""
        return cls(providers=[
            ProviderConfig(
                provider_type=ProviderType.CLAUDE,
                model_name="claude-sonnet-4-20250514",
                priority=1
            ),
            ProviderConfig(
                provider_type=ProviderType.GEMINI,
                model_name="gemini-2.0-flash-exp",
                priority=2
            ),
            ProviderConfig(
                provider_type=ProviderType.OPENROUTER,
                model_name="google/gemini-flash-1.5",
                priority=3
            ),
        ])

class UnifiedLLMProvider:
    """
    Unified LLM provider with multi-backend support and automatic fallback.
    """

    def __init__(self, config: UnifiedLLMConfig = None):
        self.config = config or UnifiedLLMConfig.default_config()
        self.providers: Dict[ProviderType, Any] = {}
        self.logger = logging.getLogger(__name__)
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize all configured providers."""
        # Sort by priority
        sorted_configs = sorted(
            self.config.providers,
            key=lambda x: x.priority
        )

        for provider_config in sorted_configs:
            try:
                provider = self._create_provider(provider_config)
                if provider:
                    self.providers[provider_config.provider_type] = provider
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize {provider_config.provider_type}: {e}"
                )

    def _create_provider(self, config: ProviderConfig) -> Optional[Any]:
        """Create provider instance from config."""
        if config.provider_type == ProviderType.CLAUDE:
            return self._create_claude_provider(config)
        elif config.provider_type == ProviderType.GEMINI:
            return self._create_gemini_provider(config)
        elif config.provider_type == ProviderType.OPENROUTER:
            return self._create_openrouter_provider(config)
        elif config.provider_type == ProviderType.LOCAL:
            return self._create_local_provider(config)
        elif config.provider_type == ProviderType.MOCK:
            return self._create_mock_provider(config)
        return None

    def _create_claude_provider(self, config: ProviderConfig):
        """Create Claude/Anthropic provider."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")

        return anthropic.Anthropic(api_key=config.api_key)

    def _create_gemini_provider(self, config: ProviderConfig):
        """Create Gemini provider (reuse existing)."""
        from .llm_provider import GeminiProvider
        return GeminiProvider(
            api_key=config.api_key,
            model_name=config.model_name,
            use_openrouter=False
        )

    def _create_openrouter_provider(self, config: ProviderConfig):
        """Create OpenRouter provider."""
        from .llm_provider import GeminiProvider
        return GeminiProvider(
            api_key=config.api_key,
            model_name=config.model_name,
            use_openrouter=True
        )

    def _create_local_provider(self, config: ProviderConfig):
        """Create local model provider (Ollama/vLLM)."""
        # To be implemented based on local infrastructure
        raise NotImplementedError("Local provider not yet implemented")

    def _create_mock_provider(self, config: ProviderConfig):
        """Create mock provider for testing."""
        from .llm_provider import MockLLMProvider
        return MockLLMProvider()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        preferred_provider: ProviderType = None,
        **kwargs
    ) -> str:
        """
        Generate text using the best available provider.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            preferred_provider: Override automatic provider selection
        """
        providers_to_try = self._get_provider_order(preferred_provider)

        last_error = None
        for provider_type in providers_to_try:
            provider = self.providers.get(provider_type)
            if not provider:
                continue

            try:
                result = self._generate_with_provider(
                    provider, provider_type, prompt, temperature, max_tokens, **kwargs
                )
                return result
            except Exception as e:
                last_error = e
                self.logger.warning(f"Provider {provider_type} failed: {e}")
                if not self.config.fallback_enabled:
                    break

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    def _generate_with_provider(
        self,
        provider: Any,
        provider_type: ProviderType,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using specific provider."""
        if provider_type == ProviderType.CLAUDE:
            return self._generate_claude(provider, prompt, temperature, max_tokens, **kwargs)
        else:
            # Use existing interface for other providers
            return provider.generate(
                prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
            )

    def _generate_claude(
        self,
        client: Any,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using Claude API."""
        model = kwargs.get('model', 'claude-sonnet-4-20250514')
        system = kwargs.get('system', '')

        messages = [{"role": "user", "content": prompt}]

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            system=system if system else None,
            temperature=temperature
        )

        return response.content[0].text

    def _get_provider_order(self, preferred: ProviderType = None) -> List[ProviderType]:
        """Get ordered list of providers to try."""
        if preferred and preferred in self.providers:
            others = [p for p in self.providers.keys() if p != preferred]
            return [preferred] + others

        # Return by priority order
        sorted_configs = sorted(
            self.config.providers,
            key=lambda x: x.priority
        )
        return [c.provider_type for c in sorted_configs if c.provider_type in self.providers]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available providers."""
        return {
            'available_providers': list(self.providers.keys()),
            'fallback_enabled': self.config.fallback_enabled,
            'provider_count': len(self.providers)
        }
```

---

## Implementation Plan

### Phase 1: Data Adapter (Priority: High)

1. Create `food_delivery_adapter.py` to parse dataset format
2. Build user ID assignment from order patterns
3. Create interaction index for Swing similarity

### Phase 2: Swing Similarity (Priority: High)

1. Implement `SwingSimilarity` class
2. Pre-compute user-item mappings from training data
3. Add caching for similarity scores
4. Unit tests with sample data

### Phase 3: Enhanced Reflector (Priority: High)

1. Implement two-stage judgment flow
2. Integrate Swing similarity retrieval
3. Build prompt templates for both stages
4. Add reflection result logging

### Phase 4: Unified LLM Provider (Priority: Medium)

1. Add Claude API support with `anthropic` package
2. Create provider configuration system
3. Implement fallback chain
4. Add provider health monitoring

### Phase 5: Integration (Priority: Medium)

1. Update Manager agent to use new components
2. Create end-to-end evaluation pipeline
3. Add performance benchmarking
4. Documentation and examples

---

## Research Findings & Recommendations

### Is This a Good Approach?

**Yes, with caveats.** Based on research into recommendation systems and the specific dataset:

#### Strengths

1. **Swing Similarity**: Anti-noise properties make it robust for sparse data. Alibaba reports significant improvements over traditional CF methods.

2. **Two-Stage Reflection**: Mimics human decision-making (initial judgment + peer evidence). Reduces both false positives and false negatives.

3. **LLM-based Reasoning**: Can capture complex patterns that traditional models miss (e.g., "pizza after beer" patterns).

#### Potential Concerns

1. **Latency**: Two LLM calls + similarity lookup adds latency. Consider:
   - Pre-computing similar users during batch processing
   - Using smaller/faster models for first-round judgment
   - Caching frequent user similarity lookups

2. **Cold Start**: New users have no similar users. Mitigations:
   - Fall back to popularity-based recommendations
   - Use content-based features (cuisine similarity)
   - Prompt engineering to handle "no similar users" case

3. **Data Scale**: Swing similarity is O(n²) in worst case. For large user bases:
   - Use locality-sensitive hashing (LSH) for approximate neighbors
   - Pre-cluster users and only search within clusters
   - Limit interaction history to recent N items

#### Alternative Approaches to Consider

1. **Hybrid Model**: Combine finetuned Qwen3 (from finetune pipeline) with retrieval augmentation
2. **Graph Neural Networks**: If scaling to millions of users, GNN-based CF may be more efficient
3. **Multi-Task Learning**: Train model to predict both "will buy" and "similar to user X"

### Recommended Metrics

Track these during evaluation:

| Metric | Target | Notes |
|--------|--------|-------|
| Accuracy | >85% | Overall prediction accuracy |
| Precision@5 | >75% | Precision of top-5 recommendations |
| Latency (p50) | <500ms | Median response time |
| Latency (p99) | <2s | 99th percentile response time |
| Similar User Hit Rate | >60% | % of users with valid similar users |

---

## References

- [Swing Algorithm - Alibaba Cloud](https://www.alibabacloud.com/help/en/airec/swing-algorithm-tools)
- [Collaborative Filtering Survey](https://www.sciencedirect.com/science/article/pii/S1319157821002652)
- MACRec Paper and Analysis (`papers/MacRec.pdf`, `MACRec_Analysis.md`)
