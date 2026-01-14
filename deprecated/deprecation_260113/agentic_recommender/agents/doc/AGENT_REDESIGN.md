# Agent System Redesign: Hybrid CF-LLM Recommendation

## Overview

This document outlines the redesign of the agent system to:
1. Integrate collaborative filtering (CF) signals into the Reflector's decision process
2. Create a unified LLM module supporting multiple providers (Claude API priority)
3. Adapt the system to work with the Delivery Hero Singapore food delivery dataset

## Research Background

Recent research (2024-2025) demonstrates significant performance gains from hybrid CF-LLM approaches:

- **Enhanced Recommendation (ACM 2025)**: Hybrid CF-LLM achieved 75.6% precision vs 72.3% (CF) and 70.1% (LLM) on MovieLens
- **A-LLMRec (KDD 2024)**: Model-agnostic integration of CF knowledge into LLM without extensive fine-tuning
- **CoLLM (IEEE TKDE 2025)**: Three-component architecture: prompt construction, hybrid encoding, LLM prediction

Key insight: CF captures behavioral patterns from user-item interactions, while LLMs provide semantic understanding and reasoning. Combining both addresses cold-start, data sparsity, and nuanced recommendation.

---

## 1. Unified LLM Module

### 1.1 Current State

The existing `llm_provider.py` supports:
- Gemini API (direct)
- OpenRouter (for various models)
- Mock provider (testing)

### 1.2 Design Goals

1. **Claude API First Priority** - Best reasoning capabilities for recommendation explanations
2. **Provider Abstraction** - Clean interface for switching between providers
3. **Chat Template Compatibility** - Match format used in Qwen3 fine-tuning

### 1.3 Architecture

```
LLMProvider (ABC)
├── ClaudeProvider      # NEW - Primary
├── GeminiProvider      # Existing
├── OpenRouterProvider  # Existing (refactored)
├── QwenLocalProvider   # NEW - For fine-tuned models
└── MockLLMProvider     # Existing
```

### 1.4 ClaudeProvider Implementation

```python
class ClaudeProvider(LLMProvider):
    """
    Claude API provider for LLM inference.
    Primary provider for high-quality reasoning tasks.

    Supports:
    - Claude 3.5 Sonnet (default - cost-effective)
    - Claude 3 Opus (high quality)
    - Claude 3 Haiku (fast/cheap)
    """

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model_name = model_name or self.DEFAULT_MODEL

        # Initialize Anthropic client
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        json_mode: bool = False,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate using Claude API with structured messaging."""

        messages = [{"role": "user", "content": prompt}]

        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=messages,
        )

        return response.content[0].text
```

### 1.5 Configuration Schema

```json
{
  "llm": {
    "mode": "claude",  // "claude" | "gemini" | "openrouter" | "local" | "mock"
    "claude": {
      "api_key": null,  // Uses ANTHROPIC_API_KEY env var if null
      "model_name": "claude-3-5-sonnet-20241022"
    },
    "gemini": {
      "api_key": null,
      "model_name": "gemini-2.0-flash-exp"
    },
    "local": {
      "model_path": "output/qwen3-movielens-qlora",
      "base_model": "Qwen/Qwen3-0.6B"
    }
  }
}
```

---

## 2. Enhanced Reflector with CF Auxiliary Matrix

### 2.1 Current Reflector Design

The existing Reflector:
- Takes first LLM attempt + ground truth
- Generates reflection on correctness
- Provides improvement guidance

### 2.2 Enhanced Design: Two-Stage Decision

```
┌─────────────────────────────────────────────────────────────────┐
│                    Enhanced Reflector                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Collaborative Filtering Signal                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  SwingSimilarity                                            │ │
│  │  ├── User-User Similarity Matrix                            │ │
│  │  ├── Item-Item Co-occurrence                                │ │
│  │  └── Output: CF Score (0.0 - 1.0)                          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                           │                                      │
│                           ▼                                      │
│  Stage 2: LLM Reasoning with CF Context                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Input:                                                     │ │
│  │  ├── First LLM Judgment (Yes/No + reasoning)                │ │
│  │  ├── CF Score + Similar Users' Behavior                     │ │
│  │  ├── User's Historical Patterns                             │ │
│  │  └── Candidate Item Features                                │ │
│  │                                                             │ │
│  │  Output:                                                    │ │
│  │  ├── Final Decision (Yes/No)                                │ │
│  │  ├── Confidence Score (0.0 - 1.0)                          │ │
│  │  └── Reasoning Explanation                                  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 CF Auxiliary Matrix Design

Based on the Delivery Hero dataset, we design the CF signal using multiple dimensions:

#### 2.3.1 User Embedding (Vector Representation)

```python
@dataclass
class UserVector:
    """User vector representation for CF."""
    user_id: str

    # Cuisine preferences (one-hot or normalized counts)
    cuisine_dist: Dict[str, float]  # {"pizza": 0.3, "burgare": 0.2, ...}

    # Temporal patterns
    day_dist: Dict[str, float]      # {"Mon": 0.1, "Tue": 0.15, ...}
    hour_dist: Dict[int, float]     # {17: 0.2, 18: 0.3, ...}

    # Price sensitivity
    avg_price: float
    price_variance: float

    # Activity level
    order_count: int
    recency_weight: float  # Higher for recent activity

def compute_user_vector(orders: List[OrderRecord]) -> UserVector:
    """Compute user vector from order history."""
    cuisine_counts = Counter(o.cuisine for o in orders)
    day_counts = Counter(o.day for o in orders)
    hour_counts = Counter(o.hour for o in orders)

    total = len(orders)

    return UserVector(
        user_id=generate_user_id(orders),
        cuisine_dist={k: v/total for k, v in cuisine_counts.items()},
        day_dist={k: v/total for k, v in day_counts.items()},
        hour_dist={k: v/total for k, v in hour_counts.items()},
        avg_price=mean(o.price for o in orders),
        price_variance=variance([o.price for o in orders]),
        order_count=total,
        recency_weight=compute_recency_weight(orders)
    )
```

#### 2.3.2 Item Embedding

```python
@dataclass
class ItemVector:
    """Item vector representation for CF."""
    product_id: str
    name: str
    cuisine: str
    price: float

    # Popularity metrics
    purchase_count: int
    unique_users: int

    # Co-purchase patterns
    co_purchased_items: Dict[str, float]  # {item_id: co-occurrence_score}
```

#### 2.3.3 User-Item Score Computation

```python
def compute_cf_score(
    user: UserVector,
    item: ItemVector,
    swing: SwingSimilarity
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute CF-based recommendation score.

    Returns:
        Tuple of (score, explanation_context)
    """
    score = 0.0
    context = {}

    # 1. Cuisine match (0.0 - 0.4)
    cuisine_score = user.cuisine_dist.get(item.cuisine, 0.0) * 0.4
    score += cuisine_score
    context['cuisine_match'] = cuisine_score / 0.4

    # 2. Price compatibility (0.0 - 0.2)
    price_diff = abs(item.price - user.avg_price)
    price_score = max(0, 0.2 - price_diff * 0.4)
    score += price_score
    context['price_compatibility'] = price_score / 0.2

    # 3. Similar users' behavior (0.0 - 0.3)
    similar_users = swing.get_similar_users(user.user_id)
    similar_purchased = 0
    total_similar = 0

    for sim_user, sim_score in similar_users:
        sim_items = swing.get_user_items(sim_user)
        # Check if similar users purchased this cuisine type
        if item.cuisine in [i.split('_')[0] for i in sim_items]:
            similar_purchased += sim_score
        total_similar += sim_score

    if total_similar > 0:
        similar_score = (similar_purchased / total_similar) * 0.3
        score += similar_score
        context['similar_users_score'] = similar_score / 0.3

    # 4. Item popularity boost (0.0 - 0.1)
    popularity_score = min(0.1, item.unique_users / 100 * 0.1)
    score += popularity_score
    context['popularity'] = popularity_score / 0.1

    return score, context
```

### 2.4 Enhanced Reflector Implementation

```python
class EnhancedReflector(Agent):
    """
    Reflector agent with CF auxiliary signals for hybrid decision making.

    Two-stage process:
    1. Compute CF score from behavioral patterns
    2. LLM reasoning with CF context to make final decision
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        swing_similarity: SwingSimilarity,
        config: Dict[str, Any] = None
    ):
        super().__init__(AgentType.REFLECTOR, llm_provider, config)

        self.swing = swing_similarity
        self.cf_weight = config.get('cf_weight', 0.3)  # Weight of CF in final decision
        self.user_vectors: Dict[str, UserVector] = {}
        self.item_vectors: Dict[str, ItemVector] = {}

    def forward(
        self,
        first_judgment: Dict[str, Any],  # {"decision": "Yes/No", "reasoning": "..."}
        user_orders: List[OrderRecord],
        candidate: CandidateProduct,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make final recommendation decision with CF + LLM reasoning.

        Args:
            first_judgment: Initial LLM decision and reasoning
            user_orders: User's order history
            candidate: Candidate product to evaluate

        Returns:
            Final decision with reasoning
        """
        # Stage 1: Compute CF Score
        user_vector = compute_user_vector(user_orders)
        item_vector = self._get_item_vector(candidate)

        cf_score, cf_context = compute_cf_score(
            user_vector, item_vector, self.swing
        )

        # Stage 2: LLM Reasoning with CF Context
        prompt = self._build_reflection_prompt(
            first_judgment=first_judgment,
            cf_score=cf_score,
            cf_context=cf_context,
            user_orders=user_orders,
            candidate=candidate
        )

        llm_response = self.llm_provider.generate(
            prompt,
            temperature=0.3,  # Lower for more deterministic
            json_mode=True
        )

        # Parse and return
        return self._parse_response(llm_response, cf_score, cf_context)

    def _build_reflection_prompt(
        self,
        first_judgment: Dict[str, Any],
        cf_score: float,
        cf_context: Dict[str, Any],
        user_orders: List[OrderRecord],
        candidate: CandidateProduct
    ) -> str:
        """Build prompt for LLM reflection with CF context."""

        # Format user history summary
        cuisine_summary = self._summarize_cuisines(user_orders)
        time_summary = self._summarize_time_patterns(user_orders)

        return f"""You are a recommendation reflector agent. Consider both the initial judgment and collaborative filtering signals to make the final recommendation decision.

## Initial LLM Judgment
Decision: {first_judgment.get('decision', 'Unknown')}
Reasoning: {first_judgment.get('reasoning', 'No reasoning provided')}

## Collaborative Filtering Analysis
CF Score: {cf_score:.3f} (0=unlikely, 1=highly likely)

Score Breakdown:
- Cuisine Match: {cf_context.get('cuisine_match', 0):.2%} (user frequently orders {candidate.cuisine})
- Price Compatibility: {cf_context.get('price_compatibility', 0):.2%}
- Similar Users: {cf_context.get('similar_users_score', 0):.2%} (users with similar taste ordered this)
- Item Popularity: {cf_context.get('popularity', 0):.2%}

## User Profile Summary
{cuisine_summary}
{time_summary}

## Candidate Product
- Name: {candidate.name}
- Cuisine: {candidate.cuisine}
- Price: ${candidate.price:.2f}

## Task
Based on:
1. The initial LLM judgment (semantic understanding)
2. The CF score (behavioral patterns from similar users)
3. The user's historical preferences

Make the FINAL recommendation decision.

Return JSON format:
{{
    "final_decision": "Yes" or "No",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation combining semantic and behavioral factors",
    "cf_contribution": "How CF signal influenced the decision",
    "override_reason": "If different from initial judgment, explain why" (optional)
}}"""
```

### 2.5 Integration with Manager

```python
class RecommendationPipeline:
    """
    Complete recommendation pipeline using hybrid CF-LLM approach.
    """

    def __init__(self, config: Dict[str, Any]):
        # Initialize LLM provider (Claude first priority)
        self.llm_provider = create_llm_provider(
            provider_type=config.get('llm_mode', 'claude')
        )

        # Initialize Swing similarity
        self.swing = SwingSimilarity(SwingConfig())

        # Initialize agents
        self.analyst = Analyst(self.llm_provider)
        self.reflector = EnhancedReflector(
            self.llm_provider,
            self.swing
        )

    def process_request(
        self,
        request: RecommendationRequest
    ) -> Dict[str, Any]:
        """
        Process single recommendation request.

        Flow:
        1. Analyst: Analyze user history and extract patterns
        2. First LLM judgment based on analysis
        3. Reflector: Combine CF score + LLM reasoning
        4. Return final decision
        """
        # Step 1: Analyst builds context
        analysis = self.analyst.analyze_user_sequence(request.orders)

        # Step 2: First judgment
        first_judgment = self._make_first_judgment(analysis, request.candidate)

        # Step 3: Enhanced reflection with CF
        final_result = self.reflector.forward(
            first_judgment=first_judgment,
            user_orders=request.orders,
            candidate=request.candidate
        )

        return final_result
```

---

## 3. Data Adapter Integration

### 3.1 Current State

- `food_delivery_adapter.py`: Parses JSONL into structured types
- `convert_to_jsonl.py`: Converts raw data to training format

### 3.2 Required Enhancements

#### 3.2.1 Build Interaction Index for Swing

```python
class InteractionIndexBuilder:
    """
    Build user-item interaction index from Delivery Hero data
    for Swing similarity computation.
    """

    @staticmethod
    def from_jsonl(
        jsonl_path: str,
        item_granularity: str = 'cuisine'  # 'cuisine' | 'product'
    ) -> List[Tuple[str, str]]:
        """
        Build interaction list from JSONL training data.

        Args:
            jsonl_path: Path to train.jsonl
            item_granularity:
                - 'cuisine': Group by cuisine type (recommended for sparse data)
                - 'product': Use individual product names

        Returns:
            List of (user_id, item_id) tuples
        """
        interactions = []

        with open(jsonl_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                request = parse_jsonl_record(record)

                user_id = request.user_id

                for order in request.orders:
                    if item_granularity == 'cuisine':
                        item_id = order.cuisine
                    else:
                        # Would need product name from original data
                        item_id = f"{order.cuisine}_{order.price:.2f}"

                    interactions.append((user_id, item_id))

        return interactions
```

#### 3.2.2 Chat Template Adapter

```python
class ChatTemplateAdapter:
    """
    Convert recommendation requests to chat format
    compatible with both fine-tuned Qwen3 and Claude.
    """

    SYSTEM_PROMPT = (
        "You are a food delivery recommendation assistant. "
        "Analyze the user's order history and decide if they would "
        "likely purchase the candidate product. Consider cuisine preferences, "
        "price patterns, and temporal habits."
    )

    @staticmethod
    def to_claude_messages(
        request: RecommendationRequest,
        include_cf_context: bool = False,
        cf_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """Format request for Claude API."""

        user_content = ChatTemplateAdapter._format_user_content(
            request, include_cf_context, cf_context
        )

        return [
            {"role": "user", "content": user_content}
        ]

    @staticmethod
    def to_qwen_messages(
        request: RecommendationRequest
    ) -> List[Dict[str, str]]:
        """
        Format request for Qwen3 chat template.
        Matches format used in finetune/scripts/utils.py
        """
        return [
            {"role": "system", "content": ChatTemplateAdapter.SYSTEM_PROMPT},
            {"role": "user", "content": format_order_history_table(request.orders) +
                f"\n\nCandidate product:\n{format_candidate_product(request.candidate)}"}
        ]
```

---

## 4. Analysis: Is This a Good Approach?

### 4.1 Strengths

1. **Complementary Signals**: CF captures behavioral patterns that LLMs miss from text alone
2. **Cold-Start Mitigation**: LLM semantic understanding helps when CF data is sparse
3. **Explainability**: Combined reasoning provides clear explanations
4. **Model-Agnostic**: CF component works with any pre-trained model
5. **Research-Backed**: Follows proven hybrid architectures (CoLLM, A-LLMRec)

### 4.2 Potential Challenges

1. **Computational Cost**: Two-stage inference increases latency
2. **CF Sparsity**: Delivery Hero data may have sparse user histories
3. **Feature Engineering**: User/item vectors require careful design
4. **Calibration**: Need to tune CF weight vs LLM confidence

### 4.3 Recommendations

1. **Start with cuisine-level CF** - More robust than product-level for sparse data
2. **Use Claude for reflection** - Best reasoning for combining signals
3. **Cache CF scores** - Precompute for batch evaluation
4. **A/B test weights** - Tune cf_weight parameter empirically
5. **Consider ensemble** - Final decision could weight both signals

### 4.4 Expected Impact

Based on research:
- **+3-5% accuracy** over pure LLM approach
- **Better cold-start handling** for new users
- **More explainable decisions** with CF context

---

## 5. Implementation Plan

### Phase 1: LLM Module (Week 1)
- [ ] Add ClaudeProvider to llm_provider.py
- [ ] Add QwenLocalProvider for fine-tuned models
- [ ] Update configuration schema
- [ ] Write provider tests

### Phase 2: CF Integration (Week 2)
- [ ] Build InteractionIndexBuilder
- [ ] Implement UserVector/ItemVector computation
- [ ] Integrate Swing with Reflector
- [ ] Write CF score computation tests

### Phase 3: Enhanced Reflector (Week 3)
- [ ] Implement EnhancedReflector class
- [ ] Design reflection prompts
- [ ] Integrate with Manager pipeline
- [ ] Write integration tests

### Phase 4: Evaluation (Week 4)
- [ ] Run on test set with CF vs without
- [ ] Measure accuracy, F1, and latency
- [ ] Tune cf_weight parameter
- [ ] Document results

---

## Appendix A: File Structure

```
agentic_recommender/
├── agents/
│   ├── base.py
│   ├── analyst.py
│   ├── reflector.py          # Enhanced with CF
│   ├── manager.py
│   └── doc/
│       └── AGENT_REDESIGN.md  # This document
├── models/
│   └── llm_provider.py        # Add Claude, Local providers
├── similarity/
│   └── swing.py               # Existing
├── data/
│   ├── food_delivery_adapter.py
│   ├── interaction_builder.py  # NEW
│   └── chat_template.py        # NEW
└── configs/
    └── config                  # Updated schema
```

## Appendix B: References

1. [Enhanced Recommendation Combining CF and LLM (ACM 2025)](https://dl.acm.org/doi/full/10.1145/3732801.3732809)
2. [CoLLM (IEEE TKDE 2025)](http://staff.ustc.edu.cn/~hexn/papers/tkde25-CoLLM.pdf)
3. [A-LLMRec (KDD 2024)](https://arxiv.org/abs/2404.11343)
4. [LLM4Rec Survey (MDPI 2025)](https://www.mdpi.com/1999-5903/17/6/252)
