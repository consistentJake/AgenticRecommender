# Prompt & Workflow Improvement Plan

This plan outlines the concrete changes required to upgrade the current LLM-driven multi-agent prompts and supporting workflow before we modify any source code. It draws contrasts with the MACRec prompt templates in `previousWorks/MACRec/config/prompts` to show what will change.

## 1. Goals & Rationale
- Increase determinism of manager actions by constraining outputs to JSON and supplying richer context.
- Tighten analyst guidance so data tools are invoked before free-form reasoning, mirroring the MACRec command grammar.
- Standardise reflection signals to produce machine-parseable lessons for continual learning.
- Improve logging hooks and context propagation so downstream evaluators can map prompt decisions to dataset artefacts.

## 2. Manager Prompt Overhaul
**Current behaviour (`agentic_recommender/agents/manager.py:70`–`150`):**
- Thinking prompt is a free-form paragraph with no guardrails on structure.
- Action prompt lists available actions but does not enforce canonical syntax or include candidate metadata.
- Output parsing accepts loosely formatted strings, falling back to `Finish` if parsing fails.

**Reference prompt (`previousWorks/MACRec/config/prompts/manager_prompt/reflect_analyse_search.json`):**
- Uses template placeholders for task type, max steps, few-shot examples, and optional reflection memory blocks.
- Explicitly supports both text and JSON response formats with examples for each action.

**Planned changes:**
1. Replace ad-hoc string prompts with template-driven builders that:
   - Add candidate item table, ground-truth (when available), and recent analyst outputs to the thinking stage.
   - Enforce JSON action schema (`{"type": "Analyse", "content": ["user", "123"]}`) and provide inline examples.
   - Inject reflection snippets (if present) into the scratchpad, similar to `{reflections}` in MACRec templates.
2. Update `_parse_action` to prefer JSON decoding before regex, aligning with the new enforced format.
3. Lower `max_tokens` for actions and explicitly set `stop` sequences to avoid multi-action outputs.

## 3. Analyst Prompt Refinement
**Current behaviour (`agentic_recommender/agents/analyst.py:48`–`121`):**
- Analyst jumps directly to a descriptive prompt without iterating through the available tools.
- No command phase; the LLM cannot decide which resources to pull before reasoning.

**Reference prompt (`previousWorks/MACRec/config/prompts/agent_prompt/analyst.json`):**
- Implements a ReAct-style loop with `UserInfo`, `ItemInfo`, `UserHistory`, `ItemHistory`, and `Finish` commands plus few-shot guidance.

**Planned changes:**
1. Reintroduce a command/action loop: analyst generates structured tool calls (text or JSON), orchestrated by the `ToolAgent` base class.
2. Supply few-shot examples from the MACRec template to encourage diverse tool usage before summarising.
3. Require analyst to return a JSON object with fields `{patterns, sequential_behavior, recent_trends, recommendations}`, simplifying downstream parsing.

## 4. Reflector Consistency
**Current behaviour (`agentic_recommender/agents/reflector.py:54`–`124`):**
- Requests JSON via `json_mode=True` but accepts arbitrary output if decoding fails.
- Scratchpad fed into reflection lacks explicit tags delineating thoughts/actions.

**Planned changes:**
1. Wrap conversation history in a formatted table before passing to the reflector to improve signal clarity.
2. Adopt MACRec-style reflection schema (`{"correctness": bool, "reason": str, "improvement": str, "next_action": str}`) and hard-fail if keys are missing, triggering a retry.
3. Persist reflection summaries in a structured store for future prompt conditioning, similar to MACRec `{reflections}` usage.

## 5. Orchestrator & Logging Support
1. Propagate dataset metadata into `RecommendationRequest.context` (e.g., category tags) so manager prompts can reference richer features.
2. Record each prompt + response pair in the log for offline prompt tuning and ablation studies.
3. Add configurable `max_iterations` per request, exposing it through pipeline config for faster sweeps.

## 6. Implementation Sequencing
1. Introduce a prompt template registry (YAML/JSON) under `agentic_recommender/config/prompts` seeded with MACRec templates.
2. Refactor manager/analyst to load templates, render with Jinja (or minimal `.format`) using the new context fields.
3. Adjust unit/integration tests to validate parsing of JSON actions and analyst outputs.
4. Update documentation and logging formats to reflect the new structured exchanges.

These steps keep the system aligned with proven MACRec prompting strategies while tailoring the content to our sequential recommendation context. Once the user approves, we can proceed with the staged code changes.
