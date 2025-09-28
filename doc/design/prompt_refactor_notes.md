# Prompt Refactor Implementation Notes

The following changes implement the plan outlined in `prompt_improvement_plan.md`.

- **Template registry**: Structured prompt JSON files now live under `agentic_recommender/configs/prompts`. They are loaded lazily via `configs/prompt_loader.py` and cached in memory for reuse.
- **Manager agent**: `agents/manager.py` renders the thinking/action prompts from the new templates, injects dataset-aware context such as candidate display names, and logs prompt/response pairs for traceability.
- **Analyst agent**: `agents/analyst.py` now runs a ReAct-style command loop, issuing JSON commands back to the tool bridge until a structured report is produced. Each turn is logged with prompt and raw model output.
- **Reflector agent**: `agents/reflector.py` enforces a JSON reflection schema (`correctness`, `reason`, `improvement`, `next_action`) with retries and records prompt/response details.
- **Pipeline + orchestrator**: additional metadata (candidate names, user profile hints) flows with every `RecommendationRequest`, and `RecommendationPipeline` exposes the configurable `max_iterations` knob.

These notes accompany the implementation for future contributors who extend the prompting system.
