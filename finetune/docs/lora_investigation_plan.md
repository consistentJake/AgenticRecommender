# Plan: Fix LoRA MovieLens SFT/Inference Issues

## Goals
- Preserve assistant labels under `cutoff_len` and avoid silent data loss.
- Align training data prep, loss masking, and inference prompts/decoding.
- Add generative evaluation that reflects real deployment behavior.
- Validate the pipeline with integration tests.

## Work Plan
1) **Cutoff & Tokenization**
   - Switch to left-side truncation (or explicit prompt trimming) before `tokenize_func` so assistant spans survive when sequences exceed `cutoff_len`.
   - Add a regression test that constructs an over-length prompt and confirms assistant labels remain after truncation.
   - Re-run preprocessing cache or add cache-busting to avoid stale right-truncated datasets.

2) **Prompt Alignment**
   - Ensure inference uses the same system prompt as training (`system` field when present; fallback matches training default).
   - Harmonize generation template usage (`apply_chat_template(add_generation_prompt=True)`) with the training format.

3) **Decoding Constraints**
   - For batch inference, prefer deterministic decoding (`do_sample=False`, small `max_new_tokens`, EOS/stop sequences) to avoid rambly outputs that hurt `extract_answer`.
   - Optionally add an explicit stop token list to terminate after “Yes”/“No”.

4) **Loss & Metrics Alignment**
   - Keep `ignore_index=-100` for prompt tokens; verify label masking via tests (already added).
   - Add a small generative eval path in training (subset of eval set) to measure real accuracy/F1 with `generate`, mirroring inference settings.
   - Compare teacher-forced metrics vs generative metrics to detect drift.

5) **Integration Tests**
   - Add an integration test that:
     - Builds a tiny in-memory dataset → formats via chat template → tokenizes with `cutoff_len` set low.
     - Runs a forward pass with a tiny/fast model (or mocking logits) to confirm loss only uses assistant tokens.
     - Exercises `extract_answer` on generated text with `<think>` and verbose prefixes.
   - Add a smoke test for inference code path that feeds a fixture prompt through `generate` with mocked model outputs to ensure parsing and metric computation behave.

6) **Config & Caches**
   - Update configs (`cutoff_len`, truncation_side, inference decoding params) and document the changes.
   - Clear and rebuild preprocessing caches after truncation change.

## Deliverables
- Updated tokenization/truncation logic and aligned inference settings.
- Generative eval hook in training with metrics reported.
- New integration tests covering truncation, loss masking, and inference parsing.
- Documentation note in `README`/`docs` describing the prompt/decoding alignment and cache reset step.
