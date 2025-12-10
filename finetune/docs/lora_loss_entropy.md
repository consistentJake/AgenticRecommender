# How LoRA training in `scripts/finetune_lora.py` computes loss (and where "entropy" fits)

This project fine-tunes a causal LM with LoRA adapters using Hugging Face TRL's `SFTTrainer`. The key steps that lead to the loss calculation are:

1) **Label construction (masked prompt tokens)**  
   - In `preprocess_datasets_parallel`, the tokenized prompt portion is masked to `-100` in `labels`, while the assistant response tokens keep their token IDs.  
   - The Hugging Face Trainer skips positions where `labels == -100`, so only assistant tokens contribute to the loss.

2) **Forward pass with LoRA adapters**  
   - LoRA adapters are attached via `peft.LoraConfig` and injected by TRL/SFTTrainer. Base weights stay frozen; only the low-rank LoRA parameters update.  
   - If `use_qlora` is enabled (CUDA), the base model runs in 4-bit (NF4) with bfloat16 compute; LoRA adapters stay in higher precision.

3) **Loss function (cross-entropy)**  
   - The trainer uses the model’s `forward` to obtain logits, then applies standard token-level cross-entropy with label smoothing disabled.  
   - Reduction is the mean over all unmasked tokens in the batch. This is what the training loop logs as `loss`.

4) **Entropy vs. cross-entropy**  
   - The script does **not** explicitly compute an entropy metric. Any “entropy” you might see is implicitly the per-token negative log-likelihood term inside the cross-entropy loss.  
   - No additional entropy regularizer or KL term is present.

5) **Evaluation metrics**  
   - `compute_metrics` converts logits to token IDs (via `preprocess_logits_for_metrics`), decodes only assistant tokens, normalizes to yes/no, and reports accuracy and F1. It does not use entropy.

Key takeaways:
- Loss = mean cross-entropy over assistant tokens only; prompt tokens are masked out.  
- LoRA updates only adapter weights; the frozen base provides the logits used in loss.  
- There is no separate entropy metric or regularizer in this script—only cross-entropy-derived loss plus accuracy/F1 for eval.  
- Early stopping callbacks monitor training loss and eval metrics, not entropy.
