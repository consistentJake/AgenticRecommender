# Finetuning Qwen3 with LoRA/QLoRA

This directory contains scripts and configs for LoRA (QLoRA on CUDA) finetuning of `Qwen/Qwen3-0.6B` on the MovieLens-style binary recommendation task.

## Prerequisites
- Install deps: `pip install -r finetune/requirements_llamafactory.txt` (or ensure `transformers`, `trl`, `peft`, `bitsandbytes`, `datasets`, `tensorboard` are available).
- Data: `finetune/data/movielens_qwen3/train.json` and `eval.json` (paths are resolved via `dataset_info.json` or the YAML config).

## Config
- Main config: `finetune/configs/qwen3_movielens_qlora.yaml`.
- Most knobs (model path, output_dir, batch sizes, cutoff_len, LoRA params, eval/save steps, gradient checkpointing, early stopping, etc.) are read from this YAML. Override or supply a different config via `--config`.

## Training Best Practices

Our training configuration follows these optimized guidelines:

### 1. Evaluation Strategy
- **Eval frequency**: `eval_steps = 2000` to balance monitoring with training efficiency
  - Reduces evaluation overhead while maintaining visibility into model progress
  - Can also use `eval_strategy = "epoch"` for smaller datasets
- **Eval batch size**: `per_device_eval_batch_size = 16` (maximize based on VRAM)
  - Larger batches speed up evaluation significantly
- **Eval dataset size**: Limited to 1000 samples (`max_eval_samples = 1000`) for quick frequent checks
  - Keeps eval time reasonable while providing reliable metrics

### 2. Batch Size & Gradient Accumulation
- **Train batch size**: `per_device_train_batch_size = 2` (can increase to 4 if VRAM allows)
- **Gradient accumulation**: `gradient_accumulation_steps = 8`
- **Effective batch size**: 16-32 samples
  - Provides training stability without excessive memory usage
  - Balances convergence speed with resource constraints

### 3. Sequence Length & Memory Optimization
- **Max sequence length**: `cutoff_len = 1024` (set close to p95 data length)
  - Avoids wasting compute on excessive padding
  - Adjust based on your actual data distribution
- **Gradient checkpointing**: `gradient_checkpointing = false`
  - Disabled for faster training when VRAM permits
  - Enable (`true`) if encountering OOM errors

### 4. Learning Rate & Schedule
- **Learning rate**: `learning_rate = 1.5e-4` (optimal for LoRA fine-tuning)
- **Warmup**: `warmup_ratio = 0.03` (gradual ramp-up prevents early instability)
- **Schedule**: `lr_scheduler_type = cosine` (smooth decay)
- **Optimizer**: AdamW with standard betas (0.9, 0.999)

### 5. Metrics & Early Stopping
- **Metrics computed**: Loss, Accuracy, F1 score
  - Accuracy measures exact match on Yes/No predictions
  - F1 score handles class imbalance in binary recommendations
- **Early stopping**: `early_stopping_patience = 3`
  - Stops training if eval metrics plateau/degrade
  - Prevents overfitting and saves compute time

### 6. LoRA Configuration
- **Rank**: `lora_rank = 16` (good balance of capacity vs. efficiency)
- **Alpha**: `lora_alpha = 64` (4x rank for stable training)
- **Dropout**: `lora_dropout = 0.05` (light regularization)
- **Target modules**: All linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)

## Training
- Run:
  ```bash
  python finetune/scripts/finetune_lora.py --config finetune/configs/qwen3_movielens_qlora.yaml
  ```
- Outputs (adapter + tokenizer + logs) land in `output/qwen3-movielens-qlora` by default.
- Training/eval metrics include loss, accuracy, and F1 on the Yes/No labels. Early stopping patience is configurable in the YAML.

## Inference
- Compare base vs adapter:
  ```bash
  python finetune/scripts/infer_lora.py
  ```
- `ADAPTER_DIR` in the script points at the default training output. Adjust if you train to a different `output_dir`.

## Monitoring (TensorBoard)
- Training logs are written to `<output_dir>/logs`. Start TensorBoard:
  ```bash
  tensorboard --logdir output/qwen3-movielens-qlora/logs --port 6006
  ```
- Open http://localhost:6006 to view loss curves, accuracy/F1, and eval checkpoints. Point `--logdir` to a parent folder (e.g., `output`) to compare multiple runs.

## Quick sequence-length audit
- Token-based (uses the same tokenizer as training):
  ```bash
  python finetune/scripts/check_seq_len.py --config finetune/configs/qwen3_movielens_qlora.yaml --max-samples 2000
  ```
- Char-only (no model/tokenizer download):
  ```bash
  python finetune/scripts/check_seq_len.py --config finetune/configs/qwen3_movielens_qlora.yaml --char-only --max-samples 2000
  ```
- Reports min/max/mean and percentiles (p50/p90/p95/p99/p100) for the train split using the same chat template formatting. Use this to set `cutoff_len` near your p95 length.

## Tuning for Your Hardware
- **Low VRAM (8GB)**: Keep `per_device_train_batch_size = 1-2`, enable `gradient_checkpointing = true`
- **Medium VRAM (16GB)**: Use `per_device_train_batch_size = 2-4`, can disable gradient checkpointing
- **High VRAM (24GB+)**: Increase batch sizes further and eval batch size to 32+
