# Fine-tuning Parameters and Performance

This document contains all the hyperparameters and performance metrics for the Qwen3-0.6B MovieLens fine-tuning experiment.

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| **Base Model** | Qwen/Qwen3-0.6B |
| **Model Architecture** | Qwen3ForCausalLM |
| **Template** | qwen |
| **Total Parameters** | 606,142,464 |
| **Trainable Parameters** | 10,092,544 |
| **Trainable Percentage** | 1.665% |
| **Context Length** | 2048 tokens |

### Model Architecture Details
- **Hidden Size**: 1024
- **Intermediate Size**: 3072
- **Number of Layers**: 28
- **Attention Heads**: 16
- **Key-Value Heads**: 8
- **Head Dimension**: 128
- **Vocabulary Size**: 151,936
- **Max Position Embeddings**: 40,960

---

## Training Method

| Parameter | Value |
|-----------|-------|
| **Fine-tuning Type** | LoRA (Low-Rank Adaptation) |
| **Quantization** | 4-bit (QLoRA) |
| **Stage** | Supervised Fine-Tuning (SFT) |

### LoRA Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **LoRA Rank** | 16 | Rank of the low-rank matrices |
| **LoRA Alpha** | 64 | Scaling factor (α/r = 4.0) |
| **LoRA Dropout** | 0.05 | Dropout rate for LoRA layers |
| **LoRA Target** | all | Apply LoRA to all linear layers |
| **Targeted Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Use RSLoRA** | false | Rank-stabilized LoRA disabled |

### Linear Modules Found
```
v_proj, gate_proj, k_proj, down_proj, o_proj, q_proj, up_proj
```

---

## Dataset Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | MovieLens 25M |
| **Task** | Binary movie recommendation |
| **Training Examples** | 78,271 |
| **Evaluation Examples** | 25 (max_eval_samples) |
| **Cutoff Length** | 1024 tokens |
| **Dataset Directory** | data |
| **Training Split** | movielens_qwen3_train |
| **Eval Split** | movielens_qwen3_eval |
| **Max Samples** | null (use all) |

---

## Training Hyperparameters

### Batch and Accumulation

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Per Device Train Batch Size** | 2 | Batch size per GPU |
| **Per Device Eval Batch Size** | 1 | Eval batch size per GPU |
| **Gradient Accumulation Steps** | 8 | Steps to accumulate before update |
| **Effective Batch Size** | 16 | per_device_batch_size × accumulation_steps × num_gpus |

### Learning Rate and Schedule

| Parameter | Value |
|-----------|-------|
| **Learning Rate** | 1.5e-4 (0.00015) |
| **LR Scheduler Type** | cosine |
| **Warmup Ratio** | 0.03 (3% of total steps) |
| **Warmup Steps** | ~880 steps |

### Optimization

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW |
| **Adam Beta1** | 0.9 |
| **Adam Beta2** | 0.999 |
| **Adam Epsilon** | 1.0e-8 |
| **Weight Decay** | 0.0 |
| **Max Gradient Norm** | 1.0 |

### Training Duration

| Parameter | Value |
|-----------|-------|
| **Number of Epochs** | 3 |
| **Total Training Steps** | ~29,352 (calculated: 78,271 samples / 16 effective batch × 3 epochs) |
| **Logging Steps** | 10 |
| **Evaluation Steps** | 2000 |
| **Save Steps** | 4000 |
| **Save Total Limit** | 2 (keep only last 2 checkpoints) |
| **Early Stopping Patience** | 3 evaluations |

---

## Precision and Performance

### Compute Precision

| Parameter | Value |
|-----------|-------|
| **FP16** | false |
| **BF16** | true |
| **Compute Dtype** | torch.bfloat16 |
| **Gradient Checkpointing** | false |
| **Flash Attention** | auto (torch SDPA used) |

### Memory Optimization

| Feature | Status |
|---------|--------|
| **Gradient Checkpointing** | ❌ Disabled (for faster training) |
| **KV Cache** | ❌ Disabled during training |
| **4-bit Quantization (QLoRA)** | ✅ Enabled (bfloat16 base) |
| **Trainable Params Upcasting** | ✅ Upcasted to float32 |

---

## Hardware and Distribution

| Parameter | Value |
|-----------|-------|
| **Device** | NVIDIA GeForce RTX 3090 |
| **CUDA Version** | 12.8 |
| **GPU Memory** | 24GB |
| **Process Rank** | 0 |
| **World Size** | 1 |
| **Distributed Training** | false |
| **Device ID** | cuda:0 |

---

## Training Performance

### Training Speed

| Metric | Value |
|--------|-------|
| **Average Training Speed** | 2.0-2.4 it/s |
| **Initial Speed** | 1.05 it/s (first step) |
| **Stabilized Speed** | 2.2-2.3 it/s (after warmup) |
| **Peak Speed** | 2.38 it/s |

### Evaluation Speed

| Metric | Value |
|--------|-------|
| **Eval Samples per Second** | ~8-9 samples/s (estimated) |
| **Eval Steps per Second** | ~8-9 steps/s (estimated) |
| **Eval Runtime** | ~3-4 seconds per eval (25 samples) |

### Time Estimates

| Metric | Value |
|--------|-------|
| **Time per Step** | ~0.42-0.50 seconds |
| **Steps per Hour** | ~7,200-8,640 steps |
| **Estimated Total Time** | 3.4-4.1 hours (at 2.0-2.4 it/s) |
| **Estimated Time per Epoch** | 1.1-1.4 hours |
| **Time per 1000 Steps** | ~7-8 minutes |

### Training Progress (as of latest checkpoint)

| Metric | Value |
|--------|-------|
| **Steps Completed** | 800+ |
| **Progress** | ~2.7% (based on config: ~29,352 total steps) |
| **Checkpoints Saved** | 4 (from previous run with different save_steps) |
| **Latest Checkpoint** | checkpoint-800 |

**Note**: The progress metrics above are from a previous training run. With current config (save_steps=4000), checkpoints will be saved at steps 4000, 8000, etc.

---

## Evaluation Metrics

### Eval Loss History

| Step | Epoch | Eval Loss |
|------|-------|-----------|
| 0 | 0.0 | 0.3672 |
| 200 | 0.01 | 0.6124 |
| 400 | 0.01 | 0.3735 |
| 600 | 0.01 | 0.5317 |

**Note**: The evaluation metrics above are from a previous training run with eval_steps=200. With current config (eval_steps=2000), evaluations will occur at steps 2000, 4000, 6000, etc.

---

## Output Configuration

| Parameter | Value |
|-----------|-------|
| **Output Directory** | output/qwen3-movielens-qlora |
| **Evaluation Strategy** | steps |
| **Reporting** | tensorboard |
| **DDP Timeout** | 180000000 ms (50 hours) |

### Output Files
- **Checkpoints**: `output/qwen3-movielens-qlora/checkpoint-{step}/`
- **Training Logs**: `output/qwen3-movielens-qlora/trainer_log.jsonl`
- **TensorBoard Logs**: `output/qwen3-movielens-qlora/runs/`

---

## Command Used

```bash
python scripts/finetune_lora.py --config configs/qwen3_movielens_qlora.yaml
```

---

## Key Findings

### Performance Characteristics
1. **Training Speed**: Stable at ~2.2-2.3 it/s after initial warmup
2. **Memory Efficiency**: 4-bit quantization enables training on 24GB GPU
3. **Parameter Efficiency**: Only 1.67% of parameters are trainable
4. **Evaluation Time**: Fast evaluation with max_eval_samples limited to 25

### Optimization Notes
- **Gradient Accumulation**: Set to 8 for effective batch size of 16
- **BF16 Training**: Using bfloat16 for stability with large learning rates
- **Gradient Checkpointing**: Disabled for faster training (VRAM allows it)
- **Flash Attention**: Using PyTorch SDPA for optimized attention computation
- **Eval Accumulation Steps**: Set to 1 to prevent OOM by moving predictions to CPU frequently

### Resource Utilization
- **GPU Memory Usage**: ~16-18GB out of 24GB
- **Training Efficiency**: Good utilization with single GPU setup
- **Checkpoint Size**: ~20MB per LoRA checkpoint (vs ~1.2GB for full model)

---

## Reproducibility

To reproduce this training run:

1. Use the config file: `configs/qwen3_movielens_qlora.yaml`
2. Use CUDA device with at least 18GB memory (24GB recommended)
3. Set random seed if deterministic results are needed

### Environment
- **Python**: 3.11
- **PyTorch**: 2.2.2
- **Transformers**: 4.57.1
- **LLaMA-Factory**: Latest from main branch
- **CUDA**: 12.8

---

## Checkpointing Behavior

LLaMA-Factory automatically saves:
- **Best Model**: Based on lowest eval_loss (automatically tracked)
- **Latest Checkpoints**: Last 2 checkpoints (configured by `save_total_limit=2`)
- **Resume Capability**: Can resume from any checkpoint if interrupted

### Checkpoint Contents
Each checkpoint includes:
- LoRA adapter weights
- Optimizer state
- Scheduler state
- Training step count
- Best metric tracking

**Automatic Best Model Saving**: The framework tracks the best evaluation loss and saves the corresponding checkpoint. If training is interrupted, the checkpoint with the lowest eval_loss is preserved.

---

*Document generated: 2025-12-05*
*Training run: gpu-run*
*Config: qwen3_movielens_qlora.yaml*


### packing the code
git archive --format=zip HEAD:finetune -o finetune.zip
