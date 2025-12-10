# MovieLens Recommendation Fine-tuning

Fine-tune LLMs (Qwen3-0.6B and Qwen3-8B) on the MovieLens 25M dataset for binary movie recommendation using LLaMA-Factory with LoRA/QLoRA.

**Status**: ✅ Training infrastructure ready | 🚀 Training in progress | 📊 Checkpoint saving enabled

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data (downloads MovieLens 25M)
python scripts/prepare_movielens.py

# 3. Start training
python scripts/train_llamafactory.py \
  --config configs/qwen3_movielens_qlora.yaml \
  --run-name experiment-1 \
  --device cuda

# 4. Monitor with TensorBoard
tensorboard --logdir output/qwen3-movielens-qlora/runs
```

💡 **See `commands` file for all available commands and examples**

📖 **See `TRAINING_PARAMETERS.md` for detailed hyperparameters and performance metrics**

## Project Structure

```
.
├── configs/                  # Training configuration files
│   ├── qwen3_movielens_qlora.yaml      # Qwen3-0.6B with QLoRA (GPU)
│   ├── qwen3_8b_movielens_qlora.yaml   # Qwen3-8B with QLoRA (GPU)
│   └── qwen3_movielens_cpu.yaml        # Qwen3-0.6B CPU config
├── data/                     # Dataset directory
│   ├── dataset_info.json    # Dataset metadata for LLaMA-Factory
│   └── movielens_qwen3/     # Prepared MovieLens data
│       ├── train.json       # Training set (78,271 examples)
│       ├── eval.json        # Evaluation set
│       └── test_raw.jsonl   # Raw test data
├── scripts/
│   ├── prepare_movielens.py        # Data preparation script
│   └── train_llamafactory.py       # Training wrapper script
├── output/                   # Training outputs (created during training)
│   └── qwen3-movielens-qlora/
│       ├── checkpoint-*/     # Model checkpoints (saved every 200 steps)
│       ├── trainer_log.jsonl # Training metrics
│       └── runs/            # TensorBoard logs
└── requirements.txt          # Python dependencies

```

## Setup

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA support (for GPU training)
- 24GB+ GPU memory recommended for Qwen3-0.6B with QLoRA

### Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

Prepare the MovieLens dataset for training:

```bash
python scripts/prepare_movielens.py
```

This script:
- Downloads MovieLens 25M dataset
- Processes ratings and movie metadata
- Creates training/eval splits formatted for Qwen3
- Generates negative samples for binary classification
- Outputs data to `data/movielens_qwen3/`

## Training

### GPU Training (Recommended)

Train Qwen3-0.6B with QLoRA on GPU:

```bash
python scripts/train_llamafactory.py \
  --config configs/qwen3_movielens_qlora.yaml \
  --run-name my-experiment \
  --device cuda
```

For smaller batch sizes (if running out of memory):

```bash
python scripts/train_llamafactory.py \
  --config configs/qwen3_movielens_qlora.yaml \
  --run-name my-experiment \
  --device cuda \
  --override per_device_train_batch_size=1 \
  --override gradient_accumulation_steps=1
```

### CPU Training (Slow, for testing only)

```bash
python scripts/train_llamafactory.py \
  --config configs/qwen3_movielens_cpu.yaml \
  --run-name cpu-test \
  --device cpu
```

### Resuming Training

If training is interrupted, you can resume from the latest checkpoint:

#### Option 1: Auto-detect Latest Checkpoint

```bash
python scripts/train_llamafactory.py \
  --config configs/qwen3_movielens_qlora.yaml \
  --run-name experiment-1 \
  --device cuda \
  --override per_device_train_batch_size=1 \
  --override gradient_accumulation_steps=1 \
  --override resume_from_checkpoint=true
```

#### Option 2: Specify Checkpoint Path

```bash
python scripts/train_llamafactory.py \
  --config configs/qwen3_movielens_qlora.yaml \
  --run-name experiment-1 \
  --device cuda \
  --override per_device_train_batch_size=1 \
  --override gradient_accumulation_steps=1 \
  --override resume_from_checkpoint=output/qwen3-movielens-qlora/checkpoint-1000
```

#### List Available Checkpoints

```bash
ls -lh output/qwen3-movielens-qlora/checkpoint-*/
```

**Note**: The training automatically saves:
- The last 2 checkpoints (configurable via `save_total_limit`)
- Optimizer and scheduler states for seamless resumption
- Step count to continue from the exact position

### Training Configuration

The default configuration (`qwen3_movielens_qlora.yaml`):
- **Model**: Qwen/Qwen3-0.6B
- **Method**: LoRA fine-tuning with QLoRA (4-bit quantization)
- **LoRA rank**: 16, alpha: 64
- **Training**: 3 epochs, batch size 1, gradient accumulation 8
- **Learning rate**: 1.5e-4 with cosine schedule
- **Precision**: bfloat16
- **Gradient checkpointing**: Enabled
- **Checkpoints**: Saved every 200 steps

## Monitoring Training

### TensorBoard

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir output/qwen3-movielens-qlora/runs --port 6006
```

Then open http://localhost:6006 in your browser.

### GPU Monitoring

Check GPU usage:

```bash
# One-time check
nvidia-smi

# Continuous monitoring (updates every second)
watch -n 1 nvidia-smi
```

### Training Logs

View training logs in real-time:

```bash
# Follow training logs
tail -f output/qwen3-movielens-qlora/trainer_log.jsonl

# View last 50 lines
tail -50 output/qwen3-movielens-qlora/trainer_log.jsonl
```

### Checkpoint Management

List available checkpoints:

```bash
# List all checkpoints
ls -lh output/qwen3-movielens-qlora/checkpoint-*/

# Check specific checkpoint contents
ls -lh output/qwen3-movielens-qlora/checkpoint-1000/
```

## Results

Training outputs are saved to the `output/` directory:

- **Model checkpoints**: `output/qwen3-movielens-qlora/checkpoint-{step}/`
  - Saved every 200 steps
  - Only the latest 2 checkpoints are kept (configurable via `save_total_limit`)

- **Training logs**: `output/qwen3-movielens-qlora/trainer_log.jsonl`
  - Contains loss, learning rate, and other metrics per step

- **TensorBoard logs**: `output/qwen3-movielens-qlora/runs/`
  - For visualization in TensorBoard

### Training Progress

- **Total examples**: 78,271
- **Total steps**: 234,813 (3 epochs)
- **Evaluation**: Every 200 steps
- **Training speed**: ~2.0-2.4 it/s on RTX 3090
- **Estimated time**: 27-33 hours for full 3 epochs

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Training Speed** | 2.0-2.4 iterations/second |
| **Eval Speed** | 7.86-8.82 samples/second |
| **Time per Step** | ~0.43-0.50 seconds |
| **GPU Memory Usage** | ~16-18GB / 24GB |
| **Trainable Parameters** | 10.09M (1.67% of total) |
| **Total Parameters** | 606.14M |
| **Checkpoint Size** | ~20MB per LoRA adapter |

**Hardware**: NVIDIA GeForce RTX 3090 (24GB VRAM) | CUDA 12.8

## Task Format

The model is trained on a binary movie recommendation task:

**Input format**:
```
Predict whether the user will like the candidate movie. Answer only with Yes or No.

User's last 15 watched movies:
1. Movie Title (Year) (rating ≈ X.X)
2. ...
15. ...

Candidate movie:
Movie Title (Year)

Should we recommend this movie to the user? Answer Yes or No.
```

**Output**: `Yes` or `No`

## Customization

### Override Configuration

You can override any config parameter from the command line:

```bash
python scripts/train_llamafactory.py \
  --config configs/qwen3_movielens_qlora.yaml \
  --run-name custom-run \
  --override learning_rate=2e-4 \
  --override num_train_epochs=5 \
  --override lora_rank=32
```

### Custom Output Directory

```bash
python scripts/train_llamafactory.py \
  --config configs/qwen3_movielens_qlora.yaml \
  --output-dir my_custom_output
```

## Troubleshooting

### Out of Memory

If you run out of GPU memory:
1. Reduce batch size: `--override per_device_train_batch_size=1`
2. Increase gradient accumulation: `--override gradient_accumulation_steps=16`
3. Use the Qwen3-0.6B model instead of 8B

### Missing TensorBoard

If you get a TensorBoard error:
```bash
pip install tensorboard
```

### CPU Training Issues

CPU training is not recommended for production. It's extremely slow and doesn't support:
- QLoRA/quantization
- bfloat16 precision (use fp32 instead)

## Model Checkpoints

Trained model checkpoints can be loaded for:
- Inference
- Continued training
- Merging with base model
- Deployment

Use LLaMA-Factory's export or inference tools to work with the checkpoints.

## Documentation

### Project Files

| File | Description |
|------|-------------|
| **README.md** | This file - project overview and usage guide |
| **TRAINING_PARAMETERS.md** | Complete hyperparameters and performance metrics |
| **commands** | Quick reference for all training commands |
| **configs/*.yaml** | Training configuration files |
| **requirements.txt** | Python dependencies |

### Key Features

✅ **Automatic Checkpointing**: Saves every 200 steps with full training state
✅ **Resume Training**: Can resume from any checkpoint after interruption
✅ **Best Model Tracking**: Automatically saves best model by eval loss
✅ **TensorBoard Integration**: Real-time training visualization
✅ **Memory Efficient**: 4-bit QLoRA enables training on 24GB GPU
✅ **Parameter Efficient**: Only 1.67% parameters trainable (10M out of 606M)

### Tips and Best Practices

1. **Monitor GPU Usage**: Use `watch -n 1 nvidia-smi` to track GPU memory
2. **Use TensorBoard**: Essential for monitoring training dynamics
3. **Save Your Run Name**: Keep track of experiments with meaningful names
4. **Backup Checkpoints**: Copy important checkpoints outside output dir
5. **Test First**: Try with `--override num_train_epochs=0.1` for quick validation
6. **Adjust Batch Size**: If OOM, reduce batch size or increase gradient accumulation

## Citation

If you use this work, please cite:
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen](https://github.com/QwenLM/Qwen)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)

## License

This project follows the licenses of its dependencies:
- LLaMA-Factory: Apache 2.0
- Qwen models: Check model card on Hugging Face
- MovieLens dataset: Check GroupLens license
