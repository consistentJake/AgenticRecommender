# Project Architecture

This document describes the file and directory structure of the Qwen3 LoRA/QLoRA finetuning project for MovieLens-based movie recommendation.

## Project Overview

This project finetunes the `Qwen/Qwen3-0.6B` model using LoRA (Low-Rank Adaptation) on a binary movie recommendation task derived from the MovieLens dataset. The goal is to train the model to predict whether a user will like a candidate movie based on their viewing history.

## Directory Structure

```
/home/workplace/finetune/
├── scripts/              # Training and data processing scripts
├── configs/              # YAML configuration files
├── data/                 # Processed training/evaluation datasets
├── output/               # Model checkpoints and training artifacts
├── docs/                 # Documentation and development notes
├── dev_data/             # Development datasets
├── .cache/               # Cached preprocessed datasets
├── README.md             # Main documentation with setup and usage
├── training.log          # Training execution log
├── requirements_sft_only_add_new_dep.txt  # Python dependencies
└── ARCHITECTURE.md       # This file
```

## Core Scripts (`/scripts`)

### 1. `finetune_lora.py`
**Purpose:** Main training script for LoRA/QLoRA finetuning

**Key Features:**
- Loads Qwen3-0.6B model with optional 4-bit quantization (QLoRA on CUDA)
- Configures LoRA adapters for efficient parameter-efficient finetuning
- Implements supervised fine-tuning using TRL's SFTTrainer
- Supports multiprocessing for 4-5x faster tokenization
- Dataset caching to avoid re-tokenization on subsequent runs
- Early stopping with configurable patience
- Evaluation metrics: accuracy and F1 score
- Flash Attention 2 support for 2-4x training speedup

**Configuration:** Loads settings from YAML config file (default: `configs/qwen3_movielens_qlora.yaml`)

**Usage:**
```bash
python scripts/finetune_lora.py --config configs/qwen3_movielens_qlora.yaml
```

**Training Phases:**
1. Model loading with quantization
2. Dataset preprocessing with multiprocessing
3. Training loop with periodic evaluation
4. Checkpoint saving

**Output:** Saves LoRA adapter weights and tokenizer to `output/qwen3-movielens-qlora/`

### 2. `prepare_movielens.py`
**Purpose:** Data preprocessing pipeline for MovieLens dataset

**Key Features:**
- Downloads MovieLens "latest-small" dataset automatically (optional)
- Builds sequential recommendation examples from user interaction history
- Converts to Alpaca/Qwen-style chat format for training
- Splits data into train/validation/test sets
- Generates labels: "Yes" for ratings ≥ 4.0, "No" otherwise

**Process Flow:**
1. Download/load MovieLens CSV files (ratings.csv, movies.csv)
2. Sort user interactions by timestamp
3. Create sequential history windows (default: 15 movies)
4. Generate candidate movie predictions
5. Split timeline: train → validation → test
6. Format as chat-style JSON records

**Output Files:**
- `data/movielens_qwen3/train.json` - Training examples
- `data/movielens_qwen3/eval.json` - Validation examples
- `data/movielens_qwen3/test_raw.jsonl` - Raw test data
- `data/movielens_qwen3/meta.json` - Dataset metadata

**Usage:**
```bash
python scripts/prepare_movielens.py \
    --output-dir data/movielens_qwen3 \
    --history-len 15 \
    --rating-threshold 4.0
```

### 3. `check_seq_len.py`
**Purpose:** Sequence length analysis tool

**Key Features:**
- Analyzes tokenized sequence lengths in training data
- Reports length statistics: min, max, mean, percentiles
- Helps validate that sequences fit within model's context window
- Can measure character counts instead of tokens (--char-only mode)

**Usage:**
```bash
python scripts/check_seq_len.py \
    --config configs/qwen3_movielens_qlora.yaml \
    --percentiles 50,90,95,99,100
```

**Output:** Statistical summary of sequence lengths to ensure data fits within max_seq_len limits

### 4. `infer_lora.py`
**Purpose:** Inference comparison script

**Key Features:**
- Loads base Qwen3-0.6B model
- Loads and merges LoRA adapter weights
- Compares predictions from base model vs finetuned model
- Uses same quantization settings as training for consistency

**Usage:**
```bash
python scripts/infer_lora.py
```

**Example Output:** Side-by-side comparison showing how finetuning improves recommendation quality

### 5. `deprecated/`
**Purpose:** Contains deprecated scripts no longer in active use


## Output Directory (`/output`)

### `qwen3-movielens-qlora/`
**Purpose:** Training outputs and model checkpoints

**Structure:**
```
output/qwen3-movielens-qlora/
├── adapter_config.json      # LoRA adapter configuration
├── adapter_model.safetensors # LoRA adapter weights
├── tokenizer.json           # Tokenizer vocabulary
├── tokenizer_config.json    # Tokenizer settings
├── checkpoint-{step}/       # Periodic training checkpoints
│   ├── adapter_model.safetensors
│   ├── optimizer.pt
│   ├── scheduler.pt
│   └── trainer_state.json
└── logs/                    # TensorBoard logs
    └── events.out.tfevents.*
```

**Checkpointing:**
- Saves every `save_steps` (default: 2000 steps)
- Keeps only `save_total_limit` checkpoints (default: 2)
- Best checkpoint loaded at end via early stopping

**Monitoring:** Use TensorBoard to view training progress:
```bash
tensorboard --logdir output/qwen3-movielens-qlora/logs
```

## Cache Directory (`/.cache`)

### `preprocessed/`
**Purpose:** Cached tokenized datasets for faster training restarts

**Structure:**
```
.cache/preprocessed/
├── train_preprocessed/      # Cached training data
│   ├── dataset_info.json
│   ├── data-00000-of-00001.arrow
│   └── state.json
└── eval_preprocessed/       # Cached evaluation data
    ├── dataset_info.json
    ├── data-00000-of-00001.arrow
    └── state.json
```

**Benefits:**
- Avoids re-tokenizing datasets on subsequent training runs
- Saves 4-5x preprocessing time with multiprocessing
- Automatically regenerated if source data changes

## Key Documentation Files

### `README.md`
**Purpose:** Main project documentation

**Contents:**
- Installation instructions
- Prerequisites (Python packages, Flash Attention 2)
- Dataset preparation steps
- Training workflow
- Monitoring and debugging tips
- Troubleshooting guide
- Training phases explanation
- OOM (Out of Memory) solutions

### `training.log`
**Purpose:** Complete training execution log

**Contains:**
- Training progress output
- Loss curves
- Evaluation metrics
- Error messages and warnings
- Checkpoint save confirmations

**Usage:** Reference for debugging training issues and tracking progress

### `TRAINING_PARAMETERS.md`
**Purpose:** Detailed explanation of training hyperparameters

**Contents:**
- Parameter descriptions
- Tuning recommendations
- Memory/speed trade-offs