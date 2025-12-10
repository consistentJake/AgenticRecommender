# Claude Project Guide

## Project Architecture

**IMPORTANT:** To understand the complete file structure, directory organization, and purpose of each component in this project, read `/home/workplace/finetune/ARCHITECTURE.md`.

The ARCHITECTURE.md file contains:
- Complete directory structure overview
- Detailed descriptions of all core scripts in `/scripts`
- Output directory structure and checkpointing information
- Cache directory organization
- Key documentation files and their purposes

## Quick Reference

This is a Qwen3-0.6B LoRA/QLoRA finetuning project for MovieLens-based movie recommendation.

**Key Components:**
- `scripts/finetune_lora.py` - Main training script
- `scripts/prepare_movielens.py` - Data preprocessing pipeline
- `configs/` - YAML configuration files
- `data/` - Processed training/evaluation datasets
- `output/qwen3-movielens-qlora/` - Model checkpoints and training artifacts
- `.cache/preprocessed/` - Cached tokenized datasets

**Important Documentation:**
- `/home/workplace/finetune/ARCHITECTURE.md` - Complete architecture and file structure
- `/home/workplace/finetune/README.md` - Setup, usage, and troubleshooting
- `/home/workplace/finetune/TRAINING_PARAMETERS.md` - Hyperparameter explanations
- `/home/workplace/finetune/DATA_SCHEMA.md` - Data format and schema details

## Working with this Project

When assisting with this project, always:
1. Consult ARCHITECTURE.md first to understand component relationships
2. Check existing documentation before making assumptions
3. Follow the established patterns in the codebase
4. Consider memory constraints when modifying training parameters
