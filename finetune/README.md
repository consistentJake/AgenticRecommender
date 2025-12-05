# Finetuning Qwen3 with LoRA/QLoRA

This directory contains scripts and configs for LoRA (QLoRA on CUDA) finetuning of `Qwen/Qwen3-0.6B` on the MovieLens-style binary recommendation task.

## Prerequisites
- Install deps: `pip install -r finetune/requirements_llamafactory.txt` (or ensure `transformers`, `trl`, `peft`, `bitsandbytes`, `datasets`, `tensorboard` are available).
- Data: `finetune/data/movielens_qwen3/train.json` and `eval.json` (paths are resolved via `dataset_info.json` or the YAML config).

## Config
- Main config: `finetune/configs/qwen3_movielens_qlora.yaml`.
- Most knobs (model path, output_dir, batch sizes, cutoff_len, LoRA params, eval/save steps, gradient checkpointing, early stopping, etc.) are read from this YAML. Override or supply a different config via `--config`.

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

## Evaluation details
- Eval runs every `eval_steps` (defaults to 2000) or per the strategy set in YAML.
- Eval batch size defaults to 16; adjust in the YAML if VRAM allows.
- Metrics are computed from generated text normalized to `Yes`/`No`. To use a larger/smaller eval split, set `max_eval_samples` in the YAML.
