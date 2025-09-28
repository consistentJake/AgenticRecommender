#!/usr/bin/env python3
"""Minimal QLoRA finetuning script aligned with the Trade Mamba workflow."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer

from agentic_recommender.finetune.dataset_builder import (
    AgenticDatasetConfig,
    build_agentic_dataset,
)


LOG = logging.getLogger(__name__)


@dataclass
class ScriptArgs:
    model_name: str
    dataset_name: Optional[str]
    dataset_config: Optional[str]
    train_file: Optional[str]
    eval_file: Optional[str]
    data_format: Optional[str]
    text_column: str
    prompt_template: Optional[str]
    output_dir: str
    max_seq_length: int
    learning_rate: float
    num_train_epochs: float
    batch_size: int
    gradient_accumulation_steps: int
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    seed: int
    packing: bool
    report_to: str
    agentic_dataset: Optional[str]
    agentic_data_root: Optional[str]
    agentic_include_reasoning: bool
    agentic_negatives: int


def parse_args() -> ScriptArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-config")
    parser.add_argument("--train-file")
    parser.add_argument("--eval-file")
    parser.add_argument("--data-format", choices=["json", "jsonl", "parquet"], help="Format for local data files")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--prompt-template", help="Optional JSON string template supporting {text}")
    parser.add_argument("--output-dir", default="outputs/llama3-qlora")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--packing", action="store_true", help="Enable sequence packing via TRL")
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--agentic-dataset", help="Name of processed Agentic dataset (e.g., beauty)")
    parser.add_argument("--agentic-data-root", help="Directory containing processed Agentic splits")
    parser.add_argument("--agentic-include-reasoning", action="store_true", help="Include reasoning-style examples")
    parser.add_argument("--agentic-negatives", type=int, default=1, help="Negatives per positive example")
    args = parser.parse_args()
    return ScriptArgs(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        train_file=args.train_file,
        eval_file=args.eval_file,
        data_format=args.data_format,
        text_column=args.text_column,
        prompt_template=args.prompt_template,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        packing=args.packing,
        report_to=args.report_to,
        agentic_dataset=args.agentic_dataset,
        agentic_data_root=args.agentic_data_root,
        agentic_include_reasoning=args.agentic_include_reasoning,
        agentic_negatives=args.agentic_negatives,
    )


def load_training_corpus(opts: ScriptArgs) -> DatasetDict:
    if opts.agentic_dataset:
        data_root = Path(opts.agentic_data_root) if opts.agentic_data_root else Path("agentic_recommender/data/outputs")
        config = AgenticDatasetConfig(
            name=opts.agentic_dataset,
            data_root=data_root,
            include_reasoning=opts.agentic_include_reasoning,
            negatives_per_positive=opts.agentic_negatives,
            seed=opts.seed,
        )
        return build_agentic_dataset(config)
    if opts.dataset_name:
        LOG.info("Loading dataset %s", opts.dataset_name)
        dataset = load_dataset(opts.dataset_name, opts.dataset_config)
    elif opts.train_file:
        if not opts.data_format:
            raise ValueError("--data-format is required when using local files")
        files: Dict[str, str] = {"train": opts.train_file}
        if opts.eval_file:
            files["validation"] = opts.eval_file
        dataset = load_dataset(opts.data_format, data_files=files)
    else:
        raise ValueError("Provide --dataset-name or --train-file")
    if isinstance(dataset, DatasetDict):
        return dataset
    raise ValueError("Dataset must expose named splits; check dataset configuration")


def maybe_apply_prompt_template(dataset: DatasetDict, column: str, template: Optional[str]) -> DatasetDict:
    if not template:
        return dataset
    try:
        template_obj = json.loads(template)
        prefix: str = template_obj.get("prefix", "")
        suffix: str = template_obj.get("suffix", "")
    except json.JSONDecodeError:
        prefix, suffix = template.split("{text}") if "{text}" in template else (template, "")

    def _formatter(example: Dict[str, str]) -> Dict[str, str]:
        value = example[column]
        example[column] = f"{prefix}{value}{suffix}"
        return example

    for split in dataset.keys():
        dataset[split] = dataset[split].map(_formatter)
    return dataset


def prepare_model(opts: ScriptArgs):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        opts.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=opts.lora_rank,
        lora_alpha=opts.lora_alpha,
        lora_dropout=opts.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    return model


def prepare_tokenizer(opts: ScriptArgs):
    tokenizer = AutoTokenizer.from_pretrained(opts.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


class LogPerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            ppl = torch.exp(torch.tensor(metrics["eval_loss"]))
            LOG.info("Eval perplexity %.2f", ppl.item())
        return control


def ensure_output_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    opts = parse_args()
    set_seed(opts.seed)

    ensure_output_dir(opts.output_dir)

    dataset = load_training_corpus(opts)
    dataset = maybe_apply_prompt_template(dataset, opts.text_column, opts.prompt_template)

    tokenizer = prepare_tokenizer(opts)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = prepare_model(opts)

    training_args = TrainingArguments(
        output_dir=opts.output_dir,
        per_device_train_batch_size=opts.batch_size,
        per_device_eval_batch_size=opts.batch_size,
        gradient_accumulation_steps=opts.gradient_accumulation_steps,
        learning_rate=opts.learning_rate,
        num_train_epochs=opts.num_train_epochs,
        weight_decay=0.0,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if "validation" in dataset else "no",
        bf16=True,
        report_to=[opts.report_to] if opts.report_to else [],
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        tokenizer=tokenizer,
        dataset_text_field=opts.text_column,
        max_seq_length=opts.max_seq_length,
        packing=opts.packing,
        data_collator=data_collator,
    )
    trainer.add_callback(LogPerplexityCallback())

    trainer.train()

    model.save_pretrained(os.path.join(opts.output_dir, "adapter"))
    tokenizer.save_pretrained(os.path.join(opts.output_dir, "adapter"))


if __name__ == "__main__":
    main()
