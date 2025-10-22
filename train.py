#!/usr/bin/env python3
"""
Fine-tune a causal language model for the CLI programming agent.

The script expects pre-tokenized text corpora stored as plain UTF-8 files for
training and (optionally) evaluation. The data should be a mix of source code,
StackOverflow Q&A, and technical documentation, concatenated with newlines.

Example:
    python train.py \
        --model-name gpt2-medium \
        --train-data data/corpus_train.txt \
        --eval-data data/corpus_eval.txt \
        --output-dir models/gpt2-coder \
        --epochs 3 \
        --batch-size 1 \
        --gradient-accumulation-steps 32
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch

from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


logger = logging.getLogger("train")


def _parse_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if not name:
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a causal language model for the CLI agent.")
    parser.add_argument("--model-name", default="gpt2-medium", help="Base model identifier or local path.")
    parser.add_argument("--train-data", required=True, type=Path, help="Path to training text file.")
    parser.add_argument("--eval-data", type=Path, help="Optional evaluation text file.")
    parser.add_argument("--output-dir", type=Path, default=Path("models/gpt2-coder"), help="Directory to save the fine-tuned model.")
    parser.add_argument("--block-size", type=int, default=1024, help="Token block size for each sample.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per device.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Number of accumulation steps.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Number of warmup steps.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision (GPU only).")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision (GPU only).")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing for large models.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping value (default 1.0).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps.")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps when evaluation data is provided.")
    parser.add_argument("--logging-steps", type=int, default=50, help="Log metrics every N steps.")
    parser.add_argument("--torch-dtype", type=str, default=None, help="Override torch dtype when loading (float16, bfloat16, float32).")
    parser.add_argument("--device-map", type=str, default=None, help="Device map for accelerate loading (e.g. 'auto', 'cpu', 'cuda').")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow remote model code (required by some Qwen/Devstral checkpoints).")
    parser.add_argument("--low-cpu-mem-usage", action="store_true", help="Stream weights in during load to reduce RAM spikes.")
    return parser.parse_args()


def prepare_dataset(
    tokenizer,
    path: Path,
    block_size: int,
) -> DatasetDict:
    logger.info("Loading dataset from %s", path)
    dataset = load_dataset("text", data_files=str(path))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    def group_texts(batch):
        concatenated = []
        for ids in batch["input_ids"]:
            concatenated.extend(ids)
        total_length = len(concatenated)
        # Drop the remainder to keep uniform block sizes.
        block = block_size
        total_length = (total_length // block) * block
        if total_length == 0:
            return {"input_ids": [], "labels": []}
        chunks = [
            concatenated[i : i + block] for i in range(0, total_length, block)
        ]
        return {
            "input_ids": chunks,
            "labels": chunks.copy(),
        }

    grouped = tokenized.map(group_texts, batched=True)
    return grouped


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    torch_dtype = _parse_dtype(args.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataset = prepare_dataset(tokenizer, args.train_data, args.block_size)["train"]
    eval_dataset = None
    if args.eval_data:
        eval_dataset = prepare_dataset(tokenizer, args.eval_data, args.block_size)["train"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Starting fine-tuning.")
    trainer.train()
    logger.info("Saving model to %s", args.output_dir)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
