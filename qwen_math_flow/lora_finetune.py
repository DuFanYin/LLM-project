"""
Add LoRA to Qwen2.5 and run fine-tuning on the math dataset.

Handles tokenized datasets with input_ids, attention_mask, labels.
Optional 4-bit/8-bit base model support via prepare_model_for_kbit_training.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments


def create_lora_model(
    model: Any,
    r: int = 8,
    lora_alpha: int = 16,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    use_4bit_or_8bit: bool = False,
) -> Any:
    """
    Wrap the base model with LoRA adapters using PEFT.

    Args:
        model: Hugging Face causal LM (e.g. Qwen2.5).
        r: LoRA rank.
        lora_alpha: LoRA alpha (scaling).
        target_modules: Module names to apply LoRA to; default Qwen2 linear layers.
        lora_dropout: Dropout in LoRA layers.
        bias: "none", "all", or "lora_only".
        task_type: PEFT task type (CAUSAL_LM for next-token prediction).
        use_4bit_or_8bit: If True, call prepare_model_for_kbit_training (for quantized base).

    Returns:
        PEFT model (PeftModel) ready for training.
    """
    if use_4bit_or_8bit:
        model = prepare_model_for_kbit_training(model)

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )
    peft_model = get_peft_model(model, config)
    return peft_model


def _default_data_collator(
    tokenizer: Any,
    pad_to_multiple_of: Optional[int] = 8,
) -> Any:
    """Build a data collator that pads to batch and respects labels."""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=pad_to_multiple_of,
    )


def run_finetune(
    peft_model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
    output_dir: Union[str, Path] = "output/lora_math",
    num_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    logging_steps: int = 10,
    save_steps: Optional[int] = None,
    save_total_limit: int = 2,
    eval_strategy: str = "no",
    max_steps: int = -1,
    bf16: bool = True,
    fp16: bool = False,
    report_to: str = "none",
    dataloader_pin_memory: bool = False,
    **training_kwargs: Any,
) -> Dict[str, float]:
    """
    Run LoRA fine-tuning with the Trainer API.

    Expects train_dataset to have "input_ids", "attention_mask", and "labels".
    Labels use -100 for positions that should not contribute to loss (e.g. prompt).

    Args:
        peft_model: PEFT-wrapped model.
        tokenizer: Pre-trained tokenizer.
        train_dataset: Dataset with tokenized input_ids, attention_mask, labels.
        eval_dataset: Optional validation dataset (same format).
        output_dir: Where to save checkpoints and adapter.
        num_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Gradient accumulation.
        learning_rate: Peak learning rate.
        warmup_ratio: Warmup ratio.
        logging_steps: Log every N steps.
        save_steps: Save every N steps. If None, derived from dataset size.
        save_total_limit: Keep only this many checkpoints.
        eval_strategy: "no", "steps", or "epoch".
        max_steps: Max training steps (-1 = use num_epochs).
        bf16 / fp16: Mixed precision.
        report_to: "none", "wandb", "tensorboard", etc.
        **training_kwargs: Passed to Trainer (e.g. callbacks).

    Returns:
        Trainer state / metrics (trainer.state.log_history, etc.).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_samples = len(train_dataset)
    batch_size = per_device_train_batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, n_samples // batch_size)
    if save_steps is None:
        save_steps = max(10, steps_per_epoch // 2)

    # Compute warmup_steps explicitly instead of using warmup_ratio arg on TrainingArguments
    if max_steps and max_steps > 0:
        total_steps = max_steps
    else:
        total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio) if warmup_ratio > 0 else 0

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        eval_strategy=eval_strategy,
        max_steps=max_steps,
        bf16=bf16,
        fp16=fp16,
        report_to=report_to,
        remove_unused_columns=False,
        dataloader_pin_memory=dataloader_pin_memory,
        **{k: v for k, v in training_kwargs.items() if k not in ("model", "args", "train_dataset", "eval_dataset", "data_collator", "tokenizer")},
    )

    data_collator = _default_data_collator(tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    return {
        "log_history": trainer.state.log_history,
        "train_loss": trainer.state.log_history[-1].get("loss", None) if trainer.state.log_history else None,
    }
