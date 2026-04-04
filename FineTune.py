#!/usr/bin/env python
# coding: utf-8

# # Fine-tune (LoRA)
# 
# Fine-tunes a configurable Qwen model on GSM8K using LoRA. Artifacts go to `output/fine_tune/` (gitignored).
# 
# | Output | Contents |
# |--------|---------|
# | Cell output | tqdm bar + `[train] step=… loss=… lr=… epoch=…` per log event |
# | `training_run.log` | Timestamped log, appended each run |
# | `trainer_log_history.json` | Per-step `loss`, `learning_rate`, `epoch` from HF Trainer |
# | `training_summary.json` | Hyperparameters + final loss / step / epoch |
# | `adapter_*/`, checkpoints | Saved by HF Trainer under the run folder |
# 
# **Single run:** `RUN_HYPERPARAMETER_SWEEP = False` — uses defaults from `hyperparameters.py`.  
# **Sweep:** `True` — one run per entry in `PARAM_COMBINATIONS`, each in its own subfolder (`run_000_…`).
# 
# Run cells top to bottom.

# ### 1 — Imports and callback
# 
# Imports the stack and defines `NotebookProgressCallback` (prints `[train]` lines per log event). Run once per session.

# In[1]:


import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import TrainerCallback

from qwen_math_flow.download_model import download_qwen_25_07b
from qwen_math_flow.load_dataset import format_gsm8k_as_chat, load_and_tokenize_math
from qwen_math_flow.lora_finetune import create_lora_model, run_finetune


class NotebookProgressCallback(TrainerCallback):
    """Print one line per Trainer log so loss / lr / epoch show in the notebook while training."""

    def on_train_begin(self, args, state, control, **kwargs):
        print(
            "[train] started | "
            f"epochs={args.num_train_epochs} | log every {args.logging_steps} step(s) | "
            "progress bar + metrics lines below",
            flush=True,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        parts = [f"step={state.global_step}"]
        if logs.get("loss") is not None:
            parts.append(f"loss={logs['loss']:.4f}")
        if logs.get("learning_rate") is not None:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        if logs.get("epoch") is not None:
            parts.append(f"epoch={logs['epoch']:.4f}")
        print("[train] " + " | ".join(parts), flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        print(f"[train] finished | global_step={state.global_step}", flush=True)


# ### 2 — Config
# 
# Set the output path, sweep toggle, and logging verbosity. Training defaults (LR, LoRA rank, batch size, etc.) come from `hyperparameters.py`; entries in `PARAM_COMBINATIONS` override only the keys they list.
# 
# Re-run this cell whenever you change sweep settings.

# In[2]:


from itertools import product

from qwen_math_flow.hyperparameters import (
    DATASET_NAME,
    DATASET_SPLIT,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOAD_IN_4BIT,
    LORA_ALPHA,
    LORA_R,
    MAX_LENGTH,
    MAX_TRAIN_SAMPLES,
    MODEL_CACHE_DIR,
    NUM_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
)

OUTPUT_BASE = "output/fine_tune"
MODEL_NAME = "Qwen/Qwen2-1.5B"

# False: one run using hyperparameters.py only. True: one run per dict in PARAM_COMBINATIONS.
RUN_HYPERPARAMETER_SWEEP = True

# Sweeps: each dict can override training keys; missing keys use hyperparameters.py defaults.
# Broader search space (edit these lists to expand/shrink):
LR_GRID = [1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4]
LORA_R_GRID = [8, 16, 32]
EPOCHS_GRID = [3]
MAX_TRAIN_SAMPLES_GRID = [None]  # use [2000, 4000, None] for faster pilot + full runs

PARAM_COMBINATIONS = [
    {
        "learning_rate": lr,
        "lora_r": r,
        "lora_alpha": r * 2,
        "num_epochs": ep,
        "max_train_samples": m,
    }
    for lr, r, ep, m in product(LR_GRID, LORA_R_GRID, EPOCHS_GRID, MAX_TRAIN_SAMPLES_GRID)
]
# Current size: 6 * 3 * 1 * 1 = 18 combinations.


def merged_training_params(overrides: dict) -> dict:
    """Defaults from hyperparameters.py, overridden by one sweep row."""
    return {
        "learning_rate": overrides.get("learning_rate", LEARNING_RATE),
        "lora_r": overrides.get("lora_r", LORA_R),
        "lora_alpha": overrides.get("lora_alpha", LORA_ALPHA),
        "num_epochs": overrides.get("num_epochs", 3),
        "per_device_train_batch_size": overrides.get(
            "per_device_train_batch_size", PER_DEVICE_TRAIN_BATCH_SIZE
        ),
        "gradient_accumulation_steps": overrides.get(
            "gradient_accumulation_steps", GRADIENT_ACCUMULATION_STEPS
        ),
        "max_length": overrides.get("max_length", MAX_LENGTH),
        "max_train_samples": overrides.get("max_train_samples", MAX_TRAIN_SAMPLES),
    }


def combination_subdir(index: int, m: dict) -> str:
    """Unique folder name for one combination."""
    lr = m["learning_rate"]
    lr_s = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"run_{index:03d}_lr{lr_s}_r{m['lora_r']}_a{m['lora_alpha']}_ep{m['num_epochs']}"


SWEEP_RUNS = PARAM_COMBINATIONS if RUN_HYPERPARAMETER_SWEEP else [{}]
NUM_COMBINATIONS = len(SWEEP_RUNS)

# Resume mode: skip run folders that already contain completed summaries.
RESUME_INCOMPLETE_ONLY = True

# Trainer: log loss / lr every N steps (1 = every step — most verbose)
LOGGING_STEPS = 1
SHOW_NOTEBOOK_LOG_LINES = True
DISABLE_TQDM = False
TRANSFORMERS_DEBUG = False

Path(OUTPUT_BASE).mkdir(parents=True, exist_ok=True)
print(f"Output base: {Path(OUTPUT_BASE).resolve()}")
print(f"Model: {MODEL_NAME}")
print(f"RUN_HYPERPARAMETER_SWEEP={RUN_HYPERPARAMETER_SWEEP}  →  {NUM_COMBINATIONS} training run(s)")
if RUN_HYPERPARAMETER_SWEEP:
    for i, o in enumerate(PARAM_COMBINATIONS):
        print(f"  [{i}] {o}  →  {combination_subdir(i, merged_training_params(o))}")
print(f"RESUME_INCOMPLETE_ONLY={RESUME_INCOMPLETE_ONLY}")
print(
    f"LOGGING_STEPS={LOGGING_STEPS}  SHOW_NOTEBOOK_LOG_LINES={SHOW_NOTEBOOK_LOG_LINES}  "
    f"DISABLE_TQDM={DISABLE_TQDM}  TRANSFORMERS_DEBUG={TRANSFORMERS_DEBUG}"
)


# ### 3 — Train
# 
# Loads the model and data, runs `run_finetune` for each combination, and writes `training_summary.json`, `trainer_log_history.json`, and `training_run.log` per run. In sweep mode, also writes `sweep_index.json` at the top level. If `RESUME_INCOMPLETE_ONLY=True`, completed run folders are skipped automatically.
# 
# Re-run config first if hyperparameters changed.

# In[3]:


def _setup_run_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.DEBUG if TRANSFORMERS_DEBUG else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8", mode="a"),
        ],
        force=True,
    )
    for name in ("transformers", "transformers.trainer", "qwen_math_flow.lora_finetune"):
        logging.getLogger(name).setLevel(logging.DEBUG if TRANSFORMERS_DEBUG else logging.INFO)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    if TRANSFORMERS_DEBUG:
        transformers.logging.set_verbosity_debug()
    else:
        transformers.logging.set_verbosity_info()


all_summaries = []
skipped_runs = 0


def _is_completed_run(out_dir: Path) -> bool:
    summary_path = out_dir / "training_summary.json"
    hist_path = out_dir / "trainer_log_history.json"
    if not summary_path.exists() or not hist_path.exists():
        return False
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        with open(hist_path, "r", encoding="utf-8") as f:
            h = json.load(f)
    except Exception:
        return False
    if not isinstance(h, list) or len(h) == 0:
        return False
    last = h[-1] if isinstance(h[-1], dict) else {}
    # Completion marker from Trainer's final summary record.
    return "train_loss" in last and s.get("global_step") is not None


for run_index, overrides in enumerate(SWEEP_RUNS):
    merged = merged_training_params(overrides)
    out = (
        Path(OUTPUT_BASE) / combination_subdir(run_index, merged)
        if RUN_HYPERPARAMETER_SWEEP
        else Path(OUTPUT_BASE)
    )
    out.mkdir(parents=True, exist_ok=True)

    if RESUME_INCOMPLETE_ONLY and _is_completed_run(out):
        print(f"[skip] run {run_index + 1}/{NUM_COMBINATIONS} already completed: {out}", flush=True)
        skipped_runs += 1
        continue

    log_file = out / "training_run.log"

    print(
        f"\n=== Run {run_index + 1}/{NUM_COMBINATIONS}  {out.name}  merged={merged} ===\n",
        flush=True,
    )
    _setup_run_logging(log_file)
    logging.info("Logging to %s", log_file.resolve())

    model, tokenizer = download_qwen_25_07b(
        model_id=MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        load_in_4bit=LOAD_IN_4BIT,
        device_map="auto" if LOAD_IN_4BIT else None,
    )
    tokenized_train = load_and_tokenize_math(
        tokenizer,
        name=DATASET_NAME,
        split=DATASET_SPLIT,
        max_samples=merged["max_train_samples"],
        max_length=merged["max_length"],
        message_formatter=format_gsm8k_as_chat,
    )
    cap_msg = (
        "full train split"
        if merged["max_train_samples"] is None
        else f"up to {merged['max_train_samples']} samples"
    )
    logging.info(
        "GSM8K training: %s samples (%s), %s epoch(s).",
        len(tokenized_train),
        cap_msg,
        merged["num_epochs"],
    )

    peft_model = create_lora_model(
        model,
        r=merged["lora_r"],
        lora_alpha=merged["lora_alpha"],
        use_4bit_or_8bit=LOAD_IN_4BIT,
    )
    callbacks = []
    if SHOW_NOTEBOOK_LOG_LINES:
        callbacks.append(NotebookProgressCallback())

    metrics = run_finetune(
        peft_model,
        tokenizer,
        tokenized_train,
        output_dir=str(out),
        num_epochs=merged["num_epochs"],
        per_device_train_batch_size=merged["per_device_train_batch_size"],
        gradient_accumulation_steps=merged["gradient_accumulation_steps"],
        learning_rate=merged["learning_rate"],
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        logging_strategy="steps",
        disable_tqdm=DISABLE_TQDM,
        callbacks=callbacks,
        save_log_history_json=True,
    )

    summary = {
        "run_index": run_index,
        "run_hyperparameter_sweep": RUN_HYPERPARAMETER_SWEEP,
        "sweep_overrides": overrides,
        "merged_params": merged,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out.resolve()),
        "log_files": {
            "training_run_log": str((out / "training_run.log").resolve()),
            "trainer_log_history_json": str((out / "trainer_log_history.json").resolve()),
        },
        "logging_steps": LOGGING_STEPS,
        "show_notebook_log_lines": SHOW_NOTEBOOK_LOG_LINES,
        "disable_tqdm": DISABLE_TQDM,
        "transformers_debug": TRANSFORMERS_DEBUG,
        "model_name": MODEL_NAME,
        "dataset": DATASET_NAME,
        "split": DATASET_SPLIT,
        "num_train_samples": len(tokenized_train),
        "num_epochs": merged["num_epochs"],
        "per_device_train_batch_size": merged["per_device_train_batch_size"],
        "gradient_accumulation_steps": merged["gradient_accumulation_steps"],
        "learning_rate": merged["learning_rate"],
        "max_length": merged["max_length"],
        "lora_r": merged["lora_r"],
        "lora_alpha": merged["lora_alpha"],
        "load_in_4bit": LOAD_IN_4BIT,
        "final_train_loss": metrics.get("train_loss"),
        "global_step": metrics.get("global_step"),
        "epoch": metrics.get("epoch"),
    }
    with open(out / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    all_summaries.append(summary)

    logging.info("Done run %s/%s: %s", run_index + 1, NUM_COMBINATIONS, out.resolve())
    del model, peft_model, tokenizer, tokenized_train
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if RUN_HYPERPARAMETER_SWEEP:
    sweep_index_path = Path(OUTPUT_BASE) / "sweep_index.json"
    with open(sweep_index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "num_runs": NUM_COMBINATIONS,
                "num_skipped": skipped_runs,
                "num_executed": len(all_summaries),
                "runs": [
                    {
                        "output_dir": s["output_dir"],
                        "final_train_loss": s.get("final_train_loss"),
                        "merged_params": s["merged_params"],
                    }
                    for s in all_summaries
                ],
            },
            f,
            indent=2,
        )
    print(f"Wrote sweep index: {sweep_index_path.resolve()}")

print(
    f"Finished: total={NUM_COMBINATIONS}, executed={len(all_summaries)}, skipped={skipped_runs}.\n"
    f"Artifacts under:\n  {Path(OUTPUT_BASE).resolve()}\n"
)
