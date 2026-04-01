# Fine-Tuning Qwen2.5 on GSM8K with LoRA

Entry point: `FineTune.ipynb`. Supporting modules in `qwen_math_flow/`.

---

## Design Rationale

**LoRA** injects small trainable rank-decomposition matrices into the attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`), keeping the base model frozen. This cuts trainable parameters by orders of magnitude vs full fine-tuning. Rank (`r`) and scaling (`alpha`) control adapter capacity — the primary axes varied in the sweep.

**QLoRA** (`LOAD_IN_4BIT = True`) additionally loads the base model in 4-bit NF4 quantization, halving VRAM. Adapters are still trained in full precision. Disabled by default — adds a dependency and can slightly affect convergence.

**GSM8K** (~7,473 train / 1,319 test examples) is a clean, well-benchmarked arithmetic reasoning dataset. Each example is formatted as a two-turn chat — user asks the question, assistant provides a chain-of-thought answer — matching Qwen2.5-Instruct's expected input format.

**Label masking** sets prompt token labels to `-100` so loss is computed only on the assistant response, not the input question.

---

## Training Design

### Hyperparameter Sweep

`RUN_HYPERPARAMETER_SWEEP = True` trains one run per entry in `PARAM_COMBINATIONS`, each saved to its own subfolder (e.g. `run_002_lr2e-5_r16_a32_ep2`). Five configurations are defined, varying across three axes:

| Axis | Values |
|------|--------|
| Learning rate | 1e-5, 2e-5, 3e-5 |
| LoRA rank / alpha | 8/16, 16/32, 32/64 |
| Epochs | 1, 2, 3 |

`lora_alpha = 2 × lora_r` throughout, maintaining a consistent effective scaling factor.

### Early Stopping

A custom `LossConvergenceCallback` stops training if loss improvement over the last 20 log steps falls below `0.005`. The window of 20 steps covers ~4% of one epoch (~467 steps/epoch at effective batch size 16) — enough to confirm a plateau rather than react to transient noise.

### Speed Optimizations

| Optimization | Effect |
|---|---|
| `bf16` mixed precision | Reduces memory bandwidth and step time on Ampere+ GPUs |
| Gradient checkpointing | Recomputes activations on backward pass, cutting VRAM ~30–40% |
| Fused AdamW (`adamw_torch_fused`) | ~10–20% faster optimizer step via single CUDA kernel |
| Pinned memory (`dataloader_pin_memory=True`) | Faster async CPU→GPU transfers |
| Parallel data loading (`dataloader_num_workers=4`) | Prefetches batches so GPU isn't idle between steps |

> `dataloader_num_workers=4` can conflict with multiprocessing on macOS — reduce to `0` or `2` if worker errors appear.

---

## Metrics and Benchmarks

Only **training loss** is currently tracked. It confirms the model is learning but does not measure task performance.

The proper evaluation metrics, computed against the **GSM8K test split**:

| Metric | Description |
|---|---|
| **Exact match accuracy** | Final numerical answer matches ground truth. GSM8K answers end with `#### {number}`, making extraction straightforward. Primary benchmark metric. |
| **Pass@1** | Fraction correct on a single greedy decode. Standard GSM8K leaderboard number — directly comparable to published baselines. |
| **Eval loss** | Cross-entropy on the test split during training. Rising eval loss alongside falling train loss signals overfitting. |

### Baseline Reference

Measured baselines from this repo (exact match on GSM8K questions):

| Results file | Examples | Exact match |
|---|---|---|
| `results.json` | 10 | 10.0% |
| `daniil_results1.json` | 100 | 21.0% |

These are pre-fine-tune runs on small samples, reflecting the base model's performance without task adaptation. Published Pass@1 for reference:

| Model | GSM8K Pass@1 |
|---|---|
| Qwen2.5-0.5B-Instruct (no fine-tune) | ~40–45% |
| Qwen2.5-7B-Instruct | ~85–90% |
| GPT-4 | ~92% |

The gap between the repo baselines and published numbers likely reflects prompt sensitivity and sample variance at small n. A well-tuned LoRA on the full dataset should close some of this gap. Note: `sweep_index.json` ranks runs by final train loss, which does not reliably correlate with accuracy — test-split evaluation is needed to identify the best adapter.

---

## Data

| Source | Size | Usage |
|---|---|---|
| GSM8K train split (`openai/gsm8k`) | 7,473 examples | Fine-tuning (all examples, streamed from HuggingFace) |
| GSM8K test split | 1,319 examples | Held out — used for inference spot-checks in `Main.ipynb`, not yet for systematic eval |

No data files are checked into the repo. The dataset is downloaded on demand via the HuggingFace `datasets` library and cached locally by HF's default mechanism.

---

## Output Files

Fine-tune artifacts are written to `output/` (gitignored). A completed single run produces:

| File | Contents |
|------|----------|
| `training_run.log` | Timestamped full training log |
| `trainer_log_history.json` | Per-step loss, learning rate, epoch |
| `training_summary.json` | Hyperparameters, final loss, step count, early-stop config |
| `adapter_model.safetensors` | LoRA adapter weights (loadable with PEFT) |
| `adapter_config.json` | LoRA config (rank, alpha, target modules) |
| `tokenizer.json` / `tokenizer_config.json` | Saved tokenizer |
| `checkpoint-{step}/` | Intermediate checkpoints (last 2 kept) |

For sweeps, each run gets its own subfolder and a top-level `sweep_index.json` summarises final loss across all runs.

Existing baseline inference results in the repo root:

| File | Description |
|---|---|
| `results.json` | 10-example inference run, exact match 10% |
| `daniil_results1.json` | 100-example inference run, exact match 21% |

---

## Key Files

| File | Role |
|------|------|
| `FineTune.ipynb` | Orchestration: config, sweep loop, artifact saving |
| `qwen_math_flow/hyperparameters.py` | Central defaults for all training knobs |
| `qwen_math_flow/lora_finetune.py` | LoRA setup, Trainer config, `LossConvergenceCallback` |
| `qwen_math_flow/load_dataset.py` | GSM8K loading, chat formatting, label masking |
| `qwen_math_flow/download_model.py` | Model/tokenizer download with optional quantization |
