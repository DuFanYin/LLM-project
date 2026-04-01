# LLM-project

Fine-tune **Qwen2.5-0.5B-Instruct** on GSM8K math with **LoRA**, then run inference augmented by a **RAG calculator layer** that evaluates arithmetic expressions the model emits (e.g. `[CALC: 12*7]`).

---

## Project structure

```
LLM-project/
├── FineTune.ipynb              # LoRA fine-tuning — sweep, early stopping, artifacts
├── Main.ipynb                  # Inference pipeline — load adapter, RAG, evaluate
├── qwen_math_flow/
│   ├── hyperparameters.py      # All training/inference knobs
│   ├── download_model.py       # Qwen2.5 model + tokenizer loader (optional 4-bit)
│   ├── load_dataset.py         # GSM8K load, chat formatting, tokenization
│   ├── lora_finetune.py        # LoRA setup, Trainer config, LossConvergenceCallback
│   ├── rag_calculator.py       # Extract [CALC:…] → evaluate → inject result
│   ├── external_calculator.py  # CalculatorClient, SafeEval, Stub implementations
│   └── test_rag_calculator.py  # Unit tests for RAG / calculator
├── results.json                # Baseline inference results (10 examples, 10% EM)
├── daniil_results1.json        # Baseline inference results (100 examples, 21% EM)
├── README_finetune.md          # Fine-tuning design, rationale, metrics
├── docs/
│   ├── REPO_STRUCTURE.md
│   └── main_notebook_explain.md
├── requirements.txt
└── output/                     # (gitignored) all generated artifacts
    ├── fine_tune/              # FineTune.ipynb outputs — one subfolder per sweep run
    │   └── run_NNN_lr…_rN_aN_epN/
    │       ├── adapter_model.safetensors
    │       ├── adapter_config.json
    │       ├── tokenizer.json / tokenizer_config.json
    │       ├── training_summary.json
    │       ├── trainer_log_history.json
    │       ├── training_run.log
    │       └── checkpoint-{step}/
    ├── sweep_index.json        # Final loss summary across all sweep runs
    └── lora_math/              # Legacy single-run adapter output
```

---

## Pipeline

**Fine-tuning** (`FineTune.ipynb`)
1. Load Qwen2.5-0.5B-Instruct from HuggingFace
2. Format GSM8K train split (~7,473 examples) as chat, tokenize with label masking
3. Wrap model with LoRA adapters (`q/k/v/o_proj`)
4. Train with HF `Trainer`; early-stop on loss convergence
5. Save adapter weights + tokenizer per run

**Inference** (`Main.ipynb`)
1. Load base model + saved LoRA adapter
2. Generate answers; RAG layer intercepts `[CALC: …]` placeholders, evaluates them, injects results
3. Score against GSM8K test split (exact match)

---

## Setup

```bash
pip install -r requirements.txt
```

Optional — 4-bit QLoRA (reduces VRAM ~40%):
```bash
pip install bitsandbytes
# then set LOAD_IN_4BIT = True in qwen_math_flow/hyperparameters.py
```

---

## Configuration

All defaults live in `qwen_math_flow/hyperparameters.py`. Key knobs:

| Parameter | Default | Effect |
|---|---|---|
| `MAX_TRAIN_SAMPLES` | `None` | `None` = full dataset; set an int for quick runs |
| `NUM_EPOCHS` | `2` | Max epochs (early stopping may end sooner) |
| `LORA_R` / `LORA_ALPHA` | `8` / `16` | LoRA rank and scaling |
| `LOAD_IN_4BIT` | `False` | Enable QLoRA (requires bitsandbytes) |

For hyperparameter sweeps, edit `PARAM_COMBINATIONS` in `FineTune.ipynb` and set `RUN_HYPERPARAMETER_SWEEP = True`.

---

## Baseline results

Pre-fine-tune exact match on GSM8K questions:

| File | Samples | Exact match |
|---|---|---|
| `results.json` | 10 | 10% |
| `daniil_results1.json` | 100 | 21% |

See `README_finetune.md` for fine-tuning design, metrics, and benchmark targets.
