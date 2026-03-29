# Repository structure (brief)

Related files are grouped below. A compact **directory tree** is first; **tables** explain each part.

Paths marked **(gitignored)** match `.gitignore` and are not tracked by Git (they may still exist locally).

---

### Directory tree

```
LLM-project/
├── README.md                      # project overview, quick start, hardware / dataset notes
├── REPO_STRUCTURE.md              # this file (tree + grouped tables)
├── main_notebook_explain.md       # cell-by-cell guide for Main.ipynb
├── requirements.txt               # Python dependencies
├── hyperparameters.py             # training / inference constants (paths, LoRA, data caps)
├── main.py                        # script entry: RAG → train → inference
├── Main.ipynb                     # same pipeline as notebook cells
├── .gitignore                     # git ignore rules
├── qwen_math_flow/                # core library package
│   ├── __init__.py                # package exports
│   ├── download_model.py          # load Qwen2.5 base model + tokenizer
│   ├── load_dataset.py            # GSM8K load + chat tokenization
│   ├── lora_finetune.py           # LoRA + Hugging Face Trainer
│   ├── rag_calculator.py          # RAG: [CALC: …] → calculator → inject
│   ├── external_calculator.py     # CalculatorClient, SafeEval, Stub
│   └── test_rag_calculator.py     # tests for RAG / calculator
├── output/                        # (gitignored) training saves live here
│   └── lora_math/                 # (gitignored) default LoRA adapter + tokenizer (generated)
├── Project.ipynb                  # project / experiment notebook
├── Prompt.ipynb                   # dataset preview + prompt experiments
├── project_prompt.ipynb           # related prompt notebook
├── math_outputs.json              # (gitignored) optional cached outputs
├── all_datasets_output.json       # (gitignored) optional dataset dump
├── gsm8k_outputs.json             # (gitignored) optional GSM8K outputs
└── …                              # other *_outputs.json (gitignored) if present
```

---

### Documentation & config

| Path | Role |
|------|------|
| `README.md` | Overview, quick start, hardware / dataset notes |
| `REPO_STRUCTURE.md` | This file |
| `main_notebook_explain.md` | Cell-by-cell guide for `Main.ipynb` |
| `requirements.txt` | Python dependencies |
| `hyperparameters.py` | Training / inference constants (paths, LoRA, data caps) |
| `.gitignore` | Git ignore rules |

---

### Entry points (how you run the pipeline)

| Path | Role |
|------|------|
| `main.py` | Script entry: RAG test → train → inference (reads `hyperparameters.py`) |
| `Main.ipynb` | Same pipeline as separate cells (imports → RAG → train → inference) |

---

### Core package (`qwen_math_flow/`)

| File | Role |
|------|------|
| `download_model.py` | Load Qwen2.5 base model + tokenizer |
| `load_dataset.py` | GSM8K load + chat tokenization |
| `lora_finetune.py` | LoRA + Hugging Face `Trainer` |
| `rag_calculator.py` | RAG: `[CALC: …]` → calculator → inject |
| `external_calculator.py` | `CalculatorClient`, SafeEval, Stub |
| `test_rag_calculator.py` | Tests for RAG / calculator |
| `__init__.py` | Package exports |

---

### Training output (generated, gitignored)

| Path | Role |
|------|------|
| `output/lora_math/` | Default LoRA adapter + tokenizer (checkpoints may appear here); entire `output/` is gitignored |

---

### Notebooks & experiments

| Path | Role |
|------|------|
| `Project.ipynb` | Project / experiment notebook |
| `Prompt.ipynb` | Dataset preview + prompt-style experiments |
| `project_prompt.ipynb` | Related prompt notebook |

---

### Data / cached outputs (optional artifacts, gitignored)

| Pattern | Role |
|---------|------|
| `math_outputs.json` | Example cached outputs (if present) |
| `*_outputs.json`, `all_datasets_output.json`, `gsm8k_outputs.json` | Other dumps (if present) |

---

### Git-ignored paths (see `.gitignore`)

| Pattern | Why |
|---------|-----|
| `output/` | Training artifacts (adapters, checkpoints, tokenizers) |
| `*_outputs.json`, `math_outputs.json`, `all_datasets_output.json`, `gsm8k_outputs.json` | Regenerable JSON dumps |
| `__pycache__/`, `*.py[cod]`, `*.egg-info/`, `dist/`, `build/` | Python build / cache |
| `.venv/`, `venv/`, `env/` | Local virtual environments |
| `.ipynb_checkpoints/` | Jupyter autosave checkpoints |
| `.DS_Store` | macOS folder metadata |
| `.env`, `.env.*` | Secrets / local env files |

---

**Start here:** `README.md` → change behaviour in `hyperparameters.py` → implementation details in `qwen_math_flow/`.
