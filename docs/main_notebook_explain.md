## Main notebook: step-by-step guide

This file explains what each cell in `Main.ipynb` does, how it maps to `main.py`, what hyperparameters it uses, and what you should expect when you run it. It is **not** a full tutorial on LoRA/RAG; it is a concise reference for this specific project.

---

### Cell 0 – Overview (markdown)

- **What it is**: Short description of the 3-stage pipeline.
- **Matches**: Top comment in `main.py` (“Three-step flow: 1. RAG test -> 2. Train and save -> 3. Inference.”).
- **What you do**: Read only. No execution.

---

### Cell 1 – Imports

- **What it does**:
  - Imports standard libs: `json`, `Path`.
  - Imports PyTorch and PEFT/Transformers:
    - `torch`
    - `PeftModel`
    - `AutoModelForCausalLM`, `AutoTokenizer`, `BitsAndBytesConfig`
  - Imports project helpers:
    - `download_qwen_25_07b` (load base Qwen2.5-0.5B-Instruct)
    - `SafeEvalCalculatorClient`, `StubCalculatorClient` (calculator backends)
    - Dataset helpers: `format_gsm8k_as_chat`, `load_and_tokenize_math`
    - LoRA helpers: `create_lora_model`, `run_finetune`
    - RAG: `RAGCalculatorLayer`
- **Matches**:
  - Same imports as in `main.py` plus they are used directly in the notebook cells instead of via wrapper functions.
- **Expected result**:
  - No output; just defines everything you need for later cells.

---

### Cell 2 – RAG test (no model)

```python
from hyperparameters import USE_SAFE_EVAL_RAG_TEST

client = SafeEvalCalculatorClient() if USE_SAFE_EVAL_RAG_TEST else StubCalculatorClient()
rag = RAGCalculatorLayer(client)
tests = [
    "Result: [CALC: 2+3*4]",
    "First compute [CALC: 10/2], then add 5.",
]
print("RAG test (augment only, no model):")
for s in tests:
    out, pairs = rag.augment(s, inject_into_context=True)
    print(f"  in:  {s}")
    print(f"  out: {out}")
    print(f"  calc: {pairs}")
print("RAG test done.\\n")
```

- **Hyperparameters used**:
  - `USE_SAFE_EVAL_RAG_TEST` (from `hyperparameters.py`)
    - **True**: use `SafeEvalCalculatorClient` (real math).
    - **False**: use `StubCalculatorClient` (dummy responses).
- **What it does**:
  - Builds a `RAGCalculatorLayer` with the chosen calculator.
  - Runs `rag.augment` on two hard-coded strings containing `[CALC: ...]`.
  - Prints original string, augmented string (with numbers substituted), and the `(expression, result)` pairs.
- **Matches**:
  - Logic of `run_rag_test` in `main.py`, but without any CLI wrapper.
- **Expected output** (SafeEval enabled):
  - Lines like:
    - `in:  Result: [CALC: 2+3*4]`
    - `out: Result: 14`
    - `calc: [('2+3*4', '14')]`
  - Confirms the calculator + RAG replacement works independently of the model.

---

### Cell 3 – Train and save

```python
from hyperparameters import (
    ADAPTER_DIR,
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

model, tokenizer = download_qwen_25_07b(
    cache_dir=MODEL_CACHE_DIR,
    load_in_4bit=LOAD_IN_4BIT,
    device_map="auto" if LOAD_IN_4BIT else None,
)
tokenized_train = load_and_tokenize_math(
    tokenizer,
    name=DATASET_NAME,
    split=DATASET_SPLIT,
    max_samples=MAX_TRAIN_SAMPLES,
    max_length=MAX_LENGTH,
    message_formatter=format_gsm8k_as_chat,
)
cap_msg = "full train split" if MAX_TRAIN_SAMPLES is None else f"up to {MAX_TRAIN_SAMPLES} samples"
print(f"GSM8K training: {len(tokenized_train)} samples ({cap_msg}), {NUM_EPOCHS} epoch(s).")
peft_model = create_lora_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    use_4bit_or_8bit=LOAD_IN_4BIT,
)
run_finetune(
    peft_model,
    tokenizer,
    tokenized_train,
    output_dir=ADAPTER_DIR,
    num_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
)
print(f"Train done. Adapter + tokenizer saved to: {ADAPTER_DIR}\\n")
```

- **Hyperparameters used** (all defined in `hyperparameters.py`):
  - **Data selection (GSM8K only)**:
    - `DATASET_NAME`: e.g. `openai/gsm8k`.
    - `DATASET_SPLIT`: usually `train`.
    - `MAX_TRAIN_SAMPLES`: cap on examples (`None` = full split).
  - **Tokenization**:
    - `MAX_LENGTH`: max sequence length (chat prompt + answer), also used for padding.
  - **LoRA / training**:
    - `LORA_R`, `LORA_ALPHA`: LoRA rank and scaling.
    - `NUM_EPOCHS`: number of epochs.
    - `PER_DEVICE_TRAIN_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`: effective batch size × steps.
    - `LEARNING_RATE`: peak LR for AdamW.
  - **Model loading**:
    - `LOAD_IN_4BIT`: whether to load base model in 4-bit (QLoRA-style).
    - `MODEL_CACHE_DIR`: optional HF cache directory.
  - **Output**:
    - `ADAPTER_DIR`: where LoRA adapter + tokenizer are saved (e.g. `output/lora_math`).
- **What it does**:
  1. **Load base model + tokenizer** via `download_qwen_25_07b`.
  2. **Load + tokenize** via `load_and_tokenize_math(...)` (GSM8K chat format, label masking on assistant only).
  3. **Wrap with LoRA**:
     - `create_lora_model(model, r=LORA_R, lora_alpha=LORA_ALPHA, use_4bit_or_8bit=LOAD_IN_4BIT)`.
  4. **Train** with `run_finetune`:
     - Uses `TrainingArguments` under the hood with computed `warmup_steps` (no deprecated `warmup_ratio`).
     - Uses a `DataCollatorForLanguageModeling` built from the tokenizer.
  5. **Save artifacts**:
     - `trainer.save_model(ADAPTER_DIR)`.
     - `tokenizer.save_pretrained(ADAPTER_DIR)`.
- **Matches**:
  - Same logic as `train_and_save` in `main.py`, but unwrapped and split into explicit steps in the cell.
- **Expected output**:
  - A line like:
    - `GSM8K training: 100 samples (up to 100 samples), 2 epoch(s).`
  - Training progress bar/logs from `transformers.Trainer`.
  - Final line:
    - `Train done. Adapter + tokenizer saved to: output/lora_math`
  - After this runs, `ADAPTER_DIR` should contain LoRA weights and tokenizer files (e.g. `adapter_config.json`, `tokenizer.json`, etc.).

---

### Cell 4 – Inference

Loads `openai/gsm8k` (`main` config), samples **`INFERENCE_NUM_QUESTIONS`** random rows from **`INFERENCE_QUESTION_SPLIT`** (default `test`), and runs the same chat + RAG generation for each. User text matches training:  
`Solve the following math problem step by step.\n\n{question}`.

```python
import random
from datasets import load_dataset
from hyperparameters import (
    ADAPTER_DIR,
    INFERENCE_NUM_QUESTIONS,
    INFERENCE_QUESTION_SPLIT,
    INFERENCE_RANDOM_SEED,
    LOAD_IN_4BIT_INFERENCE,
    MAX_NEW_TOKENS,
)
# ... load adapter, base model, PeftModel, rag ...
raw = load_dataset("openai/gsm8k", "main", split=INFERENCE_QUESTION_SPLIT)
rng = random.Random(INFERENCE_RANDOM_SEED)
indices = rng.sample(range(len(raw)), min(INFERENCE_NUM_QUESTIONS, len(raw)))
for i, idx in enumerate(indices, start=1):
    question = raw[idx]["question"]
    user_content = f"Solve the following math problem step by step.\n\n{question}"
    # apply_chat_template → generate_with_rag → print question + answer
```

- **Hyperparameters used**:
  - `ADAPTER_DIR`, `LOAD_IN_4BIT_INFERENCE`, `MAX_NEW_TOKENS` (as before).
  - `INFERENCE_NUM_QUESTIONS` (default 5), `INFERENCE_RANDOM_SEED` (reproducible samples), `INFERENCE_QUESTION_SPLIT` (`train` or `test`; default `test` to avoid overlap with training on `train`).
- **Expected output**:
  - Five blocks like `--- [1/5] (GSM8K test index <idx>) ---`, each with a **Question** and **Answer** line.

---

### How to use the notebook end-to-end

- **RAG test only**:
  - Run **Cell 1** (imports) → **Cell 2** (RAG test).
- **Train only**:
  - Adjust hyperparameters in `hyperparameters.py` if needed (e.g. `MAX_TRAIN_SAMPLES`, `NUM_EPOCHS`).
  - Run **Cell 1** → **Cell 3**.
- **Inference only (using existing adapter)**:
  - Ensure `ADAPTER_DIR` already exists from a previous run.
  - Optional: tune `INFERENCE_NUM_QUESTIONS`, `INFERENCE_RANDOM_SEED`, `INFERENCE_QUESTION_SPLIT` in `hyperparameters.py`.
  - Run **Cell 1** → **Cell 4** (inference).
- **Full pipeline in notebook**:
  - Run **Cell 1** → **Cell 2** → **Cell 3** → **Cell 4**.

This mirrors the `main.py` entrypoints (`rag_test`, `train_save`, `inference`, `all`), but with each step as an explicit cell so you can inspect and tweak intermediate state.

