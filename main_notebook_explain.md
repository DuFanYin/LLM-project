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
    - Dataset helpers: `format_gsm8k_as_chat`, `format_multi_math_as_chat`, `load_and_tokenize_math`, `load_multi_math_dataset`, `tokenize_math_dataset`
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
    MAX_PER_DATASET,
    MAX_TRAIN_SAMPLES,
    MODEL_CACHE_DIR,
    NUM_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    USE_MULTI_DATASET,
)

model, tokenizer = download_qwen_25_07b(
    cache_dir=MODEL_CACHE_DIR,
    load_in_4bit=LOAD_IN_4BIT,
    device_map="auto" if LOAD_IN_4BIT else None,
)
if USE_MULTI_DATASET:
    raw_train = load_multi_math_dataset(max_per_dataset=MAX_PER_DATASET)
    tokenized_train = tokenize_math_dataset(
        raw_train, tokenizer, message_formatter=format_multi_math_as_chat, max_length=MAX_LENGTH
    )
    cap_msg = "full dataset" if MAX_PER_DATASET is None else f"up to {MAX_PER_DATASET} per dataset"
    print(f"Multi-dataset training: {len(tokenized_train)} samples ({cap_msg}), {NUM_EPOCHS} epoch(s).")
else:
    tokenized_train = load_and_tokenize_math(
        tokenizer,
        name=DATASET_NAME,
        split=DATASET_SPLIT,
        max_samples=MAX_TRAIN_SAMPLES,
        max_length=MAX_LENGTH,
        message_formatter=format_gsm8k_as_chat,
    )
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
  - **Data selection**:
    - `USE_MULTI_DATASET`:
      - **True**: use 4-way mix (GSM8K, ASDiv, MetaMathQA, OpenMathInstruct) via `load_multi_math_dataset`.
      - **False**: use single dataset (`DATASET_NAME`, `DATASET_SPLIT`) via `load_and_tokenize_math`.
    - `MAX_PER_DATASET`:
      - If multi-dataset: max examples per source (e.g. 100 → 400 total).
      - `None`: full dataset for each source.
    - `DATASET_NAME`, `DATASET_SPLIT`, `MAX_TRAIN_SAMPLES`:
      - Used only when `USE_MULTI_DATASET` is False.
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
  2. **Load data**:
     - Multi-dataset: `load_multi_math_dataset(max_per_dataset=MAX_PER_DATASET)` → normalized `question` / `answer`.
     - Single dataset: `load_and_tokenize_math(...)` on e.g. GSM8K.
  3. **Tokenize** with Qwen chat template using `tokenize_math_dataset`:
     - Creates `input_ids`, `attention_mask`, `labels` with loss only on the assistant part.
  4. **Wrap with LoRA**:
     - `create_lora_model(model, r=LORA_R, lora_alpha=LORA_ALPHA, use_4bit_or_8bit=LOAD_IN_4BIT)`.
  5. **Train** with `run_finetune`:
     - Uses `TrainingArguments` under the hood with computed `warmup_steps` (no deprecated `warmup_ratio`).
     - Uses a `DataCollatorForLanguageModeling` built from the tokenizer.
  6. **Save artifacts**:
     - `trainer.save_model(ADAPTER_DIR)`.
     - `tokenizer.save_pretrained(ADAPTER_DIR)`.
- **Matches**:
  - Same logic as `train_and_save` in `main.py`, but unwrapped and split into explicit steps in the cell.
- **Expected output**:
  - A line like:
    - `Multi-dataset training: 400 samples (up to 100 per dataset), 2 epoch(s).`
  - Training progress bar/logs from `transformers.Trainer`.
  - Final line:
    - `Train done. Adapter + tokenizer saved to: output/lora_math`
  - After this runs, `ADAPTER_DIR` should contain LoRA weights and tokenizer files (e.g. `adapter_config.json`, `tokenizer.json`, etc.).

---

### Cell 4 – Inference

```python
from hyperparameters import ADAPTER_DIR, INFERENCE_QUERY, LOAD_IN_4BIT_INFERENCE, MAX_NEW_TOKENS

adapter_path = Path(ADAPTER_DIR)
with open(adapter_path / "adapter_config.json") as f:
    base_model_id = json.load(f).get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")

tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
base_kwargs = dict(device_map="auto", trust_remote_code=True)
if LOAD_IN_4BIT_INFERENCE:
    base_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
else:
    base_kwargs["torch_dtype"] = "auto"

base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **base_kwargs)
model = PeftModel.from_pretrained(base_model, str(adapter_path))
model.eval()

rag = RAGCalculatorLayer(SafeEvalCalculatorClient())
messages = [{"role": "user", "content": INFERENCE_QUERY}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
answer = rag.generate_with_rag(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)

print(f"Inference query: {INFERENCE_QUERY}")
print(f"Inference answer: {answer}\\n")
```

- **Hyperparameters used**:
  - `ADAPTER_DIR`: directory produced by training cell.
  - `INFERENCE_QUERY`: the user query for inference; can contain `[CALC: ...]`.
  - `LOAD_IN_4BIT_INFERENCE`: whether to load base model in 4-bit for inference.
  - `MAX_NEW_TOKENS`: cap on generated tokens for the answer.
- **What it does**:
  1. **Resolve base model ID**:
     - Reads `adapter_config.json` from `ADAPTER_DIR` and gets `base_model_name_or_path`, defaulting to `"Qwen/Qwen2.5-0.5B-Instruct"`.
  2. **Load tokenizer**:
     - From `ADAPTER_DIR` so it matches the fine-tuned model.
  3. **Load base+LoRA model**:
     - `AutoModelForCausalLM.from_pretrained(base_model_id, **base_kwargs)`.
     - Wrap with `PeftModel.from_pretrained(base_model, ADAPTER_DIR)`.
     - `model.eval()` for inference.
  4. **Build RAG layer**:
     - `RAGCalculatorLayer(SafeEvalCalculatorClient())` – uses real calculator.
  5. **Build chat prompt**:
     - `tokenizer.apply_chat_template([...], add_generation_prompt=True)` to format user query for Qwen chat.
  6. **Generate with RAG**:
     - `rag.generate_with_rag(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)`.
     - Internally:
       - Calls `model.generate(...)` with safe generation kwargs (only adds `temperature` / `top_p` if supported by the model’s `generation_config`).
       - Scans output for `[CALC: ...]`, calls calculator, and replaces placeholders.
  7. **Print result**:
     - Echoes `INFERENCE_QUERY`.
     - Prints the full final answer string.
- **Matches**:
  - Same overall behavior as `load_for_inference + run_inference + inference_with_rag` in `main.py`.
- **Expected output** (example):
  - Query:
    - `Inference query: What is [CALC: 7*8]?`
  - Answer:
    - A full Qwen chat-style response where the calculator has already replaced `[CALC: 7*8]` with `56` and the model explains the result.

---

### How to use the notebook end-to-end

- **RAG test only**:
  - Run **Cell 1** (imports) → **Cell 2** (RAG test).
- **Train only**:
  - Adjust hyperparameters in `hyperparameters.py` if needed (e.g. `MAX_PER_DATASET`, `NUM_EPOCHS`).
  - Run **Cell 1** → **Cell 3**.
- **Inference only (using existing adapter)**:
  - Ensure `ADAPTER_DIR` already exists from a previous run.
  - Set `INFERENCE_QUERY` in `hyperparameters.py`.
  - Run **Cell 1** → **Cell 4**.
- **Full pipeline in notebook**:
  - Run **Cell 1** → **Cell 2** → **Cell 3** → **Cell 4**.

This mirrors the `main.py` entrypoints (`rag_test`, `train_save`, `inference`, `all`), but with each step as an explicit cell so you can inspect and tweak intermediate state.

