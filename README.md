# LLM-project

Fine-tune **Qwen2.5** on math (e.g. GSM8K) with **LoRA**, then run inference with a **RAG calculator layer** that evaluates expressions (e.g. `[CALC: 2+3]`) via an external calculator.

---

## Overview

| Step | What happens |
|------|----------------|
| 1 | Download base model (Qwen2.5-0.5B-Instruct) and tokenizer from Hugging Face. |
| 2 | Load math dataset, format as chat, tokenize with label masking (train only on assistant reply). |
| 3 | Add LoRA adapters and train with the Trainer; save adapter + tokenizer. |
| 4 | At inference, wrap generation in a RAG layer: detect calculator placeholders → call calculator → inject results. |

**Architecture in short:** Base causal LM + LoRA (PEFT) on attention layers; training on math Q&A in chat form; inference = same model + adapter + optional RAG that calls your `CalculatorClient` (stub or real).

---

## Quick start

```bash
pip install -r requirements.txt
```

From repo root, run the full pipeline on a tiny subset (smoke test):

```python
from main import run_full_flow
from qwen_math_flow import StubCalculatorClient

result = run_full_flow(
    max_train_samples=20,
    num_epochs=1,
    output_dir="output/lora_math_smoke",
    calculator_client=StubCalculatorClient(),
)
# result["peft_model"], result["tokenizer"], result["rag_layer"]; adapter saved under output_dir
```

Confirm that `output/lora_math_smoke` exists and contains the adapter. Optionally run one inference with `inference_with_rag(result["peft_model"], result["tokenizer"], result["rag_layer"], "What is [CALC: 5*6]?")` to verify the RAG path.

---

## Step-by-step: how to run the project

Start small to verify the pipeline, then scale up.

| Phase | Goal | What to do |
|-------|------|------------|
| **1. Setup** | Environment ready | `pip install -r requirements.txt`. Ensure GPU with enough VRAM (see [Hardware](#hardware-and-training-time) below). |
| **2. Smoke test** | Pipeline runs end-to-end | Run `run_full_flow(max_train_samples=20, num_epochs=1, output_dir="...")`. Check that training finishes and `output_dir` has adapter + tokenizer. Optionally test RAG with one query. |
| **3. Small run + eval** | First real training and metric | Use ~500–1k samples, 2 epochs. Load adapter from `output_dir`, run inference on 50–100 test examples, compute exact match (or your metric). |
| **4. Full run** | Production-style training | Full GSM8K train (~7.5k), 2–3 epochs. Evaluate on full test set. If you have VRAM for 7B, switch to Qwen2.5-7B and repeat. |
| **5. Deploy** | Serve model + RAG | Replace stub with your `CalculatorClient`. Serve base model + adapter (e.g. `PeftModel.from_pretrained`); in the API path, run the RAG layer after each generation. |

---

## Reference

### Code layout

```
qwen_math_flow/
├── download_model.py       # Download Qwen2.5-0.5B-Instruct (optional 4/8-bit)
├── load_dataset.py         # Load GSM8K, chat format, tokenize (labels = -100 on prompt)
├── lora_finetune.py        # LoRA + Trainer
├── rag_calculator.py       # Extract [CALC: ...], call calculator, augment context
└── external_calculator.py  # CalculatorClient, Stub, SafeEval
main.py                     # run_full_flow(), build_rag_only(), inference_with_rag()
requirements.txt
```

### Training pipeline (details)

1. **Download** — `download_qwen_25_07b()`: base model + tokenizer; optional 4-bit/8-bit and `device_map`.
2. **Data** — `load_math_dataset()` + `format_gsm8k_as_chat()` (or custom formatter) → `tokenize_math_dataset()` / `load_and_tokenize_math()`: chat template + labels masked so only the assistant reply is trained.
3. **LoRA** — `create_lora_model()` adds PEFT LoRA (default: q/k/v/o_proj); `run_finetune()` runs Trainer and saves adapter + tokenizer to `output_dir`.
4. **RAG** — Used at inference only. `RAGCalculatorLayer` finds placeholders (`[CALC: ...]`, ` ```calc ... ``` `, `<calculator>...</calculator>`), calls `CalculatorClient.evaluate()`, and injects results into the text.

**Prompt engineering:** Not required for training. The repo optionally wraps the user message with an instruction (e.g. “Solve the following math problem step by step”); you can change or remove it in `format_gsm8k_as_chat` / `format_math_as_chat` in `load_dataset.py`.

### Calculator

- **Interface:** `CalculatorClient` with `evaluate(expression: str) -> str` and `is_available() -> bool`.
- **Stub:** `StubCalculatorClient` — placeholder responses.
- **Local:** `SafeEvalCalculatorClient` — restricted math (numbers, +, -, *, /, sqrt, etc.); use only on trusted/sanitized input.

### Deployment

- **Model:** Load base, then `PeftModel.from_pretrained(base, output_dir)`. Serve with vLLM, TGI, or plain `transformers`.
- **Calculator:** In production, use a real `CalculatorClient` (API or sandboxed process), not raw user input in `SafeEvalCalculatorClient`.
- **RAG:** In the serving path, after each generation run the RAG layer (extract placeholders → call calculator → replace); optionally loop for multiple rounds. See `inference_with_rag()` in `main.py`.

### Hardware and training time

Two model options; single-GPU, LoRA (and QLoRA). Batch size and sequence length affect actual use.

**VRAM (guideline)**

| Model size | LoRA (fp16/bf16) | QLoRA (4-bit) | Full fine-tune |
|------------|------------------|---------------|-----------------|
| **~0.5B / 0.7B** | 1× GPU, 8–12 GB (e.g. RTX 3060 12GB) | 1× GPU, 6–8 GB | 1× GPU, 16 GB+ |
| **7B** | 1× GPU, 20–24 GB (e.g. RTX 4090 24GB) | 1× GPU, 12–16 GB | Multi-GPU or 40–80 GB (e.g. A100) |

**Estimated time** (LoRA, ~512 tokens, batch 4–8, 2–3 epochs)

| Model size | ~1k samples, 2 epochs | Full GSM8K (~7.5k), 3 epochs |
|------------|------------------------|------------------------------|
| **~0.5B / 0.7B** | ~5–15 min | ~30 min–1.5 h |
| **7B** | ~30 min–1.5 h | ~3–8 h |

QLoRA is often 20–40% slower than fp16 LoRA at similar VRAM.

### Fine-tuning techniques (optional)

- **LoRA / QLoRA** — Current setup; QLoRA reduces VRAM with 4-bit base.
- **DoRA** — Weight-decomposed LoRA; can improve quality.
- **Full fine-tune** — All parameters; more data and VRAM.
- **Multi-task mix** — Mix math with general instruction data to limit forgetting.
- **Gradient checkpointing** — Trade compute for memory.
- **Packing** — Pack short examples into fixed-length windows.
- **RL / DPO / RLHF** — Reward or preference on outputs. For math, **outcome-based RL** (reward correct final answer) is often enough; **RLHF** (human preferences) helps if you want to shape *how* the model reasons (e.g. step-by-step, use calculator). SFT often gives most of the gain for small models.
- **Curriculum** — Order by difficulty or length.

### Metrics

- **Exact match (EM) / Accuracy** — Final answer matches reference (normalize numbers); standard for GSM8K/MATH.
- **Pass@k** — At least one of k samples correct.
- **BLEU / ROUGE** — Overlap with reference reasoning; less reliable than EM for math.
- **Self-consistency** — Majority vote over samples; report EM on that.
- **Token / cost efficiency** — Tokens per problem or per correct answer.

---

## Usage examples

**Full pipeline + inference with RAG**

```python
from main import run_full_flow, inference_with_rag
from qwen_math_flow import StubCalculatorClient

result = run_full_flow(
    dataset_name="openai/gsm8k",
    max_train_samples=200,
    max_length=512,
    output_dir="output/lora_math",
    num_epochs=2,
    calculator_client=StubCalculatorClient(),
)
model, tokenizer, rag = result["peft_model"], result["tokenizer"], result["rag_layer"]

answer = inference_with_rag(
    model, tokenizer, rag,
    "What is 15 * 12? Use [CALC: 15*12] for the computation.",
    max_new_tokens=128,
    max_calculator_rounds=3,
)
```

**RAG only (e.g. with SafeEval calculator)**

```python
from main import build_rag_only
from qwen_math_flow import SafeEvalCalculatorClient

rag = build_rag_only(calculator_client=SafeEvalCalculatorClient())
augmented, results = rag.augment("Result: [CALC: 2+3*4]")
# results == [("2+3*4", "14")]
```

---

## Dependencies

See `requirements.txt`: `transformers`, `peft`, `datasets`, `torch`, `accelerate`, `huggingface_hub`; optional `bitsandbytes` for 4-bit LoRA.
