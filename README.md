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

The flow is split into three steps; you can run any step alone (no need to run the full pipeline):

| Step | Command | Description |
|------|---------|-------------|
| **1. RAG test** | `python main.py rag_test` | Test calculator + RAG parse/replace only; no model load. |
| **2. Train and save** | `python main.py train_save` | Train only and save adapter/tokenizer to `output/lora_math`. |
| **3. Inference** | `python main.py inference --adapter-dir output/lora_math` | Load from saved dir and run one inference with RAG. |

Run all three in sequence (small scale): `python main.py all` (default 20 samples, 1 epoch; smoke test only).

---

## Step-by-step: how to run the project

| Phase | Goal | What to do |
|-------|------|------------|
| **1. RAG test** | Verify calculator and RAG | `python main.py rag_test` or `from main import run_rag_test; run_rag_test()`. |
| **2. Train and save** | Get adapter | `python main.py train_save` or call `train_and_save(output_dir=..., max_train_samples=..., num_epochs=...)`. For larger runs, change `--max-train-samples` / `--epochs` or pass args. |
| **3. Inference** | Use saved params for inference | `python main.py inference --adapter-dir output/lora_math` or `run_inference(adapter_dir="output/lora_math", user_query="...")`. |
| **4. Scale up** | Full training + eval | In code call `train_and_save(max_train_samples=None, num_epochs=3)` etc.; then evaluate on test set (e.g. EM). |
| **5. Deploy** | Serve model + RAG | Load with `load_for_inference(adapter_dir)`, plug in your `CalculatorClient` and RAG layer in your API. |

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
main.py                     # 1. run_rag_test()  2. train_and_save()  3. load_for_inference(), run_inference(), inference_with_rag()
requirements.txt
```

### Training pipeline (details)

1. **Download** — `download_qwen_25_07b()`: base model + tokenizer; optional 4-bit/8-bit and `device_map`.
2. **Data** — `load_math_dataset()` + `format_gsm8k_as_chat()` (or custom formatter) → `tokenize_math_dataset()` / `load_and_tokenize_math()`: chat template + labels masked so only the assistant reply is trained.
3. **LoRA** — `create_lora_model()` adds PEFT LoRA (default: q/k/v/o_proj); `run_finetune()` runs Trainer and saves adapter + tokenizer to `output_dir`. For inference only later, use `load_for_inference(adapter_dir)` (see Usage examples).
4. **RAG** — Used at inference only. `RAGCalculatorLayer` finds placeholders (`[CALC: ...]`, ` ```calc ... ``` `, `<calculator>...</calculator>`), calls `CalculatorClient.evaluate()`, and injects results into the text.

**Prompt engineering:** Not required for training. The repo optionally wraps the user message with an instruction (e.g. "Solve the following math problem step by step"); you can change or remove it in `format_gsm8k_as_chat` / `format_math_as_chat` in `load_dataset.py`.

### Calculator

- **Interface:** `CalculatorClient` with `evaluate(expression: str) -> str` and `is_available() -> bool`.
- **Stub:** `StubCalculatorClient` — placeholder responses.
- **Local:** `SafeEvalCalculatorClient` — restricted math (numbers, +, -, *, /, sqrt, etc.); use only on trusted/sanitized input.

### Deployment

- **Model:** Load base, then `PeftModel.from_pretrained(base, output_dir)`. Serve with vLLM, TGI, or plain `transformers`.
- **Calculator:** In production, use a real `CalculatorClient` (API or sandboxed process), not raw user input in `SafeEvalCalculatorClient`.
- **RAG:** In the serving path, after each generation run the RAG layer (extract placeholders → call calculator → replace); optionally loop for multiple rounds. See `inference_with_rag()` in `main.py`.

### Datasets (reference)

| Dataset | Examples | Typical file size |
|---------|----------|-------------------|
| **GSM8K** | ~9K | ~6–8 MB |
| **ASDiv** | ~2.3K | ~1–2 MB |
| **MetaMathQA** | ~395K | ~300–500 MB |
| **OpenMathInstruct** | ~1.8M | ~1.5–3 GB |

**Approximate training time per dataset** (LoRA, 3 epochs, ~512 tokens, batch 4–8; single GPU). Actual time depends on hardware, batch size, and sequence length.

| Dataset | ~0.5B / 0.7B | 7B |
|---------|----------------|-----|
| **ASDiv** (~2.3K) | ~10–25 min | ~45 min–2 h |
| **GSM8K** (~9K) | ~30 min–1.5 h | ~3–8 h |
| **MetaMathQA** (~395K) | ~20–65 h | ~4–14 days |
| **OpenMathInstruct** (~1.8M) | ~4–12 days | ~3–8 weeks |

### Hardware and training time

Two model options; single-GPU, LoRA (and QLoRA). Batch size and sequence length affect actual use.

**VRAM (guideline)**

| Model size | LoRA (fp16/bf16) | QLoRA (4-bit) | Full fine-tune |
|------------|------------------|---------------|-----------------|
| **~0.5B / 0.7B** | 1× GPU, 8–12 GB (e.g. RTX 3060 12GB) | 1× GPU, 6–8 GB | 1× GPU, 16 GB+ |
| **7B** | 1× GPU, 20–24 GB (e.g. RTX 4090 24GB) | 1× GPU, 12–16 GB | Multi-GPU or 40–80 GB (e.g. A100) |

**Estimated time** (LoRA, ~512 tokens, batch 4–8, 2–3 epochs)

| Model size | ~1k samples, 2 epochs | Full GSM8K (~9K), 3 epochs |
|------------|------------------------|-----------------------------|
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

**1. RAG test only (no model load)**

```python
from main import run_rag_test
run_rag_test(use_safe_eval=True)  # Use SafeEval for real math; False for Stub
```

**2. Train and save only**

```python
from main import train_and_save
train_and_save(
    output_dir="output/lora_math",
    max_train_samples=500,
    num_epochs=2,
)
# Adapter + tokenizer written to output_dir
```

**3. Inference only (load from saved dir)**

```python
from main import run_inference, load_for_inference, inference_with_rag, build_rag_only
from qwen_math_flow import SafeEvalCalculatorClient

# Option A: one-liner
run_inference(adapter_dir="output/lora_math", user_query="What is [CALC: 7*8]?")

# Option B: compose RAG + inference yourself
model, tokenizer = load_for_inference(adapter_dir="output/lora_math")
rag = build_rag_only(calculator_client=SafeEvalCalculatorClient())
answer = inference_with_rag(model, tokenizer, rag, "What is [CALC: 15*12]?")
```

---

## Dependencies

See `requirements.txt`: `transformers`, `peft`, `datasets`, `torch`, `accelerate`, `huggingface_hub`; optional `bitsandbytes` for 4-bit LoRA.
