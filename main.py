"""
Main flow: download Qwen2.5 0.7B → load math dataset → add LoRA → fine-tune → add RAG calculator.

Wires the full pipeline including tokenization and optional inference with RAG.
"""

from pathlib import Path
from typing import Any, Optional

from qwen_math_flow.download_model import download_qwen_25_07b
from qwen_math_flow.load_dataset import load_math_dataset, load_and_tokenize_math, format_gsm8k_as_chat
from qwen_math_flow.lora_finetune import create_lora_model, run_finetune
from qwen_math_flow.rag_calculator import RAGCalculatorLayer
from qwen_math_flow.external_calculator import CalculatorClient, StubCalculatorClient


def run_full_flow(
    model_cache_dir: Optional[str] = None,
    dataset_name: str = "openai/gsm8k",
    dataset_split: str = "train",
    max_train_samples: Optional[int] = 100,
    max_length: int = 512,
    lora_r: int = 8,
    lora_alpha: int = 16,
    output_dir: str = "output/lora_math",
    num_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    load_in_4bit: bool = False,
    calculator_client: Optional[CalculatorClient] = None,
) -> dict:
    """
    Execute the full pipeline:

    1. Download Qwen2.5 0.5B Instruct and tokenizer.
    2. Load math dataset (e.g. GSM8K), format as chat, tokenize with label masking.
    3. Add LoRA to the model and run fine-tuning.
    4. Build RAG layer with external calculator (stub if not provided).

    Returns:
        Dict with keys: model, tokenizer, peft_model, train_metrics, rag_layer.
    """
    # --- 1. Download model ---
    model, tokenizer = download_qwen_25_07b(
        cache_dir=model_cache_dir,
        load_in_4bit=load_in_4bit,
        device_map="auto" if load_in_4bit else None,
    )

    # --- 2. Load and tokenize math dataset ---
    tokenized_train = load_and_tokenize_math(
        tokenizer,
        name=dataset_name,
        split=dataset_split,
        max_samples=max_train_samples,
        max_length=max_length,
        message_formatter=format_gsm8k_as_chat,
    )

    # --- 3. LoRA + fine-tune ---
    peft_model = create_lora_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        use_4bit_or_8bit=load_in_4bit,
    )
    train_metrics = run_finetune(
        peft_model,
        tokenizer,
        tokenized_train,
        output_dir=output_dir,
        num_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
    )

    # --- 4. RAG layer with calculator (external call assumed provided) ---
    calc_client = calculator_client or StubCalculatorClient()
    rag_layer = RAGCalculatorLayer(calc_client)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "peft_model": peft_model,
        "train_metrics": train_metrics,
        "rag_layer": rag_layer,
    }


def build_rag_only(
    calculator_client: Optional[CalculatorClient] = None,
) -> RAGCalculatorLayer:
    """
    Build only the RAG calculator layer (e.g. to wrap an existing model).
    External calculator is assumed provided; otherwise uses stub.
    """
    client = calculator_client or StubCalculatorClient()
    return RAGCalculatorLayer(client)


def inference_with_rag(
    model: Any,
    tokenizer: Any,
    rag_layer: RAGCalculatorLayer,
    user_query: str,
    max_new_tokens: int = 256,
    max_calculator_rounds: int = 3,
    do_sample: bool = False,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> str:
    """
    Run one inference: format user query as chat, generate with model,
    then run RAG (extract [CALC: ...] / ```calc ... ```, call calculator, augment).
    Repeats up to max_calculator_rounds if new expressions appear after augmentation.
    """
    messages = [{"role": "user", "content": user_query}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return rag_layer.generate_with_rag(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        max_calculator_rounds=max_calculator_rounds,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )


if __name__ == "__main__":
    # Example: run full flow with defaults (stub calculator)
    result = run_full_flow(
        dataset_name="openai/gsm8k",
        max_train_samples=10,
        num_epochs=1,
        output_dir="output/lora_math",
        calculator_client=StubCalculatorClient(),
    )
    print("Flow complete. Keys:", list(result.keys()))
