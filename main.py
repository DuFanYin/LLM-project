"""
Three-step flow: 1. RAG test -> 2. Train and save -> 3. Inference.
Each step can be run separately; no need to run the full pipeline.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from qwen_math_flow.download_model import download_qwen_25_07b
from qwen_math_flow.external_calculator import (
    CalculatorClient,
    SafeEvalCalculatorClient,
    StubCalculatorClient,
)
from qwen_math_flow.load_dataset import format_gsm8k_as_chat, load_and_tokenize_math
from qwen_math_flow.lora_finetune import create_lora_model, run_finetune
from qwen_math_flow.rag_calculator import RAGCalculatorLayer


# -----------------------------------------------------------------------------
# 1. RAG test (no model load; only verify calculator + RAG parse and replace)
# -----------------------------------------------------------------------------


def build_rag_only(
    calculator_client: Optional[CalculatorClient] = None,
) -> RAGCalculatorLayer:
    """Build RAG layer only, for testing or later inference."""
    client = calculator_client or StubCalculatorClient()
    return RAGCalculatorLayer(client)


def run_rag_test(
    calculator_client: Optional[CalculatorClient] = None,
    use_safe_eval: bool = True,
) -> None:
    """
    Run RAG test only: augment a few strings containing [CALC: ...] and check
    calculator and replacement. No model load, no training.
    """
    client = calculator_client or (SafeEvalCalculatorClient() if use_safe_eval else StubCalculatorClient())
    rag = build_rag_only(calculator_client=client)

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
    print("RAG test done.\n")


# -----------------------------------------------------------------------------
# 2. Train and save (train only, write adapter/tokenizer; no inference)
# -----------------------------------------------------------------------------


def train_and_save(
    output_dir: str = "output/lora_math",
    dataset_name: str = "openai/gsm8k",
    dataset_split: str = "train",
    max_train_samples: Optional[int] = 100,
    max_length: int = 512,
    num_epochs: int = 3,
    lora_r: int = 8,
    lora_alpha: int = 16,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    load_in_4bit: bool = False,
    model_cache_dir: Optional[str] = None,
) -> dict:
    """
    Train only and save: download model -> load data -> LoRA train -> write
    adapter and tokenizer to output_dir. No RAG, no inference.
    """
    model, tokenizer = download_qwen_25_07b(
        cache_dir=model_cache_dir,
        load_in_4bit=load_in_4bit,
        device_map="auto" if load_in_4bit else None,
    )
    tokenized_train = load_and_tokenize_math(
        tokenizer,
        name=dataset_name,
        split=dataset_split,
        max_samples=max_train_samples,
        max_length=max_length,
        message_formatter=format_gsm8k_as_chat,
    )
    peft_model = create_lora_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        use_4bit_or_8bit=load_in_4bit,
    )
    metrics = run_finetune(
        peft_model,
        tokenizer,
        tokenized_train,
        output_dir=output_dir,
        num_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
    )
    print(f"Train done. Adapter + tokenizer saved to: {output_dir}\n")
    return metrics


# -----------------------------------------------------------------------------
# 3. Inference (load saved adapter, then run inference with RAG)
# -----------------------------------------------------------------------------


def load_for_inference(
    adapter_dir: str,
    base_model_id: Optional[str] = None,
    device_map: Optional[str] = "auto",
    load_in_4bit: bool = False,
) -> Tuple[Any, Any]:
    """
    Load saved LoRA adapter and tokenizer from adapter_dir for inference only.
    """
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    if base_model_id is None:
        config_path = adapter_path / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            base_model_id = config.get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
        else:
            base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    base_kwargs = dict(device_map=device_map, trust_remote_code=True)
    if load_in_4bit:
        base_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        base_kwargs["torch_dtype"] = "auto"
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **base_kwargs)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    peft_model.eval()
    return peft_model, tokenizer


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
    """Run inference with RAG: generate, then call calculator for [CALC: ...] in output and replace."""
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


def run_inference(
    adapter_dir: str,
    user_query: str = "What is [CALC: 15*12]?",
    load_in_4bit: bool = False,
    calculator_client: Optional[CalculatorClient] = None,
    max_new_tokens: int = 128,
) -> str:
    """
    Run inference only: load model and tokenizer from adapter_dir, build RAG layer, run one query.
    """
    model, tokenizer = load_for_inference(adapter_dir, load_in_4bit=load_in_4bit)
    client = calculator_client or SafeEvalCalculatorClient()
    rag = build_rag_only(calculator_client=client)
    answer = inference_with_rag(model, tokenizer, rag, user_query, max_new_tokens=max_new_tokens)
    print(f"Inference query: {user_query}")
    print(f"Inference answer: {answer}\n")
    return answer


# -----------------------------------------------------------------------------
# Entry point: run a single step or all
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1. RAG test  2. Train and save  3. Inference")
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=["rag_test", "train_save", "inference", "all"],
        help="rag_test= RAG only; train_save= train and save; inference= inference only; all= run 1->2->3 (small scale)",
    )
    parser.add_argument("--adapter-dir", default="output/lora_math", help="Output dir for training / adapter dir for inference")
    parser.add_argument("--max-train-samples", type=int, default=20, help="Samples for train_save (default 20 when using all)")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs for train_save (default 1 when using all)")
    args = parser.parse_args()

    if args.step == "rag_test" or args.step == "all":
        run_rag_test(use_safe_eval=True)
    if args.step == "train_save" or args.step == "all":
        train_and_save(
            output_dir=args.adapter_dir,
            max_train_samples=args.max_train_samples,
            num_epochs=args.epochs,
        )
    if args.step == "inference" or args.step == "all":
        run_inference(adapter_dir=args.adapter_dir, user_query="What is [CALC: 7*8]?")
