"""
Qwen math flow: download -> LoRA fine-tune -> optional RAG calculator.
"""

from .download_model import download_qwen_2_15b
from .load_dataset import (
    build_deterministic_math_prompt,
    format_gsm8k_as_deterministic_json_chat,
    format_gsm8k_as_chat,
    load_math_dataset,
    load_and_tokenize_math,
    format_math_as_chat,
    tokenize_math_dataset,
)
from .lora_finetune import create_lora_model, run_finetune
from .rag_calculator import RAGCalculatorLayer
from .external_calculator import CalculatorClient, StubCalculatorClient, SafeEvalCalculatorClient

__all__ = [
    "download_qwen_2_15b",
    "build_deterministic_math_prompt",
    "format_gsm8k_as_deterministic_json_chat",
    "load_math_dataset",
    "load_and_tokenize_math",
    "format_gsm8k_as_chat",
    "format_math_as_chat",
    "tokenize_math_dataset",
    "create_lora_model",
    "run_finetune",
    "RAGCalculatorLayer",
    "CalculatorClient",
    "StubCalculatorClient",
    "SafeEvalCalculatorClient",
]
