"""
Qwen2.5 math flow: download -> dataset -> LoRA fine-tune -> RAG calculator.
"""

from .download_model import download_qwen_25_07b
from .load_dataset import (
    load_math_dataset,
    load_and_tokenize_math,
    format_gsm8k_as_chat,
    format_math_as_chat,
    tokenize_math_dataset,
)
from .lora_finetune import create_lora_model, run_finetune
from .rag_calculator import RAGCalculatorLayer
from .external_calculator import CalculatorClient, StubCalculatorClient, SafeEvalCalculatorClient

__all__ = [
    "download_qwen_25_07b",
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
