"""
Download a Qwen causal LM from Hugging Face and optionally cache locally.
"""

from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def download_qwen_25_07b(
    model_id: str = "Qwen/Qwen2-1.5B",
    cache_dir: Optional[str] = None,
    revision: str = "main",
    trust_remote_code: bool = True,
    torch_dtype: Optional[Any] = None,
    device_map: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    bnb_4bit_compute_dtype: Optional[Any] = None,
    bnb_4bit_quant_type: str = "nf4",
) -> Tuple[Any, Any]:
    """
    Download a Qwen model and tokenizer from Hugging Face.

    Args:
        model_id: Hugging Face model repo id (for example "Qwen/Qwen2-1.5B").
        cache_dir: Directory to cache model weights. If None, uses HF default.
        revision: Git revision (branch/tag) of the model repo.
        trust_remote_code: Allow custom modeling code from the hub.
        torch_dtype: dtype for model (e.g. torch.bfloat16). None => "auto".
        device_map: Device map for multi-GPU/CPU ("auto", "cuda:0", None).
        load_in_4bit: Load base model in 4-bit quantization (saves VRAM).
        load_in_8bit: Load base model in 8-bit quantization.
        bnb_4bit_compute_dtype: Compute dtype for 4-bit (e.g. torch.bfloat16).
        bnb_4bit_quant_type: "nf4" or "fp4" for 4-bit.

    Returns:
        Tuple of (model, tokenizer).
    """
    cache_path = Path(cache_dir) if cache_dir else None
    dtype = torch_dtype if torch_dtype is not None else "auto"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=str(cache_path) if cache_path else None,
        trust_remote_code=trust_remote_code,
    )

    model_kwargs: dict = {
        "revision": revision,
        "cache_dir": str(cache_path) if cache_path else None,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    if load_in_4bit or load_in_8bit:
        compute_dtype = bnb_4bit_compute_dtype or torch.bfloat16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs.pop("torch_dtype", None)

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    return model, tokenizer
