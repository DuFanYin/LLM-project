"""
Download Qwen2.5 0.5B/0.7B from Hugging Face and optionally cache locally.

Uses Qwen2.5-0.5B-Instruct (smallest available on HF; no 0.7B exists).
"""

from pathlib import Path
from typing import Any, Optional, Tuple

# Lazy imports: transformers, optional bitsandbytes for 4-bit


def download_qwen_25_07b(
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
    Download Qwen2.5 small instruct model and tokenizer from Hugging Face.

    Model: Qwen/Qwen2.5-0.5B-Instruct (HF does not publish 0.7B; 0.5B is smallest).

    Args:
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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
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
        from transformers import BitsAndBytesConfig
        import torch

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
