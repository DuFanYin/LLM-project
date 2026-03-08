"""
Load a math dataset for fine-tuning (e.g. GSM8K, MATH) and tokenize for causal LM.

Supports Qwen chat format and label masking (train only on assistant responses).
"""

from typing import Any, Callable, Dict, List, Optional, Union

from datasets import load_dataset as hf_load


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_math_dataset(
    name: str = "openai/gsm8k",
    subset: str = "main",
    split: Union[str, List[str]] = "train",
    streaming: bool = False,
    max_samples: Optional[int] = None,
    prompt_template: Optional[Callable[[Dict[str, Any]], str]] = None,
    **kwargs: Any,
):
    """
    Load a math QA dataset from Hugging Face datasets.

    Default: GSM8K (grade school math 8K). Alternatives: "competition_math",
    "lighteval/MATH", etc.

    Args:
        name: Dataset name on Hugging Face (e.g. "openai/gsm8k").
        subset: Config name for datasets with multiple configs.
        split: "train", "test", or ["train", "validation"].
        streaming: Whether to stream the dataset.
        max_samples: Cap number of examples (None = use all).
        prompt_template: Optional fn(sample) -> str to format each example.
            If None, uses default GSM8K-style "Question: ... Answer: ...".
        **kwargs: Passed to datasets.load_dataset().

    Returns:
        If prompt_template is used and not streaming: (dataset, list of formatted strings).
        Otherwise: dataset only.
    """
    dataset = hf_load(name, subset, split=split, streaming=streaming, **kwargs)

    if max_samples is not None and not streaming:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    if prompt_template is None and name == "openai/gsm8k":
        def _default_gsm8k(sample: Dict[str, Any]) -> str:
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            return f"Question: {question}\nAnswer: {answer}"

        prompt_template = _default_gsm8k

    if prompt_template is not None and not streaming:
        formatted = [prompt_template(s) for s in dataset]
        return dataset, formatted

    return dataset


# ---------------------------------------------------------------------------
# Chat-format helpers for Qwen
# ---------------------------------------------------------------------------


def format_gsm8k_as_chat(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Format one GSM8K example as Qwen chat messages: user = question, assistant = answer.
    """
    question = sample.get("question", "")
    answer = sample.get("answer", "")
    return [
        {"role": "user", "content": f"Solve the following math problem step by step.\n\n{question}"},
        {"role": "assistant", "content": answer},
    ]


def format_math_as_chat(
    question_key: str = "question",
    answer_key: str = "answer",
    system_prompt: Optional[str] = None,
) -> Callable[[Dict[str, Any]], List[Dict[str, str]]]:
    """Build a formatter that turns a dict with question/answer keys into chat messages."""

    def _formatter(sample: Dict[str, Any]) -> List[Dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": sample.get(question_key, "")})
        messages.append({"role": "assistant", "content": sample.get(answer_key, "")})
        return messages

    return _formatter


# ---------------------------------------------------------------------------
# Tokenization for training (input_ids + labels, -100 on prompt)
# ---------------------------------------------------------------------------


def tokenize_math_dataset(
    dataset: Any,
    tokenizer: Any,
    message_formatter: Callable[[Dict[str, Any]], List[Dict[str, str]]],
    max_length: int = 512,
    max_samples: Optional[int] = None,
    padding: str = "longest",
    truncation: bool = True,
    return_tensors: Optional[str] = None,
) -> Any:
    """
    Tokenize a Hugging Face dataset for causal LM fine-tuning.

    Each example is converted to chat messages via message_formatter(sample),
    then tokenized with the tokenizer's chat template. Labels are set to -100
    for all token positions except the assistant reply, so loss is computed only
    on the model's answer.

    Args:
        dataset: Hugging Face Dataset (e.g. from load_math_dataset).
        tokenizer: Pre-trained tokenizer with chat_template (e.g. Qwen).
        message_formatter: sample -> list of {role, content} dicts.
        max_length: Max sequence length (prompt + response).
        max_samples: Optional cap on number of examples.
        padding: "longest" or "max_length".
        truncation: Whether to truncate to max_length.
        return_tensors: None for Dataset with lists; "pt" not used for Dataset.

    Returns:
        Dataset with columns input_ids, attention_mask, labels (lists of ints).
    """
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def _tokenize_one(example: Dict[str, Any]) -> Dict[str, List[int]]:
        messages = message_formatter(example)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=None,
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", [1] * len(input_ids))

        # Find assistant turn start: after the last "assistant" prompt token(s).
        # Qwen format is typically <|im_start|>assistant\n...<|im_end|>.
        # We set labels to -100 for prompt, and real token ids for the assistant part.
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_encoded = tokenizer(
            prompt_text,
            max_length=max_length,
            truncation=truncation,
            return_tensors=None,
        )
        prompt_len = len(prompt_encoded["input_ids"])
        labels = [-100] * len(input_ids)
        for i in range(prompt_len, len(input_ids)):
            labels[i] = input_ids[i]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized = dataset.map(
        _tokenize_one,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        num_proc=1,
    )
    return tokenized


def load_and_tokenize_math(
    tokenizer: Any,
    name: str = "openai/gsm8k",
    split: str = "train",
    max_samples: Optional[int] = None,
    max_length: int = 512,
    message_formatter: Optional[Callable[[Dict[str, Any]], List[Dict[str, str]]]] = None,
) -> Any:
    """
    Convenience: load GSM8K (or other) and tokenize in one call.

    Returns:
        Tokenized Dataset with input_ids, attention_mask, labels.
    """
    raw, _ = load_math_dataset(name=name, split=split, max_samples=max_samples)
    formatter = message_formatter or format_gsm8k_as_chat
    return tokenize_math_dataset(
        raw,
        tokenizer,
        formatter,
        max_length=max_length,
        max_samples=None,
    )
