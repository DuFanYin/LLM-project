"""
RAG layer that augments model outputs with calculator calls.

- Extracts expressions from text (e.g. [CALC: 2+3], ```calc ... ```).
- Calls external calculator and injects results into context.
- Can run multi-step: generate -> augment with calculator -> optionally continue.
"""

import re
from typing import Any, Callable, List, Optional, Tuple

from .external_calculator import CalculatorClient


class RAGCalculatorLayer:
    """
    RAG layer that:
    1. Takes model output (or user query).
    2. Extracts sub-expressions that need calculation (delimited patterns).
    3. Calls the external calculator for those expressions.
    4. Injects calculator results into the context for the next step.
    """

    # Patterns: (regex, group index for the expression)
    EXPRESSION_PATTERNS = [
        (re.compile(r"\[CALC:\s*([^\]]+)\]", re.IGNORECASE), 1),
        (re.compile(r"```calc\s*([^`]+)```", re.IGNORECASE), 1),
        (re.compile(r"<<calc>>\s*([^<]+)\s*<<\/calc>>", re.IGNORECASE), 1),
        (re.compile(r"<calculator>\s*([^<]+)\s*</calculator>", re.IGNORECASE), 1),
    ]

    def __init__(self, calculator: CalculatorClient):
        self.calculator = calculator

    def extract_expressions(self, text: str) -> List[str]:
        """
        Extract potential math expressions from text using known delimiters.
        Returns unique expressions in order of first occurrence.
        """
        seen: set = set()
        expressions: List[str] = []
        for pattern, group in self.EXPRESSION_PATTERNS:
            for m in pattern.finditer(text):
                expr = m.group(group).strip()
                if expr and expr not in seen:
                    seen.add(expr)
                    expressions.append(expr)
        return expressions

    def call_calculator(self, expression: str) -> str:
        """Call external calculator for one expression."""
        if not self.calculator.is_available():
            return "[Calculator unavailable]"
        return self.calculator.evaluate(expression)

    def augment(
        self,
        query_or_output: str,
        inject_into_context: bool = True,
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Run RAG: extract expressions, call calculator, optionally build augmented context.

        Args:
            query_or_output: Current model output or user query to scan for expressions.
            inject_into_context: If True, return an augmented string with results inlined.

        Returns:
            (augmented_text, list of (expression, result) pairs).
        """
        expressions = self.extract_expressions(query_or_output)
        results: List[Tuple[str, str]] = []
        for expr in expressions:
            result = self.call_calculator(expr)
            results.append((expr, result))

        if not inject_into_context or not results:
            return query_or_output, results

        augmented = query_or_output
        for expr, result in results:
            escaped = re.escape(expr)
            for pat_str, _ in [
                (r"\[CALC:\s*{}\s*\]", 1),
                (r"```calc\s*{}\s*```", 1),
                (r"<<calc>>\s*{}\s*<<\/calc>>", 1),
                (r"<calculator>\s*{}\s*</calculator>", 1),
            ]:
                try:
                    full_pat = re.compile(pat_str.format(escaped), re.IGNORECASE | re.DOTALL)
                    augmented, n = full_pat.subn(result, augmented, count=1)
                    if n:
                        break
                except re.error:
                    pass

        return augmented, results

    def run_with_rag(
        self,
        model_generate_fn: Callable[[str, Any], str],
        prompt: str,
        max_new_tokens: int = 256,
        max_calculator_rounds: int = 3,
        stop_if_no_calc: bool = True,
        **generate_kwargs: Any,
    ) -> str:
        """
        Generate with the model, then repeatedly run RAG (extract calc, call, augment)
        for up to max_calculator_rounds. If no calculator placeholders are found,
        returns the last model output.

        Args:
            model_generate_fn: (prompt_str, **kwargs) -> generated_string.
            prompt: Initial user/context prompt.
            max_new_tokens: Passed to generate.
            max_calculator_rounds: Max number of augment-and-continue rounds.
            stop_if_no_calc: If True, stop when no expressions are found.
            **generate_kwargs: Passed to model_generate_fn.

        Returns:
            Final string (augmented or last model output).
        """
        kwargs = {**generate_kwargs, "max_new_tokens": max_new_tokens}
        current = prompt
        last_output = ""

        for _ in range(max_calculator_rounds):
            last_output = model_generate_fn(current, **kwargs)
            augmented, results = self.augment(last_output, inject_into_context=True)
            if not results:
                if stop_if_no_calc:
                    return last_output
                return augmented
            current = augmented

        return current

    def generate_with_rag(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 256,
        max_calculator_rounds: int = 3,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """
        Run generate-with-RAG using a Hugging Face model and tokenizer.

        prompt: string (will be tokenized and passed to model.generate).
        Model and tokenizer are used to implement model_generate_fn internally.
        """
        if pad_token_id is None and tokenizer.pad_token_id is not None:
            pad_token_id = tokenizer.pad_token_id
        elif pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = tokenizer.eos_token_id

        gen_params: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
        }
        # Only add sampling params if the model's generation_config supports them,
        # to avoid \"invalid generation flag\" warnings in newer transformers.
        gen_cfg = getattr(model, "generation_config", None)
        if temperature is not None and gen_cfg is not None and hasattr(gen_cfg, "temperature"):
            gen_params["temperature"] = temperature
        if top_p is not None and gen_cfg is not None and hasattr(gen_cfg, "top_p"):
            gen_params["top_p"] = top_p
        gen_params.update(kwargs)

        def _generate(text: str, max_new_tokens: int = max_new_tokens, **override: Any) -> str:
            merged = {**gen_params, "max_new_tokens": max_new_tokens, **override}
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            out = model.generate(**inputs, **merged)
            return tokenizer.decode(out[0], skip_special_tokens=True)

        return self.run_with_rag(
            _generate,
            prompt,
            max_new_tokens=max_new_tokens,
            max_calculator_rounds=max_calculator_rounds,
        )
