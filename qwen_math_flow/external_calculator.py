"""
External calculator interface.

Provides:
- CalculatorClient: abstract interface for external calculator calls.
- StubCalculatorClient: stub for tests / when external service is provided elsewhere.
- SafeEvalCalculatorClient: local fallback that evaluates a restricted set of math
  expressions (no arbitrary code). Use only for trusted or sanitized input.
"""

import math
import re
from abc import ABC, abstractmethod
from typing import Any, Optional, Set


class CalculatorClient(ABC):
    """
    Abstract interface for external calculator calls.
    Implementations can be REST API, subprocess, or local safe eval.
    """

    @abstractmethod
    def evaluate(self, expression: str) -> str:
        """
        Evaluate a mathematical expression and return the result as string.

        Args:
            expression: e.g. "2 + 3 * 4", "sqrt(16)"

        Returns:
            Result string, e.g. "14", "4.0", or error message.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the calculator service is available."""
        pass


class StubCalculatorClient(CalculatorClient):
    """
    Stub implementation for development/testing.
    Replace with real client when external calculator is provided.
    """

    def evaluate(self, expression: str) -> str:
        return f"[STUB] result for: {expression}"

    def is_available(self) -> bool:
        return True


# Allowed names for SafeEvalCalculatorClient (whitelist only)
_SAFE_NAMES: Set[str] = {
    "abs", "round", "min", "max", "sum", "pow",
    "sqrt", "sin", "cos", "tan", "log", "log10", "log2", "exp",
    "pi", "e", "inf", "ceil", "floor",
}
_SAFE_MODULE = {k: getattr(math, k) for k in dir(math) if k in _SAFE_NAMES}
# Add builtins we allow
_SAFE_MODULE["abs"] = abs
_SAFE_MODULE["round"] = round
_SAFE_MODULE["min"] = min
_SAFE_MODULE["max"] = max
_SAFE_MODULE["sum"] = sum
_SAFE_MODULE["pow"] = pow


class SafeEvalCalculatorClient(CalculatorClient):
    """
    Local calculator that evaluates a restricted set of math expressions.

    Allowed: numbers, +, -, *, /, //, %, **, (), comparison, and a whitelist of
    math functions (sqrt, sin, cos, log, exp, etc.). No arbitrary code or
    attribute access. Use only for trusted or sanitized input; do not pass
    user-controlled strings from untrusted sources without additional checks.
    """

    def __init__(self, max_length: int = 200):
        self.max_length = max_length

    def is_available(self) -> bool:
        return True

    def evaluate(self, expression: str) -> str:
        expr = expression.strip()
        if len(expr) > self.max_length:
            return "[Error: expression too long]"
        # Remove common noise
        expr = re.sub(r"\s+", " ", expr)
        # Only allow safe characters (digits, operators, parentheses, comma, dot, letters for math names)
        if not re.match(r"^[\d\s+\-*/().,%a-zA-Z_]+$", expr):
            return "[Error: disallowed characters]"
        try:
            result = eval(expr, {"__builtins__": {}}, _SAFE_MODULE)
            if result is None:
                return "[Error: no value]"
            if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
                return str(result)
            return str(result)
        except Exception as e:
            return f"[Error: {type(e).__name__}: {e}]"
