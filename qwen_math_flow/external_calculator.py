"""
External calculator interface.

Provides:
- CalculatorClient: abstract interface for external calculator calls.
- StubCalculatorClient: stub for tests / when external service is provided elsewhere.
- SafeEvalCalculatorClient: local fallback that evaluates a restricted set of math
  expressions (no arbitrary code). Use only for trusted or sanitized input.
"""

import ast
import math
from abc import ABC, abstractmethod


class CalculatorClient(ABC):
    """Abstract interface for any calculator implementation."""

    @abstractmethod
    def evaluate(self, expression: str) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


class StubCalculatorClient(CalculatorClient):
    """
    Simple stub calculator.
    Useful for testing the RAG pipeline without doing real math.
    """

    def evaluate(self, expression: str) -> str:
        return f"[CALC_RESULT:{expression}]"

    def is_available(self) -> bool:
        return True


def _safe_eval_allowed_nodes() -> tuple:
    """Node types allowed in calculator expressions (Python 3.8+ uses ast.Constant; ast.Num removed in 3.14+)."""
    nodes = [
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Call,
        ast.Load,
        ast.Name,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.FloorDiv,
        ast.USub,
        ast.UAdd,
    ]
    # Legacy numeric literals (deprecated; not present on some Python versions)
    if hasattr(ast, "Num"):
        nodes.append(ast.Num)
    return tuple(nodes)


class SafeEvalCalculatorClient(CalculatorClient):
    """
    Secure evaluator for math expressions.

    Supports:
    - + - * / ** % //
    - parentheses
    - math functions (sqrt, sin, cos, log, etc.)
    """

    ALLOWED_NAMES = {
        k: getattr(math, k) for k in dir(math) if not k.startswith("_")
    }

    ALLOWED_NAMES.update({
        "abs": abs,
        "round": round,
        "pow": pow,
    })

    ALLOWED_NODES = _safe_eval_allowed_nodes()

    def is_available(self) -> bool:
        return True

    def evaluate(self, expression: str) -> str:
        try:
            node = ast.parse(expression, mode="eval")

            if not self._is_safe(node):
                return "[Invalid or unsafe expression]"

            compiled = compile(node, "<calculator>", "eval")
            result = eval(compiled, {"__builtins__": {}}, self.ALLOWED_NAMES)

            return str(result)

        except Exception as e:
            return f"[Calculation error: {e}]"

    def _is_safe(self, node):
        for subnode in ast.walk(node):
            if not isinstance(subnode, self.ALLOWED_NODES):
                return False

            if isinstance(subnode, ast.Constant):
                # Only numeric constants (bool is int subclass in Python)
                if not isinstance(subnode.value, (int, float, complex)):
                    return False

            if isinstance(subnode, ast.Name):
                if subnode.id not in self.ALLOWED_NAMES:
                    return False

        return True