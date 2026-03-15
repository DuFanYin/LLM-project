from rag_calculator import RAGCalculatorLayer
from external_calculator import SafeEvalCalculatorClient

# Create calculator
calculator = SafeEvalCalculatorClient()

# Create RAG layer
rag = RAGCalculatorLayer(calculator)

# Test string containing a calculation
text = "What is the answer to [CALC: (15*8 + 42) / 3] ?"

# Run augmentation
result, calculations = rag.augment(text)

print("Original:", text)
print("Augmented:", result)
print("Calculations:", calculations)