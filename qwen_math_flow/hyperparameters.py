"""
Hyperparameters and run configuration.
Edit this file to switch behaviour (no CLI arguments).
"""

from typing import Optional

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Supported: any path string
ADAPTER_DIR = "output/lora_math"

# -----------------------------------------------------------------------------
# Training — GSM8K only (openai/gsm8k, subset main)
# -----------------------------------------------------------------------------
# Supported: e.g. "openai/gsm8k"
DATASET_NAME = "openai/gsm8k"
# Supported: "train" | "test" | "validation"
DATASET_SPLIT = "train"
# Supported: int (cap) or None (full split)
MAX_TRAIN_SAMPLES: Optional[int] = 100

# -----------------------------------------------------------------------------
# Training — optimizer / schedule
# -----------------------------------------------------------------------------
# Supported: int (e.g. 1, 2, 3)
NUM_EPOCHS = 2
# Supported: int (e.g. 512, 1024)
MAX_LENGTH = 512
# Supported: int (e.g. 4, 8)
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
# Supported: float (e.g. 2e-5)
LEARNING_RATE = 2e-5

# -----------------------------------------------------------------------------
# Training — LoRA
# -----------------------------------------------------------------------------
# Supported: int (e.g. 8, 16)
LORA_R = 8
LORA_ALPHA = 16

# -----------------------------------------------------------------------------
# Training — model loading
# -----------------------------------------------------------------------------
# Supported: True (QLoRA, less VRAM) | False
LOAD_IN_4BIT = False
# Supported: str path or None
MODEL_CACHE_DIR: Optional[str] = None

# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
# Supported: int
MAX_NEW_TOKENS = 128
# Supported: True | False
LOAD_IN_4BIT_INFERENCE = False
# Random questions from GSM8K (openai/gsm8k, subset main) — used by Main.ipynb inference cell
INFERENCE_NUM_QUESTIONS = 5
INFERENCE_RANDOM_SEED = 42
# Supported: "train" | "test" (test avoids overlap with typical training on train)
INFERENCE_QUESTION_SPLIT = "test"

# -----------------------------------------------------------------------------
# RAG test
# -----------------------------------------------------------------------------
# Supported: True (real eval) | False (stub)
USE_SAFE_EVAL_RAG_TEST = True
