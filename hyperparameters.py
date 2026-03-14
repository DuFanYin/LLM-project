"""
Hyperparameters and run configuration.
Edit this file to switch behaviour (no CLI arguments).
"""

from typing import Optional

# -----------------------------------------------------------------------------
# Run control
# -----------------------------------------------------------------------------
# Supported: "rag_test" | "train_save" | "inference" | "all"
RUN_STEP = "train_save"

# Supported: any path string
ADAPTER_DIR = "output/lora_math"

# -----------------------------------------------------------------------------
# Training — single-dataset (used when USE_MULTI_DATASET is False)
# -----------------------------------------------------------------------------
# Supported: e.g. "openai/gsm8k", "gsm8k"
DATASET_NAME = "openai/gsm8k"
# Supported: "train" | "test" | "validation"
DATASET_SPLIT = "train"
# Supported: int (cap) or None (full split)
MAX_TRAIN_SAMPLES: Optional[int] = 20

# -----------------------------------------------------------------------------
# Training — multi-dataset
# -----------------------------------------------------------------------------
# Supported: True = 4-dataset mix (GSM8K+ASDiv+MetaMathQA+OpenMathInstruct); False = single dataset only
USE_MULTI_DATASET = True
# Supported: int (e.g. 100 for testing) or None (full dataset per source)
MAX_PER_DATASET: Optional[int] = 100  # 4 datasets × 100 = 400 samples total

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
# Supported: any string (use [CALC: expr] for calculator)
INFERENCE_QUERY = "What is [CALC: 7*8]?"
# Supported: int
MAX_NEW_TOKENS = 128
# Supported: True | False
LOAD_IN_4BIT_INFERENCE = False

# -----------------------------------------------------------------------------
# RAG test
# -----------------------------------------------------------------------------
# Supported: True (real eval) | False (stub)
USE_SAFE_EVAL_RAG_TEST = True
