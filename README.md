# LLM Project - G1 Group 9

This repository contains our experiments on math reasoning with `Qwen/Qwen2-1.5B` and the GSM8K dataset. The repo contains three main tracks: a baseline benchmark, LoRA fine-tuning, and our reinforcement learning experiments, plus the saved outputs from our experiments. 

## Quick Start

```bash
./setup.sh
```

Or install manually:

```bash
pip install -r requirements.txt
```

## Repository Structure

### Top level

- `bm_final.ipynb`: baseline benchmark notebook for the base Qwen model.
- `finetuned_final.ipynb`: loads a saved LoRA adapter and benchmarks the fine-tuned model.
- `rl_final.ipynb`: reward-based experiment notebook and evaluation export.
- `llm_as_a_judge.ipynb`: samples examples from result files and writes smaller judge-review JSON files.
- `README_finetune.md`: detailed write-up for our fine-tuning setup and results.
- `requirements.txt`: Python dependencies.
- `setup.sh`: creates `.venv`, installs dependencies, and register Jupyter kernel.

### Result files

- `bm_results.json`: baseline benchmark outputs.
- `ft_results.json`: fine-tuned model benchmark outputs.
- `rl_results.json`: reward-based experiment outputs.
- `bm_llm.json`: small random sample from baseline results for LLM as a judge/manual review.
- `ft_llm.json`: small random sample from fine-tuned results for LLM as a judge/manual review.
- `rl_llm.json`: small random sample from reward-based results for LLM as a judge/manual review.

### Core package

- `qwen_math_flow/__init__.py`: package exports.
- `qwen_math_flow/download_model.py`: downloads/loads the Qwen model and tokenizer, with optional quantization.
- `qwen_math_flow/load_dataset.py`: loads GSM8K, formats prompts/chat data, and tokenizes datasets.
- `qwen_math_flow/lora_finetune.py`: LoRA setup, Trainer configuration, and early stopping callback.
- `qwen_math_flow/hyperparameters.py`: central config values for training and inference.

## External calculator tool-calling experimentation
- `qwen_math_flow/rag_calculator.py`: calculator-augmented generation layer.
- `qwen_math_flow/external_calculator.py`: calculator interfaces and a safe local evaluator.
- `qwen_math_flow/test_rag_calculator.py`: simple smoke test for the calculator flow.

### Artifact folders
- `output/fine_tune`: final training outputs, checkpoints, summaries from 5 sweep runs of LoRA fine tuning
- `fine_tune/`: saved LoRA run artifacts from previous sweep runs, deprecated. 
- `archive/`: older notebooks and previous experiment versions kept for future reference.

Inside each fine-tune run folder, you will find the same artifact pattern:

- `adapter_model.safetensors`: trained LoRA weights.
- `adapter_config.json`: adapter configuration.
- `tokenizer.json` and `tokenizer_config.json`: tokenizer snapshot.
- `training_summary.json`: run-level summary.
- `trainer_log_history.json`: logged training metrics.
- `training_run.log`: raw training log.

## Suggested Reading Order

1. `bm_final.ipynb` : our baseline model evaluation workflow.
2. `finetune_final.ipynb`: our fine tuning experiments with diff configs, and the training logs.
3. `finetune_inference_final.ipynb`:  shows how we load and evaluate the saved LoRA adapter on the test dataset.
4. `rl_final.ipynb`: our reinforcement learning experiments.
5. `README_finetune.md`: if you want more detail on the fine-tuning setup.   
6. `qwen_math_flow/`: if you want the reusable Python code behind the notebooks.
