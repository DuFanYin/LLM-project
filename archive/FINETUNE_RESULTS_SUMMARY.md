# Fine-tune Results Summary

This document summarizes the sweep results from:
- `fine_tune/`
- `FineTune_Output.ipynb`

Run timestamp (from `fine_tune/sweep_index.json`): `2026-04-02T13:56:37Z`

## Experiment setup

- Dataset: `openai/gsm8k` (`train` split)
- Training samples: `7473`
- Sequence length: `512`
- Batch config: `per_device_train_batch_size=4`, `gradient_accumulation_steps=4`
- Quantization during training: `load_in_4bit=false`
- Logging: `logging_steps=1`, notebook progress enabled
- Sweep size: `5` parameter combinations

## Parameter combinations tested

1. `lr=1e-5, r=8, alpha=16, epochs=1`
2. `lr=2e-5, r=8, alpha=16, epochs=2`
3. `lr=2e-5, r=16, alpha=32, epochs=2`
4. `lr=3e-5, r=16, alpha=32, epochs=3`
5. `lr=3e-5, r=32, alpha=64, epochs=3`

## Final metrics by run

| Run | Params (lr, r, alpha, epochs) | final train_loss | runtime (s) | steps/s | step |
|---|---|---:|---:|---:|---:|
| `run_000_lr1e-5_r8_a16_ep1` | `1e-5, 8, 16, 1` | `1.519227` | `18.3090` | `25.561` | `20` |
| `run_001_lr2e-5_r8_a16_ep2` | `2e-5, 8, 16, 2` | `1.519245` | `17.8381` | `52.472` | `20` |
| `run_002_lr2e-5_r16_a32_ep2` | `2e-5, 16, 32, 2` | `1.513287` | `17.9746` | `52.073` | `20` |
| `run_003_lr3e-5_r16_a32_ep3` | `3e-5, 16, 32, 3` | `1.514121` | `17.9863` | `78.059` | `20` |
| `run_004_lr3e-5_r32_a64_ep3` | `3e-5, 32, 64, 3` | **`1.500700`** | `18.1094` | `77.529` | `20` |

## Best run (by train loss)

- Best directory: `fine_tune/run_004_lr3e-5_r32_a64_ep3`
- Best config: `learning_rate=3e-5`, `lora_r=32`, `lora_alpha=64`, `num_epochs=3`
- Best observed `train_loss`: `1.500700`

## Important observations

1. **All runs ended at step 20** and around `epoch=0.0428` despite different `num_epochs`.
   - `FineTune.ipynb` includes early-stop controls (`early_stop_window`, `early_stop_min_delta`), and each run summary records `early_stop_window=20`, matching the observed stop at step 20.
2. `training_summary.json` currently has `final_train_loss: null` in all runs.
   - The real final loss is available in each run's `trainer_log_history.json` (last record).
3. Relative trend from this sweep:
   - Increasing LoRA capacity (`r=32, alpha=64`) at `lr=3e-5` gave the best training loss among tested settings.
   - `run_000` and `run_001` are effectively tied and clearly behind the best run.

## Recommended next sweep

To validate whether the current winner is robust (not just short-run noise), try:

- Keep best base: `lr=3e-5, r=32, alpha=64`
- Add nearby configs:
  - `lr in {2e-5, 3e-5, 4e-5}`
  - `r in {16, 32, 48}`
  - `epochs in {3, 5}` (only useful if runs are not truncated at step 20)
- Include a held-out evaluation metric (e.g., exact match on GSM8K test subset), not only training loss.

## Artifact locations

- Sweep index: `fine_tune/sweep_index.json`
- Per-run summaries: `fine_tune/run_*/training_summary.json`
- Per-run detailed logs: `fine_tune/run_*/trainer_log_history.json`
- Notebook run output: `FineTune_Output.ipynb`
