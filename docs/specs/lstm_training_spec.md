# SPA LSTM Training/Evaluation Specification (Implementation Snapshot)

Status: Active implementation reference.

This document describes how training and evaluation currently work in this repository. It is intentionally implementation-first and should be updated as behavior changes.

## 1. Purpose

Define the current, reproducible workflow for:

1. Training thesis-aligned stateful LSTM baselines for `phi`.
2. Evaluating trained models on configured eval runs and on all available runs.
3. Persisting artifacts needed for analysis and thesis reporting.

## 2. Data Contract

## 2.1 Input dataset

Expected HDF5 path is configured per experiment (`data.h5_path`), typically:

- `outputs/preprocessed_all_trials.h5`

## 2.2 HDF5 structure

Current loader expects run tables at:

- `/runs/<run_key>`

Metadata used by this repo:

- `/meta/scaler_parameters` (for min/max de-scaling bounds)

## 2.3 Required columns

For a given config, each selected run must contain:

1. All configured feature columns (`data.features`).
2. The configured target column (`data.target`, baseline: `phi`).

`Time` is optional but included in prediction CSV output if present.

## 3. Configuration Contract

Experiment YAML sections:

1. `data`
2. `model`
3. `training`
4. `runtime`

Relevant constraints:

1. `data.scaling.mode` must be `prescaled`.
2. `training.stateful=true` requires `training.batch_size=1`.
3. `data.train_runs` and `data.val_runs` must be non-empty and equal length.
4. Split overlap between train/val/eval is rejected in training preflight checks.

## 4. Model Variants

Supported variants:

1. `slm_lstm` (single-layer multivariate, 512 units)
2. `tlm_lstm` (two-layer multivariate, 256+256 units)
3. `slu_lstm` (single-layer univariate, 512 units)
4. `tlu_lstm` (two-layer univariate, 256+256 units)

Compile settings:

1. Optimizer: Adam (`model.learning_rate`)
2. Loss: MSE
3. Metrics: RMSE, MAE

## 5. Training Workflow

Primary entry points:

1. `scripts/train.py` (single config)
2. `scripts/train_all.py` (directory of configs)

## 5.1 Training loop semantics

For each paired `(train_run_i, val_run_i)`:

1. Convert features/target from `[T,F]` and `[T,1]` to stream batches `[T,1,F]` and `[T,1,1]`.
2. Reset recurrent states.
3. `model.fit(..., epochs=1, batch_size=1, shuffle=False)`.
4. Reset recurrent states.
5. `model.evaluate(..., batch_size=1, return_dict=True)`.

Epoch-level metrics are means across all run pairs:

1. `train_loss_mean`, `train_rmse_mean`, `train_mae_mean`
2. `val_loss_mean`, `val_rmse_mean`, `val_mae_mean`

Early stopping:

1. Monitor: `val_loss_mean`
2. Trigger: `patience` consecutive non-improving epochs

## 5.2 Resume/checkpoint behavior

During training, each epoch writes:

1. `latest.keras` (latest model state)
2. `resume_state.json` (next epoch + best metadata + epoch history)
3. `best.keras` when the current epoch is best so far

At run end:

1. `final.keras` is saved from latest state.
2. Best weights are restored and `best.keras` is written again.

Resume path:

1. `--resume` loads `latest.keras` + `resume_state.json` if present.
2. Best weights are restored from `best.keras` when available.

## 5.3 Training artifacts

Written under:

- `<runtime.output_dir>/<runtime.run_name>/`

Artifacts:

1. `best.keras`
2. `final.keras`
3. `latest.keras`
4. `history.csv`
5. `training_summary.json`
6. `run_manifest.json`
7. `config_snapshot.json`
8. `resume_state.json`
9. `scaler_bounds.json`
10. `resource_usage.csv` (if monitor starts successfully)
11. `training_error.log` (on failure)

## 5.4 Resource monitoring

Resource monitor samples on a background thread and writes:

1. CPU percent
2. RAM percent
3. GPU utilization and memory (via `nvidia-smi` when available)

Sampling interval has a hard minimum of 15 seconds.

## 6. Evaluation Workflow

Primary entry points:

1. `scripts/evaluate.py` (explicit model path)
2. `scripts/evaluate_best.py` (best model for one config)
3. `scripts/evaluate_all.py` (best models for all configs in a directory)

## 6.1 Evaluation scopes

`evaluate_model(..., scope=...)` supports:

1. `scope="eval"`: evaluate only config `data.eval_runs`
2. `scope="all"`: evaluate all run keys discovered under `/runs/*`

## 6.2 Inference behavior

Per run:

1. Features/target are reshaped to stream batches `[T,1,F]` and `[T,1,1]`.
2. Recurrent states are reset at run start.
3. Predictions are generated with `batch_size=1`.
4. Target/predictions are de-scaled using bounds from:
   - `<run_dir>/scaler_bounds.json` if present, else
   - `/meta/scaler_parameters` from HDF5.

## 6.3 Per-run metadata in metrics

Each run record includes:

1. `split_role`: `train`, `val`, `eval`, `unseen`, or `overlap`
2. `is_train_run`, `is_val_run`, `is_eval_run`, `is_unseen_run`
3. `motion_type`: `static`/`dynamic`
4. `n_samples`
5. `rmse`, `mae`
6. Legacy-compatible aliases: `rmse_deg`, `mae_deg`
7. `prediction_csv` path

## 6.4 Evaluation outputs

For `scope="eval"`:

1. `eval_metrics.json`
2. `eval_summary.json`
3. `predictions/<run_key>.csv`

For `scope="all"`:

1. `eval_metrics_all_runs.json`
2. `eval_summary_all_runs.json`
3. `predictions_all_runs/<run_key>.csv`

Summary files include:

1. Overall weighted RMSE/MAE
2. Aggregation by `split_role`
3. Aggregation by `motion_type`

## 7. CLI Reference

## 7.1 Train one config

```bash
.venv/bin/python scripts/train.py --config <config.yaml> [--resume]
```

Optional training overrides:

- `--verbose`
- `--fit-verbose`
- `--eval-verbose`
- `--log-each-fit` / `--no-log-each-fit`

## 7.2 Train all configs in directory

```bash
.venv/bin/python scripts/train_all.py --config-dir <dir> [--resume]
```

## 7.3 Evaluate one model

```bash
.venv/bin/python scripts/evaluate.py \
  --config <config.yaml> \
  --model <model.keras> \
  [--scope eval|all]
```

## 7.4 Evaluate best model for one config

```bash
.venv/bin/python scripts/evaluate_best.py \
  --config <config.yaml> \
  [--scope eval|all]
```

## 7.5 Evaluate all best models in a directory

```bash
.venv/bin/python scripts/evaluate_all.py \
  --config-dir <dir> \
  [--scope eval|all]
```

## 8. Known Current Limitations

1. Only `prescaled` scaling mode is supported.
2. Metric/prediction field names retain legacy `*_deg` suffixes for compatibility.
3. Full `scope="all"` evaluation can be long-running on large datasets.
