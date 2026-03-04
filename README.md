# SPA-LSTM Training

Training and evaluation pipeline for thesis-aligned stateful LSTM baselines that estimate SPA bending angle (`phi`) from pressure and IMU inputs.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev,train,viz]
pytest -q
```

## Data Prerequisite

The repo expects a preprocessed HDF5 dataset (default: `outputs/preprocessed_all_trials.h5`) with run tables under `/runs/<run_key>`.

See:

- `context/lstm_baseline_handoff.md`
- `scripts/inspect_hdf5.py`
- `scripts/check_config_data.py`

## Training Workflow (Current Implementation)

Entry points:

- Single config: `scripts/train.py`
- Batch directory: `scripts/train_all.py`

Example:

```bash
.venv/bin/python scripts/train.py \
  --config configs/experiments/thesis_slm_lstm.yaml
```

Useful flags:

- `--resume` resumes from `latest.keras` + `resume_state.json`
- `--allow-overwrite` bypasses non-empty output-dir safety check
- `--verbose`, `--fit-verbose`, `--eval-verbose`, `--log-each-fit` override training logging controls

Training behavior:

1. Loads run-level train/val splits from config (`train_runs` and `val_runs` must be equal length).
2. Uses prescaled values directly from HDF5 (`scaling.mode: prescaled`).
3. Stateful training is run per pair with shape conversion to stream batches `[T,1,F]`.
4. Resets recurrent state before each train pass and validation pass.
5. Aggregates epoch metrics (`loss`, `rmse`, `mae`) and applies early stopping on mean validation loss.
6. Saves resume artifacts every epoch (`latest.keras`, `resume_state.json`, and `best.keras` on improvement).

Training artifacts are written to:

- `outputs/experiments/<run_name>/`

Primary outputs:

- `best.keras`, `final.keras`, `latest.keras`
- `history.csv`
- `training_summary.json`
- `run_manifest.json`
- `config_snapshot.json`
- `resume_state.json`
- `scaler_bounds.json`
- `resource_usage.csv` (CPU/RAM/GPU sampling, minimum 15s interval)

## Evaluation Workflow (Current Implementation)

Entry points:

- Evaluate explicit model: `scripts/evaluate.py`
- Evaluate best model for one config: `scripts/evaluate_best.py`
- Evaluate best models for all configs in a directory: `scripts/evaluate_all.py`

Evaluate best model on configured eval runs:

```bash
.venv/bin/python scripts/evaluate_best.py \
  --config configs/experiments/thesis_slm_lstm.yaml \
  --scope eval
```

Evaluate best model on all runs in HDF5:

```bash
.venv/bin/python scripts/evaluate_best.py \
  --config configs/experiments/thesis_slm_lstm.yaml \
  --scope all
```

Batch evaluation (all configs in a directory):

```bash
.venv/bin/python scripts/evaluate_all.py \
  --config-dir configs/experiments/baseline \
  --scope eval
```

Evaluation scopes:

- `eval`: only `data.eval_runs` from config
- `all`: all available `/runs/*` keys in dataset

Evaluation outputs:

- `eval`: `eval_metrics.json`, `eval_summary.json`, `predictions/*.csv`
- `all`: `eval_metrics_all_runs.json`, `eval_summary_all_runs.json`, `predictions_all_runs/*.csv`

Per-run metrics include split membership metadata:

- `split_role`: `train`, `val`, `eval`, `unseen` (or `overlap`)
- `is_train_run`, `is_val_run`, `is_eval_run`, `is_unseen_run`
- `motion_type`: `static` or `dynamic`

## Notes

- Current scaler policy is intentionally strict: only `prescaled` mode is supported.
- Metric/prediction field names retain legacy `*_deg` naming for compatibility with existing viewer tooling.

## Design Reference

- `docs/specs/lstm_training_spec.md`
