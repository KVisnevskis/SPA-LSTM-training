# SPA-LSTM Training

Training and evaluation pipeline for thesis-aligned stateful LSTM baselines that estimate SPA bending angle (`phi`) from pressure and IMU inputs.

Part of the codebase for:

> Krisjanis Visnevskis, *Data-Driven Bending Angle Estimation for Soft Pneumatic Actuators: Dataset, Methods, and Comparative Evaluation*, PhD Thesis, University of Aberdeen, 2026.
> Code: https://github.com/KVisnevskis/SPA-LSTM-training

Companion repositories:
[SPA-data-pre-processing](https://github.com/KVisnevskis/SPA-data-pre-processing) ·
[SPA-basic-estimators](https://github.com/KVisnevskis/SPA-basic-estimators) ·
[SPA-visualizer](https://github.com/KVisnevskis/SPA-visualizer) ·
[SPA-prediction-viewer](https://github.com/KVisnevskis/SPA-prediction-viewer)

Datasets:
[Raw sensor recordings](https://doi.org/10.5281/zenodo.18697336) ·
[Preprocessed HDF5 (direct download)](https://doi.org/10.5281/zenodo.19666505)

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev,train,viz]
pytest -q
```

## Data Prerequisite

The pipeline requires a preprocessed HDF5 dataset that is **not** included in this repository.

**Quick path — download preprocessed file directly:**

```bash
# Download preprocessed_all_trials.h5 from Zenodo (223 MB)
# https://doi.org/10.5281/zenodo.19666505
# Then place it and create the symlink the configs expect:
mkdir -p data outputs
mv preprocessed_all_trials.h5 data/
ln -s ../data/preprocessed_all_trials.h5 outputs/preprocessed_all_trials.h5
```

**Alternative — reproduce from raw data:**

1. Download raw sensor recordings from Zenodo: https://doi.org/10.5281/zenodo.18697336
2. Run the preprocessing pipeline in [SPA-data-pre-processing](https://github.com/KVisnevskis/SPA-data-pre-processing).
3. Place the resulting file at `data/preprocessed_all_trials.h5` and create the symlink above.

Use `scripts/inspect_hdf5.py` to verify the HDF5 structure (`/runs/<run_key>` tables) and `scripts/check_config_data.py` to confirm that a given config's run keys are present.

## Reproduce Main Results

After completing the Data Prerequisite step above, run the following to reproduce the four thesis baseline LSTM variants and their evaluation outputs:

```bash
# Train all 4 baseline variants (slm, slu, tlm, tlu)
.venv/bin/python scripts/train_all.py --config-dir configs/experiments/baseline

# Evaluate on the dedicated held-out split
.venv/bin/python scripts/evaluate_all.py --config-dir configs/experiments/baseline --scope eval

# Evaluate on the full dataset (all runs)
.venv/bin/python scripts/evaluate_all.py --config-dir configs/experiments/baseline --scope all

# Export error-analysis bundle used for thesis figures and tables
.venv/bin/python scripts/export_error_analysis_bundle.py --bundle-dir outputs/error_analysis_bundle
```

Training the four baseline variants takes approximately 25 hours total on a single GPU (see `docs/lstm_experiment_internal_note.md` for per-model timing). For a quick sanity check without committing to full training, use `scripts/run_smoke_e2e.py`.

For the MLP HPO sweep, width ablation, seed replication, and additional-train-set studies, see the full command sequences in `docs/lstm_experiment_internal_note.md` and `docs/mlp_experiment_internal_note.md`.

## Training Workflow (Current Implementation)

Entry points:

- Single config: `scripts/train.py`
- Batch directory: `scripts/train_all.py`

Example:

```bash
.venv/bin/python scripts/train.py \
  --config configs/experiments/baseline/baseline_slm_lstm.yaml
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
  --config configs/experiments/baseline/baseline_slm_lstm.yaml \
  --scope eval
```

Evaluate best model on all runs in HDF5:

```bash
.venv/bin/python scripts/evaluate_best.py \
  --config configs/experiments/baseline/baseline_slm_lstm.yaml \
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
- `outputs/experiments/thesis_slm_lstm/` is a historical artifact from early development; it has no corresponding source config YAML. The canonical equivalent is `configs/experiments/baseline/baseline_slm_lstm.yaml`.

## Environment

Developed and tested on:

- Python 3.12.3, TensorFlow 2.18.1, NumPy 2.0.2
- Linux (WSL2)
- NVIDIA GPU (required for reasonable training time; the baseline 512-unit LSTM takes ~6 hours on GPU)

Install with `pip install -e .[dev,train,viz]`. The TF version constraint (`<2.19`) in `pyproject.toml` is load-bearing.

## Design Reference

- `docs/specs/lstm_training_spec.md`
- `docs/lstm_experiment_internal_note.md` — per-experiment results, reproducibility commands, and seed/width sensitivity notes
- `docs/mlp_experiment_internal_note.md` — MLP HPO sweep results and reproducibility commands
