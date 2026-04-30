# Figure Helpers (Baseline Models)

These scripts generate Chapter 5 baseline figures from existing artifacts only (no retraining).

## Prerequisites
- Run commands from repo root (the directory containing `pyproject.toml`)
- Python 3
- `matplotlib` installed in your environment

## Notes
- Scripts use only baseline experiment outputs under `outputs/experiments/baseline/...`.
- In this repo history, some `*_deg` fields are legacy-named and still stored in radians; scripts convert to degrees where needed.

## Scripts
- `plot_baseline_loss_curves.py`
  - Purpose: single figure with training + validation loss curves and stopping markers for all 4 baselines.
  - Usage:
    - `python3 tools/figures/plot_baseline_loss_curves.py`
    - `python3 tools/figures/plot_baseline_loss_curves.py --save outputs/figures/ch4/baseline_loss_curves.png --no-show`

- `plot_baseline_loss_curves_split.py`
  - Purpose: two separate figures (training loss and validation loss) for all 4 baselines.
  - Usage:
    - `python3 tools/figures/plot_baseline_loss_curves_split.py`
    - `python3 tools/figures/plot_baseline_loss_curves_split.py --no-show`
  - Default outputs:
    - `outputs/figures/ch4/baseline_training_loss_curves.pdf`
    - `outputs/figures/ch4/baseline_validation_loss_curves.pdf`

- `plot_baseline_seeded_loss_curves_split.py`
  - Purpose: two separate figures (training loss and validation loss) for seeded SLM-LSTM runs, including the original `Seed 42` baseline by default.
  - Usage:
    - `python3 tools/figures/plot_baseline_seeded_loss_curves_split.py`
    - `python3 tools/figures/plot_baseline_seeded_loss_curves_split.py --no-show`
    - `python3 tools/figures/plot_baseline_seeded_loss_curves_split.py --exclude-seeds 97 --no-show`
  - Default outputs:
    - `outputs/figures/ch4/baseline_seeded_training_loss_curves.pdf`
    - `outputs/figures/ch4/baseline_seeded_validation_loss_curves.pdf`

- `plot_baseline_eval_timeseries_stacked.py`
  - Purpose: stacked representative eval time-series comparison (ground truth + baseline predictions).
  - Usage:
    - `python3 tools/figures/plot_baseline_eval_timeseries_stacked.py --no-show`
    - `python3 tools/figures/plot_baseline_eval_timeseries_stacked.py --groups slm_slu --output outputs/figures/ch4/my_eval_stack.pdf --no-show`

- `plot_baseline_eval_timeseries_stacked_layer_depth.py`
  - Purpose: stacked representative eval comparison by layer depth (`SLU+TLU` and `SLM+TLM`).
  - Usage:
    - `python3 tools/figures/plot_baseline_eval_timeseries_stacked_layer_depth.py --no-show`
    - `python3 tools/figures/plot_baseline_eval_timeseries_stacked_layer_depth.py --groups multivariate_depth --no-show`

- `plot_baseline_error_distribution_all_datasets.py`
  - Purpose: per-model histogram of prediction error distribution.
  - Usage:
    - `python3 tools/figures/plot_baseline_error_distribution_all_datasets.py --no-show`
    - `python3 tools/figures/plot_baseline_error_distribution_all_datasets.py --split-roles eval,unseen --bins 120 --no-show`
  - Default output: `outputs/figures/ch4/baseline_error_distribution_all_datasets.pdf`

- `plot_slm_worst_run_timeseries.py`
  - Purpose: plot prediction vs ground truth for the worst-RMSE SLM-LSTM run.
  - Usage:
    - `python3 tools/figures/plot_slm_worst_run_timeseries.py --no-show`
    - `python3 tools/figures/plot_slm_worst_run_timeseries.py --split-roles eval,unseen --no-show`
  - Default output: `outputs/figures/ch4/slm_lstm_representative_poor_performance_run.pdf`
