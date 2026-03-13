# Table Helpers (Baseline Models)

These scripts generate Chapter 5 baseline tables from saved artifacts only (no retraining).

## Prerequisites
- Run commands from repo root: `/home/visne/projects/SPA-LSTM/SPA-LSTM-training`
- Python 3

## Notes
- Scope is baseline model run directories under `outputs/experiments/baseline/...`.
- Angle metrics are reported in degrees.
- Legacy `*_deg` fields may contain radians in this repo history; scripts handle conversion.

## Quick Repro Workflow
1. RMSE grouped table:
   - `python3 tools/tables/generate_baseline_rmse_grouped_table.py`
2. MAE grouped table:
   - `python3 tools/tables/generate_baseline_mae_grouped_table.py`
3. P95 tables (two-step):
   - `python3 tools/tables/compute_baseline_p95_per_run.py`
   - `python3 tools/tables/generate_baseline_p95_grouped_table.py`
4. Extreme error count table:
   - `python3 tools/tables/generate_baseline_extreme_error_counts_table.py`
5. Compact held-out eval comparison table:
   - `python3 tools/tables/generate_baseline_eval_main_comparison_table.py`

## Script Purpose and Default Outputs
- `generate_baseline_rmse_grouped_table.py`
  - Per-run RMSE grouped by `train -> val -> eval -> unseen`.
  - Output: `tables/tab_ch5/baseline_rmse_per_run_grouped.tex`

- `generate_baseline_mae_grouped_table.py`
  - Per-run MAE grouped by `train -> val -> eval -> unseen`.
  - Output: `tables/tab_ch5/baseline_mae_per_run_grouped.tex`

- `compute_baseline_p95_per_run.py`
  - Computes per-run `P95(|e|)` from prediction traces.
  - Output: `tables/tab_ch5/baseline_p95_per_run_all_models.csv`

- `generate_baseline_p95_grouped_table.py`
  - Converts computed P95 CSV to grouped LaTeX table.
  - Output: `tables/tab_ch5/baseline_p95_per_run_grouped.tex`

- `generate_baseline_extreme_error_counts_table.py`
  - Counts samples exceeding an absolute error threshold (default `360 deg`) and reports percentages.
  - Default split roles: `eval,unseen`
  - Output: `tables/tab_ch5/baseline_extreme_error_counts_360deg.tex`

- `generate_baseline_eval_main_comparison_table.py`
  - Compact held-out eval table with mean/median RMSE, mean MAE, mean bias, mean `R^2`, mean per-run P95AE, worst-run RMSE, and best-on-run count.
  - Outputs:
    - `tables/tab_ch5/baseline_eval_main_comparison.csv`
    - `tables/tab_ch5/baseline_eval_main_comparison.tex`

## Useful Options
- Most scripts support `--repo-root` to point at another checkout.
- Grouped tables support `--decimals` and `--no-color`.
- Extreme-error and P95 scripts support `--split-roles` for filtering partitions.

