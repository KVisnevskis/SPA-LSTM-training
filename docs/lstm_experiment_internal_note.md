# LSTM Experiment Internal Note

## Scope

This note covers the config-backed LSTM experiments archived under `configs/experiments/*` and `outputs/experiments/*` for the `spa_lstm` implementation in `src/spa_lstm`.

Included experiment families:

- `configs/experiments/baseline/`: the 4 thesis-aligned baseline variants
- `configs/experiments/baseline_slm_seeded_replication/`: seed-only replications of the multivariate single-layer LSTM
- `configs/experiments/hpo/width_ablation/`: hidden-width sweep for the multivariate single-layer LSTM
- `configs/experiments/lstm_additional_train_sets/`: multivariate single-layer LSTM with one additional train run and one additional validation run

Excluded from the quantitative tables:

- `outputs/experiments/baseline_replication/baseline_slm_lstm_replication/`: same-seed rerun with matching training summary to the main baseline, but no archived evaluation summaries
- `outputs/experiments/smoke*`: smoke-test artifacts
- `outputs/experiments/thesis_*`: older/orphaned artifacts not backed by the current `configs/experiments/*` structure

## Implementation Summary

- Task: sequence regression of SPA bending angle `phi` from pressure and IMU inputs.
- Dataset: `outputs/preprocessed_all_trials.h5`.
- Scaling policy: strict `prescaled` mode only; training consumes already scaled HDF5 values directly.
- Sequence formulation: each run is treated as one ordered time stream. The trainer reshapes `[T, F]` and `[T, 1]` into `[T, 1, F]` and `[T, 1, 1]` via `as_sequence_batch(...)`.
- Training mode: stateful Keras LSTMs with `batch_size=1`, `shuffle=False`, and recurrent-state resets before every training pass and every validation pass.
- Train/validation organization: runs are paired by position, so each epoch consists of five or six `train_run -> val_run` pairs depending on the config. Epoch metrics are the mean across all run pairs.
- Early stopping: manual patience-based stopping on mean validation loss across the paired validation runs.
- Saved checkpoints: `latest.keras` every epoch, `best.keras` on validation-loss improvement, `final.keras` at the final epoch reached.
- Evaluation path: `best.keras` is evaluated either on the dedicated eval split (`eval_summary.json`) or on the full HDF5 dataset (`eval_summary_all_runs.json`).
- Metrics: MSE loss during training; RMSE and MAE tracked during training and evaluation.
- Model construction:
  - `slm_lstm` and `slu_lstm`: one LSTM layer, default width 512 unless `hidden_units` overrides it
  - `tlm_lstm` and `tlu_lstm`: two stacked 256-unit LSTM layers
  - all LSTM layers use `tanh`, `return_sequences=True`, and a per-timestep Dense(1) output head
- Feature sets:
  - multivariate variants use `pressure, acc_x, acc_y, acc_z`
  - univariate variants use `pressure` only
- Runtime environment recorded in the baseline manifests: Python 3.12.3, TensorFlow 2.18.1, NumPy 2.0.2, Linux WSL2, GPU `/physical_device:GPU:0`.

## Timing Note

Unlike the MLP artifacts, the LSTM trainer logs sampled wall-clock resource usage to `resource_usage.csv`. For the 4 baseline runs, these traces are complete enough to use for approximate training duration and peak GPU-memory discussion.

For some resumed seed-replication runs, the resource traces are clearly partial and should not be treated as definitive wall-clock totals. Those secondary tables therefore emphasize convergence and error metrics rather than elapsed time.

## Fixed Baseline Split

The 4 baseline variants and the later width/seed studies all use the same core split:

- Train: `Freehand_tt_1` (12,225), `Freehand_static_03V_1` (2,174), `Freehand_static_09V_1` (2,249), `Freehand_sin_1` (2,274), `run_0roll_0pitch_tt_1` (12,433). Total: 31,355 rows across 5 runs.
- Validation: `Freehand_tt_2` (12,122), `Freehand_static_03V_2` (2,377), `Freehand_static_09V_2` (2,289), `Freehand_sin_2` (2,463), `run_0roll_0pitch_tt_2` (12,428). Total: 31,679 rows across 5 runs.
- Held-out / dedicated eval: `run_0roll_90pitch_tt_1` (12,414), `Freehand_tt_3` (12,010), `Freehand_static_03V_3` (2,215), `Freehand_static_06V_3` (2,340), `Freehand_static_09V_3` (2,146), `Freehand_sin_3` (2,331). Total: 33,456 rows across 6 runs.

The additional-train-set study changes this to 6 training runs and 6 validation runs by adding:

- Train: `run_180roll_45pitch_tt_1`
- Validation: `run_135roll_45pitch_tt_1`

This increases the archived row counts to 43,766 train rows and 44,109 validation rows.

## Core Baseline Variants

Training-history quantities (`best_val_loss`, `best_val_rmse`) come from `history.csv` and therefore remain in the scaled target space. Held-out RMSE/MAE values below are inverse-scaled physical-angle errors in radians, with degree equivalents added for readability.

| Model | Inputs | LSTM stack | Params | Epochs | Best epoch | Approx. train time | Peak GPU mem | Best val loss | Held-out RMSE | Held-out MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_slm_lstm | pressure + accel | 1 x 512 | 1,059,329 | 79 | 59 | 5.72 h | 7.68 GB | 0.002985 | 0.1846 rad / 10.58 deg | 0.0823 rad / 4.72 deg |
| baseline_slu_lstm | pressure only | 1 x 512 | 1,053,185 | 53 | 33 | 3.75 h | 7.50 GB | 0.038534 | 1.0224 rad / 58.58 deg | 0.9747 rad / 55.84 deg |
| baseline_tlm_lstm | pressure + accel | 2 x 256 | 792,833 | 79 | 59 | 7.88 h | 7.65 GB | 0.007263 | 0.5076 rad / 29.08 deg | 0.4850 rad / 27.79 deg |
| baseline_tlu_lstm | pressure only | 2 x 256 | 789,761 | 98 | 78 | 7.53 h | 6.37 GB | 0.027190 | 0.9865 rad / 56.52 deg | 0.9381 rad / 53.75 deg |

Key takeaways from the baseline family:

- `baseline_slm_lstm` is the strongest of the 4 thesis baseline variants by a large margin on the dedicated held-out split.
- The pressure-only variants (`slu`, `tlu`) are substantially worse than the multivariate variants.
- The two-layer multivariate model (`tlm`) underperforms the one-layer multivariate baseline despite lower parameter count.
- For `baseline_slm_lstm`, held-out dynamic runs are markedly harder than held-out static runs:
  - dynamic held-out weighted RMSE: `0.1999 rad`
  - static held-out weighted RMSE: `0.1029 rad`
- The broader all-run evaluation is much harsher than the dedicated held-out split. For `baseline_slm_lstm`, the archived `unseen` weighted RMSE across 46 not-in-split runs is `2.9552 rad`, far above the dedicated held-out RMSE of `0.1846 rad`.

## Held-Out Run RMSE Matrix For The 4 Baseline Variants

Values below are RMSE in radians on the 6 dedicated eval runs from `eval_metrics.json`.

| Model | run_0roll_90pitch_tt_1 | Freehand_tt_3 | Freehand_static_03V_3 | Freehand_static_06V_3 | Freehand_static_09V_3 | Freehand_sin_3 | Weighted held-out RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_slm_lstm | 0.0523 | 0.2876 | 0.0796 | 0.1138 | 0.1114 | 0.1346 | 0.1846 |
| baseline_slu_lstm | 1.0241 | 1.0853 | 0.8171 | 0.8558 | 0.9697 | 1.0550 | 1.0224 |
| baseline_tlm_lstm | 0.5195 | 0.5083 | 0.5053 | 0.4568 | 0.4991 | 0.4971 | 0.5076 |
| baseline_tlu_lstm | 0.9561 | 1.0133 | 0.8860 | 1.0247 | 0.9720 | 1.0676 | 0.9865 |

## Archived `slm_lstm` Seed Sensitivity

Only `eval_summary_all_runs.json` is archived for these runs, so the “held-out” column below is taken from `by_split_role.eval.weighted_rmse`, which still corresponds to the same 6 dedicated eval runs.

| Model | Seed | Epochs | Best epoch | Best val loss | Held-out RMSE | Validation RMSE | Unseen RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_slm_lstm_seed_7 | 7 | 56 | 36 | 0.004031 | 0.2105 | 0.2271 | 0.5323 |
| baseline_slm_lstm_seed_17 | 17 | 64 | 43 | 0.003779 | 0.4839 | 0.1969 | 3.3347 |
| baseline_slm_lstm_seed_29 | 29 | 34 | 13 | 0.003590 | 0.4890 | 0.2848 | 1.4417 |
| baseline_slm_lstm_seed_53 | 53 | 67 | 47 | 0.003353 | 3.2170 | 0.1675 | 12.1002 |
| baseline_slm_lstm_seed_97 | 97 | 61 | 41 | 0.002762 | 0.2001 | 0.1854 | 25.0679 |

Observations:

- The `slm_lstm` family is highly seed-sensitive in the archived runs.
- Lower validation loss does not reliably imply safer out-of-split behavior.
- Seed 97 achieves the best archived held-out RMSE within the seed sweep (`0.2001 rad`) but catastrophically poor unseen RMSE (`25.0679 rad`).
- Seed 7 is not the best held-out run in this sweep, but it is the most stable of the archived seed runs when judged jointly on held-out and unseen performance.

## Archived `slm_lstm` Width Ablation

This study keeps the baseline split fixed and changes only the hidden width of a single-layer multivariate LSTM, all at seed 7. It therefore probes width sensitivity rather than reproducing the seed-42 thesis baseline exactly.

| Model | Hidden units | Params | Epochs | Best epoch | Best val loss | Held-out RMSE | Validation RMSE | Unseen RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| slm_width_ablation__baseline__u016__seed007 | 16 | 1,361 | 67 | 47 | 0.002961 | 0.1640 | 0.1720 | 0.4724 |
| slm_width_ablation__baseline__u032__seed007 | 32 | 4,769 | 43 | 23 | 0.004416 | 0.3368 | 0.3111 | 0.4750 |
| slm_width_ablation__baseline__u064__seed007 | 64 | 17,729 | 50 | 30 | 0.005437 | 0.3656 | 0.3498 | 0.5380 |
| slm_width_ablation__baseline__u128__seed007 | 128 | 68,225 | 51 | 31 | 0.002661 | 0.1964 | 0.1756 | 0.5267 |
| slm_width_ablation__baseline__u192__seed007 | 192 | 151,489 | 46 | 26 | 0.003657 | 0.2299 | 0.2351 | 0.3693 |
| slm_width_ablation__baseline__u256__seed007 | 256 | 267,521 | 59 | 39 | 0.003901 | 0.8767 | 0.1893 | 4.2576 |
| slm_width_ablation__baseline__u384__seed007 | 384 | 597,889 | 58 | 38 | 0.003946 | 0.1691 | 0.1895 | 0.3770 |

Observations:

- The width sweep is not monotonic.
- The strongest archived held-out results in this sweep are at 16 units (`0.1640 rad`) and 384 units (`0.1691 rad`), both better than the seed-7 512-unit run.
- Width 256 is a clear instability case: respectable validation RMSE but very poor held-out and unseen behavior.
- Width 384 provides the best tradeoff in this archived sweep when considering both held-out and unseen RMSE together.

## Archived Additional-Train-Set Study

This study keeps the single-layer multivariate LSTM architecture and adds one extra training run plus one extra validation run. As above, the reported held-out numbers come from `eval_summary_all_runs.json -> by_split_role.eval`.

| Model | Seed | Train runs | Val runs | Epochs | Best epoch | Best val loss | Held-out RMSE | Validation RMSE | Unseen RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| slm_lstm_additional_train_set__seed007 | 7 | 6 | 6 | 47 | 27 | 0.004681 | 0.2010 | 0.2345 | 0.6860 |
| slm_lstm_additional_train_set__seed017 | 17 | 6 | 6 | 30 | 10 | 0.009864 | 0.3491 | 0.3708 | 0.4864 |
| slm_lstm_additional_train_set__seed027 | 27 | 6 | 6 | 39 | 19 | 0.005093 | 0.1205 | 0.1875 | 0.7983 |

Observations:

- `slm_lstm_additional_train_set__seed027` achieves the best archived dedicated held-out RMSE of any config-backed LSTM artifact in this repository: `0.1205 rad`.
- That improvement does not translate into better broad unseen performance; its unseen RMSE is `0.7983 rad`, worse than the baseline seed-7 run and worse than the best width-ablation cases.
- The additional-train-set study therefore looks promising for the narrow dedicated held-out split, but not decisively better for broader generalization across the full archived run collection.

## Reproducibility Commands

Canonical LSTM entry points in this repository are:

```bash
.venv/bin/python scripts/train.py --config configs/experiments/baseline/baseline_slm_lstm.yaml
.venv/bin/python scripts/evaluate_best.py --config configs/experiments/baseline/baseline_slm_lstm.yaml --scope eval
.venv/bin/python scripts/evaluate_best.py --config configs/experiments/baseline/baseline_slm_lstm.yaml --scope all

.venv/bin/python scripts/train_all.py --config-dir configs/experiments/baseline
.venv/bin/python scripts/evaluate_all.py --config-dir configs/experiments/baseline --scope eval
.venv/bin/python scripts/evaluate_all.py --config-dir configs/experiments/baseline --scope all
```

For packaged comparative-analysis outputs already built from these artifacts, see:

- `outputs/error_analysis_bundle/`: baseline 4 LSTM variants
- `outputs/error_analysis_bundle_NN/`: baseline variants plus seeded replications, width ablation, and additional-train-set studies
