# MLP Experiment Internal Note

## Scope

This note covers the archived MLP experiments under `configs/experiments/mlp`, `outputs/experiments/mlp`, and `src/mlp`. It excludes the LSTM implementation.

The repository contains 8 trained MLP artifact directories under `outputs/experiments/mlp/hpo/`, corresponding to the YAML files in `configs/experiments/mlp/hpo/`. The top-level `configs/experiments/mlp/baseline_mlp.yaml` captures the canonical split and default hyperparameters, but there is no matching archived output directory for `outputs/experiments/mlp/baseline_mlp`, so the historical results below come from the HPO sweep artifacts.

## Implementation Summary

- Task: row-wise regression from `pressure`, `acc_x`, `acc_y`, and `acc_z` to `phi`.
- Data source: `outputs/preprocessed_all_trials.h5`.
- Data handling: the MLP path concatenates rows from the configured runs into one training matrix and one validation matrix; there is no sequence windowing or recurrent state.
- Split policy: the MLP configs reuse the same fixed run memberships as the baseline LSTM split.
- Scaling: training uses prescaled HDF5 inputs/targets. Evaluation inverse-transforms predictions back to physical `phi` units. The saved prediction CSVs keep legacy `phi_true_deg` and `phi_pred_deg` column names even though the stored values are radians.
- Architecture: the model in `src/mlp/model.py` is a feed-forward Keras network with ReLU hidden layers, optional dropout after each hidden layer, and a linear scalar output head.
- Archived sweep variables: hidden-layer layout and learning rate only. All archived runs use `dropout=0.0`, `batch_size=128`, `patience=15`, `seed=42`, `epochs=200`, and `activation=relu`.
- Optimizer and objectives: Adam optimizer, MSE loss, RMSE and MAE tracked during training.
- Training logic: one epoch at a time via `model.fit(...)`, shuffled training rows, manual early stopping on validation loss, `final.keras` saved from the terminal epoch, and `best.keras` saved from the best validation-loss epoch.
- Evaluation logic: `best.keras` is evaluated on both the dedicated eval split and then on all runs in the HDF5 dataset.
- Recorded runtime environment in every `run_manifest.json`: Python 3.12.3, TensorFlow 2.18.1, NumPy 2.0.2, Linux WSL2, GPU `/physical_device:GPU:0`.

## Fixed Dataset Split

- Train: `Freehand_tt_1` (12,225), `Freehand_static_03V_1` (2,174), `Freehand_static_09V_1` (2,249), `Freehand_sin_1` (2,274), `run_0roll_0pitch_tt_1` (12,433). Total: 31,355 rows across 5 runs.
- Validation: `Freehand_tt_2` (12,122), `Freehand_static_03V_2` (2,377), `Freehand_static_09V_2` (2,289), `Freehand_sin_2` (2,463), `run_0roll_0pitch_tt_2` (12,428). Total: 31,679 rows across 5 runs.
- Held-out / dedicated eval: `run_0roll_90pitch_tt_1` (12,414), `Freehand_tt_3` (12,010), `Freehand_static_03V_3` (2,215), `Freehand_static_06V_3` (2,340), `Freehand_static_09V_3` (2,146), `Freehand_sin_3` (2,331). Total: 33,456 rows across 6 runs.
- All-run evaluation was later executed across 62 runs and 617,247 rows. Of these, 46 runs and 520,757 rows are outside the configured train/val/eval split and appear as `unseen` in the exported bundle.

## Sweep Coverage And Convergence

- Sweep coverage: `[32]`, `[64]`, `[128]`, `[64,32]`, `[64,64]`, and `[128,64]` hidden-layer layouts, with learning rates `1e-3` and `3e-4` where archived.
- Parameter counts below are reconstructed from dense-layer shapes with biases, assuming a `4 -> hidden layers -> 1` network.
- Every archived run stopped early; none reached the 200-epoch cap.
- `epochs_completed` ranges from 16 to 31.
- `best_epoch` ranges from 1 to 16, so useful validation performance was reached very early in all archived runs.
- The trainer did not log wall-clock duration directly. The times below are inferred from the modification timestamps of `scaler_bounds.json` and `history.csv`, so they should be treated as approximate run durations rather than precise timers.
- The lowest archived best validation loss is `baseline_mlp__u064__lr3e4` at `0.002866879` (best epoch 16).
- The lowest archived held-out weighted RMSE is `baseline_mlp__u064_u064__lr1e3` at `0.1019 rad` (`5.84 deg`).
- No explicit model-selection manifest was found in the repository, so the archive preserves both a validation-loss ranking and a held-out ranking but not an authoritative final-choice marker.

Training-history metrics (`best_val_loss`, `best_val_rmse`, `best_val_mae`, `final_val_*`) are in the scaled target space from `history.csv`. Held-out and unseen RMSE/MAE values below are inverse-scaled physical-angle errors in radians, with degree equivalents shown for the held-out aggregate.

## Model-Level Summary

| Model | Hidden layers | Params | LR | Epochs | Best epoch | Inferred train time (s) | Best val loss | Held-out RMSE | Held-out MAE | Unseen RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_mlp__u064_u064__lr1e3 | 64, 64 | 4,545 | 0.001 | 16 | 1 | 20.0 | 0.002885 | 0.1019 rad / 5.84 deg | 0.0704 rad / 4.04 deg | 0.3368 |
| baseline_mlp__u064_u032__lr1e3 | 64, 32 | 2,433 | 0.001 | 20 | 5 | 22.2 | 0.002901 | 0.1048 rad / 6.00 deg | 0.0737 rad / 4.22 deg | 0.3533 |
| baseline_mlp__u128__lr1e3 | 128 | 769 | 0.001 | 20 | 5 | 20.9 | 0.002900 | 0.1059 rad / 6.07 deg | 0.0717 rad / 4.11 deg | 0.3093 |
| baseline_mlp__u064__lr3e4 | 64 | 385 | 0.0003 | 31 | 16 | 30.3 | 0.002867 | 0.1062 rad / 6.08 deg | 0.0698 rad / 4.00 deg | 0.3393 |
| baseline_mlp__u128_u064__lr1e3 | 128, 64 | 8,961 | 0.001 | 21 | 6 | 25.3 | 0.002936 | 0.1070 rad / 6.13 deg | 0.0765 rad / 4.38 deg | 0.3131 |
| baseline_mlp__u064__lr1e3 | 64 | 385 | 0.001 | 25 | 10 | 24.5 | 0.002935 | 0.1073 rad / 6.15 deg | 0.0708 rad / 4.06 deg | 0.3387 |
| baseline_mlp__u064_u032__lr3e4 | 64, 32 | 2,433 | 0.0003 | 23 | 8 | 23.7 | 0.002934 | 0.1078 rad / 6.18 deg | 0.0760 rad / 4.35 deg | 0.4182 |
| baseline_mlp__u032__lr1e3 | 32 | 193 | 0.001 | 22 | 7 | 22.9 | 0.002923 | 0.1083 rad / 6.20 deg | 0.0724 rad / 4.15 deg | 0.3107 |

## Held-Out Run RMSE Matrix

Values below are RMSE in radians for the 6 dedicated eval runs.

| Model | run_0roll_90pitch_tt_1 | Freehand_tt_3 | Freehand_static_03V_3 | Freehand_static_06V_3 | Freehand_static_09V_3 | Freehand_sin_3 | Weighted held-out RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_mlp__u064_u064__lr1e3 | 0.0560 | 0.1321 | 0.0729 | 0.1152 | 0.1040 | 0.1184 | 0.1019 |
| baseline_mlp__u064_u032__lr1e3 | 0.0632 | 0.1303 | 0.0752 | 0.1429 | 0.1021 | 0.1162 | 0.1048 |
| baseline_mlp__u128__lr1e3 | 0.0602 | 0.1386 | 0.0697 | 0.1246 | 0.1007 | 0.1144 | 0.1059 |
| baseline_mlp__u064__lr3e4 | 0.0618 | 0.1410 | 0.0678 | 0.1057 | 0.1022 | 0.1173 | 0.1062 |
| baseline_mlp__u128_u064__lr1e3 | 0.0683 | 0.1343 | 0.0736 | 0.1385 | 0.0996 | 0.1158 | 0.1070 |
| baseline_mlp__u064__lr1e3 | 0.0631 | 0.1430 | 0.0683 | 0.1115 | 0.0987 | 0.1131 | 0.1073 |
| baseline_mlp__u064_u032__lr3e4 | 0.0657 | 0.1394 | 0.0668 | 0.1307 | 0.1022 | 0.1132 | 0.1078 |
| baseline_mlp__u032__lr1e3 | 0.0585 | 0.1429 | 0.0738 | 0.1219 | 0.1035 | 0.1221 | 0.1083 |

## Reproducibility Commands

The repository now contains a dedicated MLP bundle exporter at `scripts/export_mlp_error_analysis_bundle.py`. The relevant commands are:

```bash
.venv/bin/python scripts/train_mlp_all.py --config-dir configs/experiments/mlp/hpo
.venv/bin/python scripts/evaluate_mlp_all.py --config-dir configs/experiments/mlp/hpo --scope eval
.venv/bin/python scripts/evaluate_mlp_all.py --config-dir configs/experiments/mlp/hpo --scope all
.venv/bin/python scripts/export_mlp_error_analysis_bundle.py --bundle-dir outputs/error_analysis_bundle_MLP
```

The two main generated deliverables are:

- Internal note: `docs/mlp_experiment_internal_note.md`
- Comparative-analysis bundle: `outputs/error_analysis_bundle_MLP/`
