# SPA LSTM Training Specification

Status: Draft baseline specification for LSTM based on legacy implementation

## 1. Purpose
This document defines the technical specification for a rigorous, reproducible LSTM training pipeline that estimates SPA bending angle (`phi`) from onboard pressure and IMU data.

It is the core implementation reference for this repository.

## 2. Sources and Precedence
Implementation choices in this spec are derived from:

1. Thesis material in `context/KJ_Thesis.zip` (primary source for architecture intent).
2. Prior implementations:
   - Attempt 1: `context/colab_LSTM.zip`
   - Attempt 2: `context/wsl_LSTM_legacy.zip`
3. Preprocessing handoff: `context/lstm_baseline_handoff.md`

When sources conflict, precedence is:

1. Thesis intent and methodology.
2. Data contract from preprocessing handoff.
3. Legacy implementation behavior.

## 3. Objectives
The pipeline must:

1. Train and evaluate LSTM models for `phi` estimation using synchronized 48 Hz data.
2. Reproduce thesis-style baseline experiments (stateful sequential training, batch size 1).
3. Eliminate ambiguous training behavior (data leakage, undocumented splits, hidden defaults).
4. Produce traceable artifacts (configs, checkpoints, metrics, predictions, metadata).
5. Support extension for additional features (gyro, cascaded models, alternative architectures).

## 4. Non-Goals (Baseline)
This baseline spec does not require:

1. Real-time embedded deployment.
2. Hyperparameter search automation.
3. Multi-objective loss across `phi/theta/psi`.
4. Distributed training.

## 5. Data Contract

### 5.1 Upstream data source
Expected input is a preprocessed HDF5 produced by preprocessing stage, default path:

- `outputs/preprocessed_all_trials.h5`

Raw CSV ingestion is out-of-scope for this repo's baseline.

### 5.2 HDF5 layout
Primary expected layout (handoff):

- Per-run tables: `/runs/<normalized_run_key>`
- Metadata tables:
  - `/meta/runs`
  - `/meta/run_logs`
  - `/meta/run_scaling`
  - `/meta/scaler_parameters`
  - `/meta/calibration`
  - `/meta/export_settings`

The loader must also support legacy flat keys (e.g. `/sFreehand_tt_1`) for backward compatibility.

### 5.3 Required columns
Minimum required for multivariate baseline:

- `pressure`
- `acc_x`
- `acc_y`
- `acc_z`
- `phi`

Recommended retained for diagnostics:

- `Time`
- `theta`, `psi`
- other columns from preprocessing schema

### 5.4 Temporal semantics

1. Default sampling rate is 48 Hz after 240 Hz -> moving-average + decimation by 5.
2. `Time` is seconds from run start (rebased).
3. Runs are independent sequences; windows must never cross run boundaries.

## 6. Feature, Target, and Scaling Policies

### 6.1 Baseline features and target

1. Multivariate feature set (thesis primary): `pressure`, `acc_x`, `acc_y`, `acc_z`.
2. Univariate comparison feature set: `pressure`.
3. Target: `phi` (major bending axis).

### 6.2 Scaling policy modes
The implementation must support three explicit scaling modes:

1. `fixed_bounds_thesis` (default for thesis reproduction)
2. `fit_train_only_minmax` (strict anti-leakage mode)
3. `passthrough` (when upstream export already applies approved scaling)

### 6.3 Fixed bounds (thesis-compatible constants)
From legacy training scripts used to reproduce thesis results:

- Accelerometer (m/s^2): `[-12, 12]`
- Pressure: `[400, 1800]`
- Target phi (deg): `[-200, 50]`
- Mapped range for scaled values: `[-1, 1]`

Behavioral requirement:

1. If raw accelerometer is in `g`, convert to m/s^2 before scaling.
2. Bounds must be persisted into run artifacts.
3. Denormalization for reporting must use the same bounds.

### 6.4 Leakage controls

1. Split must be run-level, not row-level.
2. If scaler fitting is used, fit strictly on training runs only.
3. Scaling mode must be recorded in manifest and metrics output.

## 7. Model Variants

### 7.1 Required architectures
As defined in thesis methodology:

1. `SLM-LSTM`: single-layer multivariate LSTM
2. `TLM-LSTM`: two-layer multivariate LSTM
3. `SLU-LSTM`: single-layer univariate LSTM
4. `TLU-LSTM`: two-layer univariate LSTM

### 7.2 Baseline layer specs

1. Single-layer models: 1 LSTM layer, 512 units, `tanh`, Dense(1) output.
2. Two-layer models: 2 LSTM layers, 256 units each, Dense(1) output.
3. Loss: MSE.
4. Optimizer: Adam, LR = 1e-3.

### 7.3 Stateful baseline behavior
For thesis-aligned baseline runs:

1. Use stateful LSTM configuration.
2. Batch size fixed at 1.
3. Process each run as one continuous sequence.
4. Reset recurrent states at run boundaries.

## 8. Dataset Splitting Strategy

### 8.1 Split granularity

- Splits are by run identity (entire runs), never by time rows.

### 8.2 Thesis-aligned train/validation pairing
Training is paired by condition (train repetition + validation repetition from same condition family), including:

1. One fixed-orientation predefined trajectory pair.
2. Dynamic predefined trajectory pair.
3. Dynamic low-pressure hold pair (~15 kPa family).
4. Dynamic higher-pressure hold pair (~45 kPa family).
5. Dynamic sinusoidal pair.

Implementation detail:

- Because dataset IDs differ between tables/scripts (indexing conventions), run-key resolution must come from `/meta/runs` metadata plus config-defined selection criteria, not hardcoded numeric assumptions.

### 8.3 Evaluation split
Evaluation set must include unseen runs across:

1. Fixed orientation not used in training.
2. Dynamic predefined trajectory unseen repetition.
3. Novel static pressure levels (including 30 kPa family where available).
4. Sinusoidal unseen repetition.

## 9. Training Loop Specification

### 9.1 Epoch routine (stateful profile)
For each epoch:

1. Iterate training run list in fixed deterministic order.
2. Before each run: reset model recurrent states.
3. Train exactly one pass on full run sequence.
4. Evaluate on paired validation run sequence.
5. Aggregate validation loss across all pairs.

### 9.2 Early stopping and checkpoints

1. Early stopping monitor: aggregate validation loss.
2. Patience: 10 epochs without improvement (thesis criterion).
3. Save best model checkpoint.
4. Save periodic checkpoints (configurable).

### 9.3 Reproducibility

1. Seed Python, NumPy, TensorFlow RNGs.
2. Persist exact config snapshot per run.
3. Log environment (Python/TensorFlow/CUDA/GPU info).
4. Record git commit hash and dirty flag.

## 10. Inference and Evaluation Specification

### 10.1 Inference mode

1. Use stateful inference for thesis profile.
2. Reset states at each new run.
3. No future-sample leakage during prediction.

### 10.2 Required metrics

1. RMSE (primary, in degrees).
2. MAE (secondary, in degrees).
3. Optional per-run MSE for debugging parity.

### 10.3 Required outputs
Per experiment run, emit:

1. `metrics.json` with per-run and aggregate metrics.
2. `predictions.h5` (or CSV bundle) with:
   - run key
   - time index / `Time`
   - `phi_true_deg`
   - `phi_pred_deg`
3. Training history log.
4. Best checkpoint path.

## 11. Repository Architecture (Target Scaffold)

### 11.1 High-level layout

- `configs/` for experiment and data definitions.
- `docs/specs/` for normative docs.
- `scripts/` for user entry points.
- `src/spa_lstm/` for importable package.
- `tests/` for unit and contract tests.

### 11.2 Package modules

1. `spa_lstm.data`
   - HDF5 loading and schema validation
   - split resolution
   - scaling utilities
2. `spa_lstm.models`
   - architecture factory for the four model variants
3. `spa_lstm.training`
   - stateful trainer orchestration
   - callbacks/checkpoint policy
4. `spa_lstm.evaluation`
   - metrics
   - prediction export

## 12. Configuration Requirements

Configurations must be declarative and versioned. Minimum config domains:

1. Data source + split selectors.
2. Feature/target selection.
3. Scaling mode and bounds.
4. Model variant and hyperparameters.
5. Training controls (epochs, patience, checkpoint cadence).
6. Output paths and run naming.

## 13. Engineering Quality Requirements

### 13.1 Validation and tests
At minimum:

1. Unit tests for scaling and denormalization correctness.
2. Unit tests for metric functions.
3. Loader tests for key resolution (legacy and `/runs/*` layouts).
4. Smoke test script for TensorFlow GPU availability (already present).

### 13.2 Logging and observability

1. Structured logging for run lifecycle events.
2. Persist per-epoch metrics and learning rate.
3. Save full exception traceback to run log directory on failure.

### 13.3 Failure handling
Fail fast with clear messages for:

1. Missing required columns.
2. Empty run selections.
3. Unknown run keys.
4. Invalid scaling mode.

## 14. Acceptance Criteria
This specification is satisfied when the scaffolded codebase can:

1. Validate and load HDF5 runs with documented schema checks.
2. Materialize thesis baseline config profiles without code edits.
3. Train a stateful baseline model end-to-end from config.
4. Export per-run predictions and RMSE/MAE metrics.
5. Re-run the same config reproducibly with traceable artifacts.

## 15. Planned Implementation Phases

1. Phase 1: data contract + config plumbing + loader/scaler tests.
2. Phase 2: model factory + stateful trainer core.
3. Phase 3: evaluation/export + reproducibility metadata.
4. Phase 4: parity checks versus legacy scripts and thesis plots.

