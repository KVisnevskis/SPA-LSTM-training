# SPA-LSTM Training

Repository scaffold for a rigorous, reproducible LSTM training pipeline that estimates soft pneumatic actuator (SPA) bending angle (`phi`) from pressure and IMU data.

## Current State

The repo now contains:

1. A normative training specification: `docs/specs/lstm_training_spec.md`
2. A thesis-aligned architectural decision record: `docs/adr/0001-thesis-stateful-baseline.md`
3. A modular package scaffold under `src/spa_lstm`
4. Versioned experiment/data/training configs under `configs/`
5. Initial unit tests for math and scaling utilities under `tests/`

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
```

Run utility commands:

```bash
python3 scripts/inspect_hdf5.py --h5 outputs/preprocessed_all_trials.h5
python3 scripts/train.py --config configs/experiments/thesis_slm_lstm.yaml
python3 scripts/evaluate.py --config configs/experiments/thesis_slm_lstm.yaml
python3 scripts/view_predictions.py --root outputs/experiments
```

## Data Prerequisite

Raw datasets are not tracked in git. See preprocessing handoff notes in `context/lstm_baseline_handoff.md` and generate the HDF5 export before training.

## Design Reference

Use this document as source of truth for implementation details:

- `docs/specs/lstm_training_spec.md`
