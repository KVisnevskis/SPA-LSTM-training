"""MLP row-wise dataset assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlp.config import ExperimentConfig
from spa_lstm.config import ColumnBounds
from spa_lstm.data.hdf5_loader import load_runs_as_dataframes
from spa_lstm.data.scaling import load_hdf5_scaler_bounds
from spa_lstm.data.splits import assert_disjoint_splits, assert_no_duplicate_runs

__all__ = ["PreparedTrainingData", "prepare_training_data"]


@dataclass(frozen=True)
class PreparedTrainingData:
    """Concatenated train/validation arrays for MLP training."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    bounds: dict[str, ColumnBounds]
    train_row_counts: dict[str, int]
    val_row_counts: dict[str, int]


def _require_dataset_exists(h5_path: str) -> None:
    dataset_path = Path(h5_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"HDF5 dataset not found at '{dataset_path}'. "
            "Generate it first (see context/lstm_baseline_handoff.md)."
        )


def _validate_run_arrays(
    runs: dict[str, Any],
    run_keys: list[str],
    feature_columns: list[str],
    target_column: str,
    split_name: str,
) -> None:
    columns = list(dict.fromkeys(feature_columns + [target_column]))
    for run_key in run_keys:
        df = runs[run_key]
        if len(df) == 0:
            raise ValueError(f"{split_name} run '{run_key}' has zero rows.")
        try:
            arr = df[columns].to_numpy(dtype=np.float64)
        except Exception as exc:
            raise ValueError(
                f"{split_name} run '{run_key}' has non-numeric values in required columns {columns}."
            ) from exc
        if arr.ndim != 2 or arr.shape[1] != len(columns):
            raise ValueError(
                f"{split_name} run '{run_key}' has unexpected array shape {arr.shape} for columns {columns}."
            )
        if not np.isfinite(arr).all():
            raise ValueError(
                f"{split_name} run '{run_key}' has non-finite values (NaN/Inf) in required columns {columns}."
            )


def _concat_xy(
    runs: dict[str, Any],
    run_keys: list[str],
    features: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    row_counts: dict[str, int] = {}

    for run_key in run_keys:
        df = runs[run_key]
        x_part = df[features].to_numpy(dtype=np.float32)
        y_part = df[[target]].to_numpy(dtype=np.float32)
        x_parts.append(x_part)
        y_parts.append(y_part)
        row_counts[run_key] = int(len(df))

    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return x, y, row_counts


def prepare_training_data(cfg: ExperimentConfig) -> PreparedTrainingData:
    """Load and concatenate row-wise train/validation data for MLP training."""

    cfg.validate()
    _require_dataset_exists(cfg.data.h5_path)
    assert_no_duplicate_runs(cfg.data.train_runs, cfg.data.val_runs, cfg.data.eval_runs)
    assert_disjoint_splits(cfg.data.train_runs, cfg.data.val_runs, cfg.data.eval_runs)

    required_columns = tuple(dict.fromkeys(cfg.data.features + [cfg.data.target]))
    train_raw = load_runs_as_dataframes(cfg.data.h5_path, cfg.data.train_runs, required_columns)
    val_raw = load_runs_as_dataframes(cfg.data.h5_path, cfg.data.val_runs, required_columns)

    _validate_run_arrays(train_raw, cfg.data.train_runs, cfg.data.features, cfg.data.target, "train")
    _validate_run_arrays(val_raw, cfg.data.val_runs, cfg.data.features, cfg.data.target, "val")

    x_train, y_train, train_row_counts = _concat_xy(train_raw, cfg.data.train_runs, cfg.data.features, cfg.data.target)
    x_val, y_val, val_row_counts = _concat_xy(val_raw, cfg.data.val_runs, cfg.data.features, cfg.data.target)

    scale_columns = list(dict.fromkeys(cfg.data.features + [cfg.data.target]))
    bounds = load_hdf5_scaler_bounds(cfg.data.h5_path, columns=scale_columns)

    return PreparedTrainingData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        bounds=bounds,
        train_row_counts=train_row_counts,
        val_row_counts=val_row_counts,
    )
