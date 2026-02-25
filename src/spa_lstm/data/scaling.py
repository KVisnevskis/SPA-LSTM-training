"""Scaling and denormalization utilities for SPA-LSTM experiments."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from spa_lstm.config import ColumnBounds, ScalingConfig
from spa_lstm.data.constants import THESIS_FIXED_BOUNDS


def minmax_scale(values: np.ndarray, bounds: ColumnBounds, out_min: float, out_max: float) -> np.ndarray:
    """Map values from [lo, hi] -> [out_min, out_max]."""

    return (values - bounds.lo) / (bounds.hi - bounds.lo) * (out_max - out_min) + out_min


def minmax_inverse(values: np.ndarray, bounds: ColumnBounds, out_min: float, out_max: float) -> np.ndarray:
    """Map values from [out_min, out_max] -> [lo, hi]."""

    return (values - out_min) / (out_max - out_min) * (bounds.hi - bounds.lo) + bounds.lo


def _convert_acc_to_mps2(df: pd.DataFrame) -> None:
    for column in ("acc_x", "acc_y", "acc_z"):
        if column in df.columns:
            df.loc[:, column] = df[column] * 9.81


def compute_train_minmax_bounds(train_runs: dict[str, pd.DataFrame], columns: list[str]) -> dict[str, ColumnBounds]:
    """Compute per-column min/max across training runs only."""

    bounds: dict[str, ColumnBounds] = {}
    for column in columns:
        values = np.concatenate([df[column].to_numpy(dtype=np.float64) for df in train_runs.values()])
        lo = float(np.min(values))
        hi = float(np.max(values))
        if hi <= lo:
            raise ValueError(f"Cannot compute bounds for '{column}': max <= min.")
        bounds[column] = ColumnBounds(lo=lo, hi=hi)
    return bounds


def resolve_scaling_bounds(
    scaling_cfg: ScalingConfig,
    train_runs: dict[str, pd.DataFrame],
    feature_columns: list[str],
    target_column: str,
) -> dict[str, ColumnBounds]:
    """Resolve concrete bounds based on selected scaling policy."""

    if scaling_cfg.mode == "passthrough":
        return {}

    if scaling_cfg.mode == "fixed_bounds_thesis":
        return scaling_cfg.bounds or THESIS_FIXED_BOUNDS

    if scaling_cfg.mode == "fit_train_only_minmax":
        columns = list(dict.fromkeys(feature_columns + [target_column]))
        return compute_train_minmax_bounds(train_runs, columns)

    raise ValueError(f"Unknown scaling mode: {scaling_cfg.mode}")


def scale_dataframe(
    df: pd.DataFrame,
    scaling_cfg: ScalingConfig,
    bounds: dict[str, ColumnBounds],
    feature_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    """Scale a run DataFrame according to configured policy."""

    out = df.copy()

    if scaling_cfg.accelerometer_in_g and scaling_cfg.mode != "passthrough":
        _convert_acc_to_mps2(out)

    if scaling_cfg.mode == "passthrough":
        return out

    scale_columns = list(dict.fromkeys(feature_columns + [target_column]))
    for column in scale_columns:
        if column not in bounds:
            raise KeyError(f"Missing bounds for column '{column}'.")
        out.loc[:, column] = minmax_scale(
            out[column].to_numpy(dtype=np.float64),
            bounds[column],
            scaling_cfg.output_min,
            scaling_cfg.output_max,
        )

    return out


def denormalize_target(
    target_values: np.ndarray,
    target_column: str,
    scaling_cfg: ScalingConfig,
    bounds: dict[str, ColumnBounds],
) -> np.ndarray:
    """Map scaled target back to physical units."""

    if scaling_cfg.mode == "passthrough":
        return target_values
    if target_column not in bounds:
        raise KeyError(f"Missing bounds for target column '{target_column}'.")
    return minmax_inverse(target_values, bounds[target_column], scaling_cfg.output_min, scaling_cfg.output_max)


def save_bounds_json(path: str | Path, bounds: dict[str, ColumnBounds]) -> None:
    """Persist bounds for reproducibility and downstream inference."""

    serializable = {column: asdict(value) for column, value in bounds.items()}
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def load_bounds_json(path: str | Path) -> dict[str, ColumnBounds]:
    """Load bounds from JSON artifact."""

    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {column: ColumnBounds(lo=float(item["lo"]), hi=float(item["hi"])) for column, item in raw.items()}

