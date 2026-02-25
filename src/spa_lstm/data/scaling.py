"""De-scaling utilities for model inference outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from spa_lstm.config import ColumnBounds


def minmax_inverse(values: np.ndarray, bounds: ColumnBounds, out_min: float = -1.0, out_max: float = 1.0) -> np.ndarray:
    """Map values from normalized range [out_min, out_max] back to [lo, hi]."""

    return (values - out_min) / (out_max - out_min) * (bounds.hi - bounds.lo) + bounds.lo


def load_hdf5_scaler_bounds(h5_path: str, columns: list[str] | None = None) -> dict[str, ColumnBounds]:
    """Load per-column scaler bounds from /meta/scaler_parameters."""

    with pd.HDFStore(h5_path, mode="r") as store:
        if "/meta/scaler_parameters" not in store.keys():
            raise KeyError("Missing required scaler metadata table: /meta/scaler_parameters")
        scaler_df = store["/meta/scaler_parameters"]

    required = {"column", "min", "max"}
    missing = required - set(scaler_df.columns)
    if missing:
        raise KeyError(f"/meta/scaler_parameters missing required columns: {sorted(missing)}")

    requested = set(columns or [])
    bounds: dict[str, ColumnBounds] = {}
    for _, row in scaler_df.iterrows():
        column = str(row["column"])
        if requested and column not in requested:
            continue
        bounds[column] = ColumnBounds(lo=float(row["min"]), hi=float(row["max"]))

    if requested:
        unresolved = sorted(requested - set(bounds))
        if unresolved:
            raise KeyError(f"Scaler metadata missing requested columns: {unresolved}")

    return bounds


def denormalize_target(
    target_values: np.ndarray,
    target_column: str,
    bounds: dict[str, ColumnBounds],
    out_min: float = -1.0,
    out_max: float = 1.0,
) -> np.ndarray:
    """De-scale target predictions from normalized range to physical units."""

    if target_column not in bounds:
        raise KeyError(f"Missing bounds for target column '{target_column}'.")
    return minmax_inverse(np.asarray(target_values, dtype=np.float64), bounds[target_column], out_min, out_max)
