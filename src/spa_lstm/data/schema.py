"""Schema validation for run data tables."""

from __future__ import annotations

import pandas as pd


def require_columns(df: pd.DataFrame, required: list[str] | tuple[str, ...], run_key: str) -> None:
    """Raise an error when any required column is missing from a run table."""

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Run '{run_key}' is missing required columns: {missing}")

