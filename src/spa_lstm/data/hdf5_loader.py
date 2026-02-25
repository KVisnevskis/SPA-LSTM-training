"""HDF5 loading helpers for expected layout: /runs/<run_key>."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from spa_lstm.data.schema import require_columns


def _open_hdfstore(h5_path: str, mode: str = "r") -> pd.HDFStore:
    try:
        return pd.HDFStore(h5_path, mode=mode)
    except ImportError as exc:
        raise RuntimeError(
            "PyTables dependency is missing. Install package 'tables' to read HDF5 datasets."
        ) from exc


def list_run_keys(h5_path: str) -> list[str]:
    """List available run keys under /runs/."""

    with _open_hdfstore(h5_path, mode="r") as store:
        keys = store.keys()

    return sorted(key[len("/runs/") :] for key in keys if key.startswith("/runs/"))


def _as_runs_key(run_key: str) -> str:
    key = run_key.strip()
    if key.startswith("/runs/"):
        return key
    if key.startswith("/"):
        key = key[1:]
    return f"/runs/{key}"


def resolve_store_key(store: pd.HDFStore, run_key: str) -> str:
    """Resolve a run key to /runs/<run_key> in the expected layout."""

    candidate = _as_runs_key(run_key)
    all_keys = store.keys()
    if candidate in all_keys:
        return candidate

    available = sorted(key[len("/runs/") :] for key in all_keys if key.startswith("/runs/"))
    preview = ", ".join(available[:10]) if available else "<none>"
    raise KeyError(
        f"Run key '{run_key}' not found in HDF5 store. "
        f"Expected key '{candidate}' under /runs/. "
        f"Available /runs keys (first 10): {preview}"
    )


def load_runs_as_dataframes(
    h5_path: str,
    run_keys: Iterable[str],
    required_columns: list[str] | tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Load requested runs as DataFrames keyed by requested run key."""

    result: dict[str, pd.DataFrame] = {}
    with _open_hdfstore(h5_path, mode="r") as store:
        for run_key in run_keys:
            store_key = resolve_store_key(store, run_key)
            df = store[store_key].copy()
            require_columns(df, required_columns, run_key)
            result[run_key] = df
    return result
