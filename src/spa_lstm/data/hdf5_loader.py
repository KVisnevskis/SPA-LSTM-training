"""HDF5 loading helpers with support for both legacy and new run layouts."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from spa_lstm.data.schema import require_columns


def list_run_keys(h5_path: str) -> list[str]:
    """List available run keys without leading group prefixes."""

    with pd.HDFStore(h5_path, mode="r") as store:
        keys = store.keys()

    run_keys: list[str] = []
    for key in keys:
        if key.startswith("/meta/"):
            continue
        if key.startswith("/runs/"):
            run_keys.append(key[len("/runs/") :])
        else:
            run_keys.append(key.lstrip("/"))
    return sorted(run_keys)


def _canonical(raw_key: str) -> str:
    key = raw_key
    if key.startswith("/runs/"):
        key = key[len("/runs/") :]
    key = key.lstrip("/")
    return key.lower()


def resolve_store_key(store: pd.HDFStore, run_key: str) -> str:
    """Resolve a run key to an actual HDFStore key.

    Supports:
    - `/runs/<key>` (new layout)
    - `/<key>` (legacy flat layout)
    - case-insensitive matching
    """

    candidates = [f"/runs/{run_key}", f"/{run_key}"]
    all_keys = store.keys()

    for candidate in candidates:
        if candidate in all_keys:
            return candidate

    canonical_map = {_canonical(key): key for key in all_keys if not key.startswith("/meta/")}
    canonical_query = _canonical(run_key)
    if canonical_query in canonical_map:
        return canonical_map[canonical_query]

    raise KeyError(f"Run key '{run_key}' not found in HDF5 store.")


def load_runs_as_dataframes(
    h5_path: str,
    run_keys: Iterable[str],
    required_columns: list[str] | tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Load requested runs as DataFrames keyed by requested run key."""

    result: dict[str, pd.DataFrame] = {}
    with pd.HDFStore(h5_path, mode="r") as store:
        for run_key in run_keys:
            store_key = resolve_store_key(store, run_key)
            df = store[store_key].copy()
            require_columns(df, required_columns, run_key)
            result[run_key] = df
    return result

