#!/usr/bin/env python3
"""Validate that an experiment config is consistent with an HDF5 dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that config run keys, columns, and scaler metadata exist in HDF5."
    )
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    args = parser.parse_args()

    from spa_lstm.config import load_experiment_config
    from spa_lstm.data.constants import REQUIRED_BASE_COLUMNS
    from spa_lstm.data.hdf5_loader import list_run_keys, load_runs_as_dataframes
    from spa_lstm.data.scaling import load_hdf5_scaler_bounds

    try:
        cfg = load_experiment_config(args.config)
    except Exception as exc:
        print(f"Config load failed: {exc}", file=sys.stderr)
        return 1

    h5_path = Path(cfg.data.h5_path)
    if not h5_path.exists():
        print(f"HDF5 dataset not found: {h5_path}", file=sys.stderr)
        return 1

    available_runs = set(list_run_keys(str(h5_path)))
    requested_runs = _ordered_unique(cfg.data.train_runs + cfg.data.val_runs + cfg.data.eval_runs)
    missing_runs = sorted(set(requested_runs) - available_runs)

    if missing_runs:
        preview = ", ".join(missing_runs[:10])
        print(
            f"Missing run keys in {h5_path}: {preview}"
            + (" ..." if len(missing_runs) > 10 else ""),
            file=sys.stderr,
        )
        return 1

    required_columns = tuple(dict.fromkeys(list(REQUIRED_BASE_COLUMNS) + cfg.data.features + [cfg.data.target]))
    try:
        load_runs_as_dataframes(str(h5_path), requested_runs, required_columns)
    except Exception as exc:
        print(f"Run table validation failed: {exc}", file=sys.stderr)
        return 1

    scale_columns = list(dict.fromkeys(cfg.data.features + [cfg.data.target]))
    try:
        load_hdf5_scaler_bounds(str(h5_path), columns=scale_columns)
    except Exception as exc:
        print(f"Scaler metadata validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Config data check passed: {args.config}")
    print(f"HDF5: {h5_path}")
    print(
        "Runs: "
        f"train={len(cfg.data.train_runs)}, val={len(cfg.data.val_runs)}, eval={len(cfg.data.eval_runs)}, "
        f"unique={len(requested_runs)}"
    )
    print(f"Columns checked: {', '.join(required_columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
