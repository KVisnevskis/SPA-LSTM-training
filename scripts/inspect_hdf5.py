#!/usr/bin/env python3
"""Inspect available run keys in a preprocessed HDF5 file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="List run keys in an HDF5 dataset.")
    parser.add_argument("--h5", required=True, help="Path to HDF5 file.")
    args = parser.parse_args()

    from spa_lstm.data.hdf5_loader import list_run_keys

    for run_key in list_run_keys(args.h5):
        print(run_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
