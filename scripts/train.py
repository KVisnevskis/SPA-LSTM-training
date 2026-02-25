#!/usr/bin/env python3
"""Run a training experiment from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SPA-LSTM model from config.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    args = parser.parse_args()

    from spa_lstm.config import load_experiment_config
    from spa_lstm.training.workflow import run_training

    cfg = load_experiment_config(args.config)
    out_dir = run_training(cfg)
    print(f"Training complete. Artifacts in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
