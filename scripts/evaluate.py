#!/usr/bin/env python3
"""Evaluate a trained model on configured evaluation runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SPA-LSTM model from config.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument("--model", required=True, help="Path to trained Keras model file.")
    parser.add_argument("--run-dir", default=None, help="Optional output directory override.")
    args = parser.parse_args()

    from spa_lstm.config import load_experiment_config
    from spa_lstm.evaluation.workflow import evaluate_model

    cfg = load_experiment_config(args.config)
    metrics_path = evaluate_model(cfg, args.model, args.run_dir)
    print(f"Evaluation complete. Metrics file: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
