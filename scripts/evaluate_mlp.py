#!/usr/bin/env python3
"""Evaluate a trained MLP model on configured evaluation runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MLP baseline model from config.")
    parser.add_argument("--config", required=True, help="Path to MLP experiment config YAML.")
    parser.add_argument("--model", required=True, help="Path to trained Keras model file.")
    parser.add_argument("--run-dir", default=None, help="Optional output directory override.")
    parser.add_argument(
        "--scope",
        choices=("eval", "all"),
        default="eval",
        help="Evaluation scope: 'eval' for config eval runs, 'all' for all HDF5 runs.",
    )
    args = parser.parse_args()

    from mlp.config import load_experiment_config
    from mlp.evaluation import evaluate_model

    cfg = load_experiment_config(args.config)
    metrics_path = evaluate_model(cfg, args.model, args.run_dir, scope=args.scope)
    print(f"Evaluation complete. Metrics file: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
