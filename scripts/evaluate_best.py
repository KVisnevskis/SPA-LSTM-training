#!/usr/bin/env python3
"""Evaluate the best checkpoint for an experiment config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the best SPA-LSTM model from config.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional run directory override. Defaults to <output_dir>/<run_name> from config.",
    )
    args = parser.parse_args()

    from spa_lstm.config import load_experiment_config
    from spa_lstm.evaluation.workflow import evaluate_model

    cfg = load_experiment_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else Path(cfg.runtime.output_dir) / cfg.runtime.run_name
    model_path = run_dir / cfg.runtime.save_best_path

    if not model_path.exists():
        print(f"Best model not found: {model_path}", file=sys.stderr)
        return 1

    metrics_path = evaluate_model(cfg, str(model_path), str(run_dir))
    print(f"Evaluation complete. Model: {model_path}")
    print(f"Metrics file: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
