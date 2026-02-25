#!/usr/bin/env python3
"""Run a training experiment from a YAML config."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SPA-LSTM model from config.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint in the run output directory.",
    )
    args = parser.parse_args()

    from spa_lstm.config import load_experiment_config
    from spa_lstm.training.workflow import run_training

    cfg = load_experiment_config(args.config)
    output_dir = Path(cfg.runtime.output_dir) / cfg.runtime.run_name
    try:
        out_dir = run_training(cfg, resume=args.resume)
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        error_log = output_dir / "training_error.log"
        with error_log.open("w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"Training failed: {exc}", file=sys.stderr)
        print(f"Traceback saved to: {error_log}", file=sys.stderr)
        return 1

    print(f"Training complete. Artifacts in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
