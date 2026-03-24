#!/usr/bin/env python3
"""Run an MLP training experiment from a YAML config."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MLP baseline model from config.")
    parser.add_argument("--config", required=True, help="Path to MLP experiment config YAML.")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow training into an existing non-empty run directory (disabled by default).",
    )
    parser.add_argument(
        "--verbose",
        "--training-verbose",
        dest="training_verbose",
        type=int,
        default=None,
        help="Override training.verbose from config (e.g. 1).",
    )
    args = parser.parse_args()

    from mlp.config import load_experiment_config
    from mlp.training import run_training

    cfg = load_experiment_config(args.config)
    if args.training_verbose is not None:
        cfg.training.verbose = int(args.training_verbose)

    print(f"training flags: verbose={cfg.training.verbose}")

    output_dir = Path(cfg.runtime.output_dir) / cfg.runtime.run_name
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_overwrite:
        print(
            f"Run directory already exists and is non-empty: {output_dir}. "
            "Use --allow-overwrite to bypass this safety check.",
            file=sys.stderr,
        )
        return 1

    try:
        out_dir = run_training(cfg)
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
