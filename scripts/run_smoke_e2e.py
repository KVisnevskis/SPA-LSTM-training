#!/usr/bin/env python3
"""Run a short end-to-end smoke cycle (train + eval) from a baseline config.

This helper keeps smoke artifacts organized under:
  <output-dir>/<run-name>/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short smoke training + evaluation.")
    parser.add_argument(
        "--base-config",
        default="configs/experiments/baseline/baseline_slm_lstm.yaml",
        help="Base experiment config YAML to derive smoke settings from.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/experiments/smoke",
        help="Parent output directory for smoke runs.",
    )
    parser.add_argument(
        "--run-name",
        default="smoke_slm_lstm_short_1epoch",
        help="Smoke run directory name.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs (default: 1).")
    parser.add_argument("--patience", type=int, default=1, help="Early-stopping patience (default: 1).")
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=1,
        help="Max number of (train,val) run pairs to keep from the base config (default: 1).",
    )
    parser.add_argument(
        "--max-eval-runs",
        type=int,
        default=2,
        help="Max number of eval runs to keep from the base config (default: 2).",
    )
    parser.add_argument(
        "--scope",
        choices=("eval", "all"),
        default="eval",
        help="Evaluation scope after training (default: eval).",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow reuse of an existing non-empty smoke run directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.epochs <= 0:
        print("--epochs must be > 0", file=sys.stderr)
        return 2
    if args.patience < 0:
        print("--patience must be >= 0", file=sys.stderr)
        return 2
    if args.max_pairs <= 0:
        print("--max-pairs must be > 0", file=sys.stderr)
        return 2
    if args.max_eval_runs <= 0:
        print("--max-eval-runs must be > 0", file=sys.stderr)
        return 2

    from spa_lstm.config import load_experiment_config
    from spa_lstm.evaluation.workflow import evaluate_model
    from spa_lstm.training.workflow import run_training

    cfg = load_experiment_config(args.base_config)

    pair_count = min(args.max_pairs, len(cfg.data.train_runs), len(cfg.data.val_runs))
    if pair_count == 0:
        print("Base config has no usable train/val run pairs for smoke training.", file=sys.stderr)
        return 2

    cfg.data.train_runs = cfg.data.train_runs[:pair_count]
    cfg.data.val_runs = cfg.data.val_runs[:pair_count]
    cfg.data.eval_runs = cfg.data.eval_runs[: args.max_eval_runs]
    if not cfg.data.eval_runs:
        print("Base config has no eval runs for smoke evaluation.", file=sys.stderr)
        return 2

    cfg.training.epochs = args.epochs
    cfg.training.patience = args.patience
    cfg.training.fit_verbose = 0
    cfg.training.eval_verbose = 0
    cfg.training.log_each_fit = False

    cfg.runtime.output_dir = args.output_dir
    cfg.runtime.run_name = args.run_name

    run_dir = Path(cfg.runtime.output_dir) / cfg.runtime.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_overwrite:
        print(
            f"Smoke run directory is non-empty: {run_dir}\n"
            "Use --allow-overwrite to reuse it.",
            file=sys.stderr,
        )
        return 1

    print("=== Smoke Train ===")
    print(f"base_config={args.base_config}")
    print(f"run_dir={run_dir}")
    print(f"train_pairs={pair_count}, eval_runs={len(cfg.data.eval_runs)}")
    trained_run_dir = run_training(cfg, resume=False)
    print(f"training_complete={trained_run_dir}")

    best_model = trained_run_dir / cfg.runtime.save_best_path
    if not best_model.exists():
        print(f"Best model missing after smoke training: {best_model}", file=sys.stderr)
        return 1

    print("=== Smoke Eval ===")
    metrics_path = evaluate_model(
        cfg=cfg,
        model_path=str(best_model),
        run_dir=str(trained_run_dir),
        scope=args.scope,
    )
    print(f"evaluation_complete={metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

