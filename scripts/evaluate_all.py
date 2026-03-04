#!/usr/bin/env python3
"""Run best-model evaluation for all experiment configs in a directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _collect_configs(config_dir: Path) -> list[Path]:
    configs = sorted(config_dir.glob("*.yaml"))
    configs.extend(sorted(config_dir.glob("*.yml")))

    unique: list[Path] = []
    seen: set[str] = set()
    for cfg in configs:
        key = str(cfg.resolve())
        if key in seen:
            continue
        unique.append(cfg)
        seen.add(key)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate all best models for configs in a directory.")
    parser.add_argument(
        "--config-dir",
        default="configs/experiments/baseline",
        help="Directory containing experiment YAML configs.",
    )
    parser.add_argument(
        "--scope",
        choices=("eval", "all"),
        default="eval",
        help="Evaluation scope: 'eval' for configured eval runs, 'all' for all HDF5 runs.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any config evaluation fails.",
    )
    parser.add_argument(
        "--fail-missing-model",
        action="store_true",
        help="Fail when best model is missing (default: skip missing models).",
    )
    args = parser.parse_args()

    from spa_lstm.config import load_experiment_config
    from spa_lstm.evaluation.workflow import evaluate_model

    config_dir = Path(args.config_dir).resolve()
    if not config_dir.exists() or not config_dir.is_dir():
        print(f"Config directory not found: {config_dir}", file=sys.stderr)
        return 1

    configs = _collect_configs(config_dir)
    if not configs:
        print(f"No YAML configs found in: {config_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(configs)} config(s) in {config_dir}")
    successes: list[Path] = []
    failures: list[tuple[Path, str]] = []
    skipped: list[tuple[Path, str]] = []

    for idx, cfg_path in enumerate(configs, 1):
        print(f"\n[{idx}/{len(configs)}] Evaluating config: {cfg_path}")
        try:
            cfg = load_experiment_config(cfg_path)
            run_dir = Path(cfg.runtime.output_dir) / cfg.runtime.run_name
            model_path = run_dir / cfg.runtime.save_best_path
            if not model_path.exists():
                message = f"Best model not found: {model_path}"
                if args.fail_missing_model:
                    raise FileNotFoundError(message)
                skipped.append((cfg_path, message))
                print(f"SKIP: {message}")
                continue

            metrics_path = evaluate_model(cfg, str(model_path), str(run_dir), scope=args.scope)
            successes.append(cfg_path)
            print(f"OK: model={model_path}")
            print(f"    metrics={metrics_path}")
        except Exception as exc:
            failures.append((cfg_path, str(exc)))
            print(f"FAILED: {cfg_path}", file=sys.stderr)
            print(f"Reason: {exc}", file=sys.stderr)
            if args.stop_on_error:
                break

    print("\n=== Evaluation Summary ===")
    print(f"Success: {len(successes)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed : {len(failures)}")
    if skipped:
        for path, reason in skipped:
            print(f"- SKIP {path}: {reason}")
    if failures:
        for path, reason in failures:
            print(f"- FAIL {path}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
