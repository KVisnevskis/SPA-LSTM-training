#!/usr/bin/env python3
"""Run training for all MLP experiment configs in a directory."""

from __future__ import annotations

import argparse
import sys
import traceback
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


def _run_dir_has_artifacts(run_dir: Path) -> bool:
    if not run_dir.exists():
        return False
    return any(run_dir.iterdir())


def main() -> int:
    parser = argparse.ArgumentParser(description="Train all MLP configs in a directory.")
    parser.add_argument(
        "--config-dir",
        default="configs/experiments/mlp/hpo",
        help="Directory containing MLP experiment YAML configs.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when any config fails.",
    )
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
        help="Override training.verbose for all loaded configs (e.g. 1).",
    )
    args = parser.parse_args()

    from mlp.config import load_experiment_config
    from mlp.training import run_training

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

    for idx, cfg_path in enumerate(configs, 1):
        print(f"\n[{idx}/{len(configs)}] Training config: {cfg_path}")
        cfg = None
        try:
            cfg = load_experiment_config(cfg_path)
            if args.training_verbose is not None:
                cfg.training.verbose = int(args.training_verbose)

            print(f"  training flags: verbose={cfg.training.verbose}")

            run_dir = Path(cfg.runtime.output_dir) / cfg.runtime.run_name
            if _run_dir_has_artifacts(run_dir) and not args.allow_overwrite:
                raise FileExistsError(
                    f"Run directory already exists and is non-empty: {run_dir}. "
                    "Use --allow-overwrite to bypass this safety check."
                )

            out_dir = run_training(cfg)
        except Exception as exc:
            error_msg = str(exc)
            failures.append((cfg_path, error_msg))
            print(f"FAILED: {cfg_path}", file=sys.stderr)
            print(f"Reason: {error_msg}", file=sys.stderr)

            try:
                if cfg is not None and not isinstance(exc, FileExistsError):
                    output_dir = Path(cfg.runtime.output_dir) / cfg.runtime.run_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    error_log = output_dir / "training_error.log"
                    with error_log.open("w", encoding="utf-8") as f:
                        f.write(traceback.format_exc())
                    print(f"Traceback saved to: {error_log}", file=sys.stderr)
            except Exception:
                pass

            if args.stop_on_error:
                break
            continue

        successes.append(cfg_path)
        print(f"OK: {cfg_path}")
        print(f"Artifacts: {out_dir}")

    print("\n=== Training Summary ===")
    print(f"Success: {len(successes)}")
    print(f"Failed : {len(failures)}")
    if failures:
        for path, err in failures:
            print(f"- {path}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
