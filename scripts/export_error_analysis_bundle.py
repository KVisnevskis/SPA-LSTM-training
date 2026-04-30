#!/usr/bin/env python3
"""Export a comparative-analysis bundle from experiment outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spa_lstm.evaluation.error_analysis_bundle import (
    build_error_analysis_bundle,
    discover_experiment_configs,
    discover_experiment_configs_many,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package experiment predictions into an HDF5/CSV error-analysis bundle "
            "compatible with the example comparative-analysis export."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="config_paths",
        help=(
            "Explicit experiment config YAML to package. "
            "Repeat to package an exact set of configs."
        ),
    )
    parser.add_argument(
        "--config-dir",
        action="append",
        dest="config_dirs",
        help=(
            "Directory containing experiment YAML configs to package. "
            "Repeat to include multiple experiment families. "
            "Default: configs/experiments/baseline"
        ),
    )
    parser.add_argument(
        "--glob",
        default="*.yaml",
        help="Glob pattern used inside --config-dir (default: %(default)s).",
    )
    parser.add_argument(
        "--bundle-dir",
        default="outputs/error_analysis_bundle",
        help="Output directory for the exported bundle (default: %(default)s).",
    )
    parser.add_argument(
        "--store-name",
        default="spa_lstm_error_analysis.h5",
        help="Filename for the bundled HDF5 store (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    config_paths: list[Path] = []
    seen: set[Path] = set()

    explicit_config_paths = [Path(path).resolve() for path in (args.config_paths or [])]
    for path in explicit_config_paths:
        if path in seen:
            continue
        seen.add(path)
        config_paths.append(path)

    config_dirs = args.config_dirs or []
    if not config_dirs and not explicit_config_paths:
        config_dirs = ["configs/experiments/baseline"]

    discovered_paths: list[Path] = []
    if len(config_dirs) == 1:
        discovered_paths = discover_experiment_configs(config_dirs[0], glob_pattern=args.glob)
    elif len(config_dirs) > 1:
        discovered_paths = discover_experiment_configs_many(config_dirs, glob_pattern=args.glob)

    for path in discovered_paths:
        resolved = Path(path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        config_paths.append(resolved)

    result = build_error_analysis_bundle(
        config_paths=config_paths,
        bundle_dir=args.bundle_dir,
        repo_root=repo_root,
        store_name=args.store_name,
    )

    print(f"Bundle written to: {result.bundle_root}")
    print(f"Models included: {result.included_model_count}")
    print(f"Run tables included: {result.included_run_table_count}")
    print(f"Prediction rows included: {result.included_prediction_row_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
