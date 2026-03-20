#!/usr/bin/env python3
"""Generate a compact LaTeX/CSV table summarizing seeded training history.

The table compares the seeded SLM-LSTM runs under
`outputs/experiments/baseline_slm_seeded_replication` and, by default, also
includes the original baseline SLM-LSTM run as `Seed 42`.

Timing notes:
- `Measured wall time [h]` is derived from the final `elapsed_seconds` sample in
  `resource_usage.csv`.
- For resumed runs, this only reflects the final resumed segment because the
  resource log is overwritten on resume.
- `Time to best epoch [h]` is therefore only reported for non-resumed runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _latex_escape(text: str) -> str:
    escaped = text
    replacements = (
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    )
    for old, new in replacements:
        escaped = escaped.replace(old, new)
    return escaped


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_seed_run_info(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "training_summary.json"
    snapshot_path = run_dir / "config_snapshot.json"
    resource_path = run_dir / "resource_usage.csv"
    if not summary_path.exists() or not snapshot_path.exists():
        return None

    summary = _load_json(summary_path)
    snapshot = _load_json(snapshot_path)
    if not isinstance(summary, dict) or not isinstance(snapshot, dict):
        return None

    training_cfg = snapshot.get("training")
    if not isinstance(training_cfg, dict):
        return None

    raw_seed = training_cfg.get("seed")
    if not isinstance(raw_seed, (int, float)):
        return None
    seed = int(raw_seed)

    epochs_completed = int(summary.get("epochs_completed", 0))
    best_epoch = int(summary.get("best_epoch", 0))
    best_val_loss = float(summary.get("best_val_loss", float("nan")))
    resumed = bool(summary.get("resumed_from_checkpoint", False))
    stopped_early = bool(summary.get("stopped_early", False))
    measured_hours = _read_measured_hours(resource_path)

    time_to_best_hours: float | None
    timing_scope: str
    if resumed:
        time_to_best_hours = None
        timing_scope = "final segment only"
    else:
        time_to_best_hours = (
            measured_hours * (best_epoch / epochs_completed)
            if measured_hours is not None and epochs_completed > 0
            else None
        )
        timing_scope = "full run"

    return {
        "seed": seed,
        "run_name": run_dir.name,
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "post_best_epochs": max(0, epochs_completed - best_epoch),
        "best_val_loss": best_val_loss,
        "measured_wall_time_h": measured_hours,
        "time_to_best_epoch_h": time_to_best_hours,
        "stopped_early": stopped_early,
        "resumed": resumed,
        "timing_scope": timing_scope,
    }


def _read_measured_hours(resource_path: Path) -> float | None:
    if not resource_path.exists():
        return None

    last_elapsed_seconds: float | None = None
    with resource_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_elapsed = row.get("elapsed_seconds")
            if raw_elapsed is None:
                continue
            try:
                last_elapsed_seconds = float(raw_elapsed)
            except ValueError:
                continue

    if last_elapsed_seconds is None:
        return None
    return last_elapsed_seconds / 3600.0


def _discover_seed_rows(
    seed_run_root: Path,
    base_run_dir: Path | None,
    include_base_run: bool,
) -> list[dict[str, Any]]:
    discovered_by_seed: dict[int, dict[str, Any]] = {}

    candidate_dirs: list[Path] = []
    if include_base_run:
        if base_run_dir is None:
            raise ValueError("include_base_run=True requires a base_run_dir.")
        candidate_dirs.append(base_run_dir)
    candidate_dirs.extend(sorted(path for path in seed_run_root.iterdir() if path.is_dir()))

    for run_dir in candidate_dirs:
        row = _load_seed_run_info(run_dir)
        if row is None:
            continue
        seed = int(row["seed"])
        if seed in discovered_by_seed:
            other_dir = discovered_by_seed[seed]["run_name"]
            raise ValueError(f"Duplicate training history entries for seed {seed}: {other_dir} and {run_dir.name}")
        discovered_by_seed[seed] = row

    if not discovered_by_seed:
        raise ValueError(f"No seeded training runs found in {seed_run_root}")
    return [discovered_by_seed[seed] for seed in sorted(discovered_by_seed)]


def _fmt_float(value: float | None, decimals: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def _fmt_bool(value: bool) -> str:
    return "yes" if value else "no"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "seed",
        "run_name",
        "best_epoch",
        "epochs_completed",
        "post_best_epochs",
        "best_val_loss",
        "measured_wall_time_h",
        "time_to_best_epoch_h",
        "stopped_early",
        "resumed",
        "timing_scope",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_tex(
    path: Path,
    rows: list[dict[str, Any]],
    caption: str,
    label: str,
    decimals: int,
) -> None:
    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + _latex_escape(caption) + r"}")
    lines.append(r"\label{" + _latex_escape(label) + r"}")
    lines.append(r"\begin{tabular}{rrrrrrrc}")
    lines.append(r"\toprule")
    lines.append(
        r"Seed & Best epoch & Epochs completed & Post-best epochs & Best val. loss & "
        r"Measured wall time [h] & Time to best epoch [h] & Timing scope \\"
    )
    lines.append(r"\midrule")
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                int(row["seed"]),
                int(row["best_epoch"]),
                int(row["epochs_completed"]),
                int(row["post_best_epochs"]),
                _fmt_float(float(row["best_val_loss"]), decimals=4),
                _fmt_float(row["measured_wall_time_h"], decimals=decimals),
                _fmt_float(row["time_to_best_epoch_h"], decimals=decimals),
                _latex_escape(str(row["timing_scope"])),
            )
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate a compact training-history table for seeded SLM-LSTM runs."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help=f"Repository root path (default: {default_repo_root}).",
    )
    parser.add_argument(
        "--seed-run-root",
        type=Path,
        default=Path("outputs/experiments/baseline_slm_seeded_replication"),
        help="Directory containing seeded run subdirectories.",
    )
    parser.add_argument(
        "--base-run-dir",
        type=Path,
        default=Path("outputs/experiments/baseline/baseline_slm_lstm"),
        help="Original baseline SLM-LSTM run directory to include as Seed 42.",
    )
    parser.add_argument(
        "--include-base-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the original baseline SLM-LSTM run as Seed 42 (default: true).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/tables/ch4/baseline_slm_seeded_training_history.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=Path("outputs/tables/ch4/baseline_slm_seeded_training_history.tex"),
        help="Output LaTeX table path.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Training-history comparison for seeded SLM-LSTM baseline runs. "
            "Measured wall time is derived from the final resource-monitor sample. "
            "For resumed runs, this reflects only the final resumed segment, so "
            "time to best epoch is reported as n/a."
        ),
        help="LaTeX caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:baseline-slm-seeded-training-history",
        help="LaTeX table label.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places for wall-time values (default: 2).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    seed_run_root = args.seed_run_root if args.seed_run_root.is_absolute() else (repo_root / args.seed_run_root)
    base_run_dir = args.base_run_dir if args.base_run_dir.is_absolute() else (repo_root / args.base_run_dir)
    output_csv = args.output_csv if args.output_csv.is_absolute() else (repo_root / args.output_csv)
    output_tex = args.output_tex if args.output_tex.is_absolute() else (repo_root / args.output_tex)

    rows = _discover_seed_rows(
        seed_run_root=seed_run_root,
        base_run_dir=base_run_dir,
        include_base_run=bool(args.include_base_run),
    )
    _write_csv(output_csv, rows)
    _write_tex(
        path=output_tex,
        rows=rows,
        caption=args.caption,
        label=args.label,
        decimals=args.decimals,
    )

    print(f"Wrote CSV: {output_csv}")
    print(f"Wrote TeX: {output_tex}")
    print(f"Seeds: {', '.join(str(int(row['seed'])) for row in rows)}")


if __name__ == "__main__":
    main()
