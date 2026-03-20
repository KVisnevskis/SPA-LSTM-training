#!/usr/bin/env python3
"""Generate a compact training-summary table for seeded SLM-LSTM runs.

This mirrors the structure of the main `baseline_training_summary` table, but
uses seed as the comparison axis and intentionally omits training time.

By default it includes:
- seeded replications under `outputs/experiments/baseline_slm_seeded_replication`
- the original baseline SLM-LSTM run as `Seed 42`
"""

from __future__ import annotations

import argparse
import csv
import json
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


def _read_last_history_row(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No history rows found in {path}")
    return rows[-1]


def _load_seed_row(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "training_summary.json"
    snapshot_path = run_dir / "config_snapshot.json"
    history_path = run_dir / "history.csv"
    if not summary_path.exists() or not snapshot_path.exists() or not history_path.exists():
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

    last_history = _read_last_history_row(history_path)
    return {
        "seed": int(raw_seed),
        "run_name": run_dir.name,
        "epochs_completed": int(summary["epochs_completed"]),
        "best_epoch": int(summary["best_epoch"]),
        "final_train_loss": float(last_history["train_loss_mean"]),
        "final_val_loss": float(last_history["val_loss_mean"]),
        "resumed": bool(summary.get("resumed_from_checkpoint", False)),
    }


def _discover_rows(
    seed_run_root: Path,
    base_run_dir: Path | None,
    include_base_run: bool,
) -> list[dict[str, Any]]:
    discovered_by_seed: dict[int, dict[str, Any]] = {}

    candidate_dirs: list[Path] = []
    if include_base_run:
        if base_run_dir is None:
            raise ValueError("include_base_run=True requires base_run_dir.")
        candidate_dirs.append(base_run_dir)
    candidate_dirs.extend(sorted(path for path in seed_run_root.iterdir() if path.is_dir()))

    for run_dir in candidate_dirs:
        row = _load_seed_row(run_dir)
        if row is None:
            continue
        seed = int(row["seed"])
        if seed in discovered_by_seed:
            other_run = discovered_by_seed[seed]["run_name"]
            raise ValueError(f"Duplicate entries for seed {seed}: {other_run} and {run_dir.name}")
        discovered_by_seed[seed] = row

    if not discovered_by_seed:
        raise ValueError(f"No seeded training runs found in {seed_run_root}")
    return [discovered_by_seed[seed] for seed in sorted(discovered_by_seed)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "seed",
        "run_name",
        "epochs_completed",
        "best_epoch",
        "final_train_loss",
        "final_val_loss",
        "resumed",
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
    loss_decimals: int,
    float_spec: str,
) -> None:
    lines: list[str] = []
    lines.append(r"\begin{table}[" + float_spec + r"]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{tabular}{rrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Seed} & \textbf{Epochs} & \textbf{Best ep.} & "
        r"\textbf{Final train loss} & \textbf{Final val. loss} \\"
    )
    lines.append(r"\midrule")
    for row in rows:
        lines.append(
            "{} & {} & {} & {:.{}f} & {:.{}f} \\\\".format(
                int(row["seed"]),
                int(row["epochs_completed"]),
                int(row["best_epoch"]),
                float(row["final_train_loss"]),
                loss_decimals,
                float(row["final_val_loss"]),
                loss_decimals,
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
        description="Generate the seeded SLM-LSTM training-summary table."
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
        default=Path("outputs/tables/ch4/baseline_slm_seeded_training_summary.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=Path("outputs/tables/ch4/baseline_slm_seeded_training_summary.tex"),
        help="Output LaTeX table path.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Training summary for seeded SLM-LSTM baseline runs. Best ep.\\ denotes "
            "the epoch at which the minimum validation loss was observed."
        ),
        help="LaTeX caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:baseline-slm-seeded-training-summary",
        help="LaTeX label.",
    )
    parser.add_argument(
        "--loss-decimals",
        type=int,
        default=3,
        help="Decimal places for train/validation losses (default: 3).",
    )
    parser.add_argument(
        "--float-spec",
        default="H",
        help="LaTeX table float specifier (default: H).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    seed_run_root = args.seed_run_root if args.seed_run_root.is_absolute() else (repo_root / args.seed_run_root)
    base_run_dir = args.base_run_dir if args.base_run_dir.is_absolute() else (repo_root / args.base_run_dir)
    output_csv = args.output_csv if args.output_csv.is_absolute() else (repo_root / args.output_csv)
    output_tex = args.output_tex if args.output_tex.is_absolute() else (repo_root / args.output_tex)

    rows = _discover_rows(
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
        loss_decimals=args.loss_decimals,
        float_spec=args.float_spec,
    )

    print(f"Wrote CSV: {output_csv}")
    print(f"Wrote TeX: {output_tex}")
    print(f"Seeds: {', '.join(str(int(row['seed'])) for row in rows)}")


if __name__ == "__main__":
    main()
