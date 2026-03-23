#!/usr/bin/env python3
"""Generate a compact training-summary table for the SLM width-ablation runs.

The width-ablation study is defined by experiment configs under
`configs/experiments/hpo/width_ablation`. For each matching config, this helper
reads the corresponding run directory under `outputs/experiments/hpo/width_ablation`
and builds a table from:
- `training_summary.json` for `epochs_completed` and `best_epoch`
- `history.csv` final row for final train/validation loss
- `resource_usage.csv` final `elapsed_seconds` sample for train time in hours
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


CONFIG_NAME_RE = re.compile(r"^slm_width_ablation__baseline__u(?P<width>\d+)__seed(?P<seed>\d+)$")


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


def _read_train_time_hours(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    raw_elapsed = rows[-1].get("elapsed_seconds")
    if raw_elapsed is None:
        return None
    try:
        return float(raw_elapsed) / 3600.0
    except ValueError:
        return None


def _fmt(value: float | None, decimals: int) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def _discover_width_runs(config_dir: Path, run_root: Path) -> list[tuple[int, str, Path]]:
    discovered: list[tuple[int, str, Path]] = []
    for config_path in sorted(config_dir.glob("slm_width_ablation__baseline__u*__seed*.yaml")):
        match = CONFIG_NAME_RE.match(config_path.stem)
        if match is None:
            continue
        width = int(match.group("width"))
        run_name = config_path.stem
        run_dir = run_root / run_name
        if not run_dir.exists():
            raise FileNotFoundError(
                f"Expected run directory for config {config_path} at {run_dir}, but it does not exist."
            )
        discovered.append((width, run_name, run_dir))

    if not discovered:
        raise ValueError(f"No matching width-ablation configs found in {config_dir}")
    return sorted(discovered, key=lambda item: item[0])


def _build_rows(config_dir: Path, run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for width, run_name, run_dir in _discover_width_runs(config_dir=config_dir, run_root=run_root):
        summary = _load_json(run_dir / "training_summary.json")
        if not isinstance(summary, dict):
            raise ValueError(f"Expected JSON object in {run_dir / 'training_summary.json'}")

        last_history = _read_last_history_row(run_dir / "history.csv")
        train_time_h = _read_train_time_hours(run_dir / "resource_usage.csv")

        rows.append(
            {
                "width": width,
                "run_name": run_name,
                "epochs_completed": int(summary["epochs_completed"]),
                "best_epoch": int(summary["best_epoch"]),
                "final_train_loss": float(last_history["train_loss_mean"]),
                "final_val_loss": float(last_history["val_loss_mean"]),
                "train_time_h": train_time_h,
                "resumed": bool(summary.get("resumed_from_checkpoint", False)),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "width",
        "run_name",
        "epochs_completed",
        "best_epoch",
        "final_train_loss",
        "final_val_loss",
        "train_time_h",
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
    time_decimals: int,
    float_spec: str,
) -> None:
    lines: list[str] = []
    lines.append(r"\begin{table}[" + float_spec + r"]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{tabular}{rrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Width} & \textbf{Epochs} & \textbf{Best ep.} & "
        r"\textbf{Final train loss} & \textbf{Final val. loss} & \textbf{Train time [h]} \\"
    )
    lines.append(r"\midrule")
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                int(row["width"]),
                int(row["epochs_completed"]),
                int(row["best_epoch"]),
                _fmt(float(row["final_train_loss"]), loss_decimals),
                _fmt(float(row["final_val_loss"]), loss_decimals),
                _fmt(row["train_time_h"], time_decimals),
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
        description="Generate the SLM width-ablation training-summary table."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help=f"Repository root path (default: {default_repo_root}).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/experiments/hpo/width_ablation"),
        help="Directory containing the width-ablation experiment configs.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("outputs/experiments/hpo/width_ablation"),
        help="Directory containing the width-ablation run directories.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/tables/ch4/width_ablation_training_summary.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=Path("outputs/tables/ch4/width_ablation_training_summary.tex"),
        help="Output LaTeX table path.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Training summary for the SLM-LSTM width-ablation runs. Width denotes the number "
            "of hidden units in the single LSTM layer. Best ep.\\ denotes the epoch at which "
            "the minimum validation loss was observed."
        ),
        help="LaTeX caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:width-ablation-training-summary",
        help="LaTeX label.",
    )
    parser.add_argument(
        "--loss-decimals",
        type=int,
        default=3,
        help="Decimal places for train/validation losses (default: 3).",
    )
    parser.add_argument(
        "--time-decimals",
        type=int,
        default=3,
        help="Decimal places for train time in hours (default: 3).",
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
    config_dir = args.config_dir if args.config_dir.is_absolute() else (repo_root / args.config_dir)
    run_root = args.run_root if args.run_root.is_absolute() else (repo_root / args.run_root)
    output_csv = args.output_csv if args.output_csv.is_absolute() else (repo_root / args.output_csv)
    output_tex = args.output_tex if args.output_tex.is_absolute() else (repo_root / args.output_tex)

    rows = _build_rows(config_dir=config_dir, run_root=run_root)
    _write_csv(output_csv, rows)
    _write_tex(
        path=output_tex,
        rows=rows,
        caption=args.caption,
        label=args.label,
        loss_decimals=args.loss_decimals,
        time_decimals=args.time_decimals,
        float_spec=args.float_spec,
    )
    print(f"Wrote CSV: {output_csv}")
    print(f"Wrote LaTeX: {output_tex}")
    print(f"Rows: {len(rows)}")
    resumed_widths = [str(int(row["width"])) for row in rows if bool(row.get("resumed"))]
    if resumed_widths:
        print(
            "Warning: the following widths were resumed and may have partial train-time values: "
            + ", ".join(resumed_widths)
        )


if __name__ == "__main__":
    main()
