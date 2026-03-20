#!/usr/bin/env python3
"""Generate a compact training-summary table for the 4 baseline models.

The table is built from:
- `training_summary.json` for `epochs_completed` and `best_epoch`
- `history.csv` final row for final train/validation loss
- `resource_usage.csv` final `elapsed_seconds` sample for train time in hours

Note:
- Train time is only as reliable as the final `resource_usage.csv` segment.
- If a run was resumed, earlier monitor segments are not preserved by the
  current training workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


MODEL_RUNS: tuple[tuple[str, str], ...] = (
    ("SLM-LSTM", "outputs/experiments/baseline/baseline_slm_lstm"),
    ("SLU-LSTM", "outputs/experiments/baseline/baseline_slu_lstm"),
    ("TLM-LSTM", "outputs/experiments/baseline/baseline_tlm_lstm"),
    ("TLU-LSTM", "outputs/experiments/baseline/baseline_tlu_lstm"),
)


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


def _build_rows(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_label, run_dir_rel in MODEL_RUNS:
        run_dir = repo_root / run_dir_rel
        summary = _load_json(run_dir / "training_summary.json")
        if not isinstance(summary, dict):
            raise ValueError(f"Expected JSON object in {run_dir / 'training_summary.json'}")

        last_history = _read_last_history_row(run_dir / "history.csv")
        train_time_h = _read_train_time_hours(run_dir / "resource_usage.csv")

        rows.append(
            {
                "model": model_label,
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
        "model",
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
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{Epochs} & \textbf{Best ep.} & "
        r"\textbf{Final train loss} & \textbf{Final val. loss} & \textbf{Train time [h]} \\"
    )
    lines.append(r"\midrule")
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                _latex_escape(str(row["model"])),
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
        description="Generate the 4-model baseline training-summary table."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help=f"Repository root path (default: {default_repo_root}).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/tables/ch4/baseline_training_summary.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=Path("outputs/tables/ch4/baseline_training_summary.tex"),
        help="Output LaTeX table path.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Training summary for the four baseline models. Best ep.\\ denotes the epoch "
            "at which the minimum validation loss was observed. Reported evaluation results "
            "elsewhere in the chapter are based on the weights from this best-validation epoch."
        ),
        help="LaTeX caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:ch4_training_summary",
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
    output_csv = args.output_csv if args.output_csv.is_absolute() else (repo_root / args.output_csv)
    output_tex = args.output_tex if args.output_tex.is_absolute() else (repo_root / args.output_tex)

    rows = _build_rows(repo_root)
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
    print(f"Wrote TeX: {output_tex}")


if __name__ == "__main__":
    main()
