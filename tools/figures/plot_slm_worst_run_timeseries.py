#!/usr/bin/env python3
"""Plot a representative poor-performance run for SLM-LSTM.

By default this script:
1) reads `eval_metrics_all_runs.json` from the SLM-LSTM baseline run directory,
2) selects the run with the highest RMSE,
3) plots prediction vs ground truth over time.

Legacy `*_deg` prediction fields are treated as radians in this repository
history and converted to degrees for display by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


RAD_TO_DEG = 180.0 / math.pi
DEFAULT_RUN_DIR = Path("outputs/experiments/baseline/baseline_slm_lstm")
FONT_SCALE = 2.0
TITLE_SIZE = int(12 * FONT_SCALE)
LABEL_SIZE = int(11 * FONT_SCALE)
TICK_SIZE = int(10 * FONT_SCALE)
LEGEND_SIZE = int(10 * FONT_SCALE)
DETAIL_SIZE = int(10 * FONT_SCALE)


@dataclass(frozen=True)
class WorstRunInfo:
    run_key: str
    split_role: str
    rmse_rad: float
    n_samples: int | None
    prediction_csv: Path


def _safe_float(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _parse_split_roles(raw: str) -> tuple[str, ...]:
    roles = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not roles:
        raise ValueError("No split roles selected.")
    return roles


def _load_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON list in {metrics_path}")
    return [row for row in rows if isinstance(row, dict)]


def _resolve_prediction_csv(
    repo_root: Path,
    run_dir: Path,
    pred_dir_name: str,
    run_key: str,
    prediction_csv_field: Any,
) -> Path:
    if isinstance(prediction_csv_field, str) and prediction_csv_field.strip():
        candidate = prediction_csv_field.strip()
        csv_path = Path(candidate)
        if not csv_path.is_absolute():
            csv_path = repo_root / csv_path
        if csv_path.exists():
            return csv_path
    fallback = run_dir / pred_dir_name / f"{run_key}.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Prediction CSV not found for run '{run_key}'. "
        f"Checked field path and fallback: {fallback}"
    )


def _find_worst_run(
    repo_root: Path,
    run_dir: Path,
    metrics_filename: str,
    pred_dir_name: str,
    split_roles: tuple[str, ...],
) -> WorstRunInfo:
    metrics_path = run_dir / metrics_filename
    rows = _load_metrics(metrics_path)
    allowed = set(split_roles)

    worst: WorstRunInfo | None = None
    for row in rows:
        split_role = str(row.get("split_role", "")).strip().lower()
        if split_role and split_role not in allowed:
            continue

        run_key = str(row.get("run_key", "")).strip()
        if not run_key:
            continue

        rmse = _safe_float(row.get("rmse"))
        if rmse is None:
            rmse = _safe_float(row.get("rmse_deg"))
        if rmse is None:
            continue

        n_samples_raw = row.get("n_samples")
        n_samples: int | None = None
        if isinstance(n_samples_raw, (int, float)):
            n_samples = int(n_samples_raw)

        pred_csv = _resolve_prediction_csv(
            repo_root=repo_root,
            run_dir=run_dir,
            pred_dir_name=pred_dir_name,
            run_key=run_key,
            prediction_csv_field=row.get("prediction_csv"),
        )

        candidate = WorstRunInfo(
            run_key=run_key,
            split_role=split_role or "unknown",
            rmse_rad=rmse,
            n_samples=n_samples,
            prediction_csv=pred_csv,
        )
        if worst is None or candidate.rmse_rad > worst.rmse_rad:
            worst = candidate

    if worst is None:
        raise ValueError(
            f"No valid runs found in {metrics_path} for split roles: {', '.join(split_roles)}"
        )
    return worst


def _read_prediction_series(csv_path: Path, display_unit: str) -> tuple[list[float], list[float], list[float]]:
    times: list[float] = []
    y_true: list[float] = []
    y_pred: list[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"phi_true_deg", "phi_pred_deg"}
        if not required.issubset(fieldnames):
            raise ValueError(f"CSV missing required columns {sorted(required)}: {csv_path}")
        has_time = "Time" in fieldnames

        for idx, row in enumerate(reader):
            yt = _safe_float(row.get("phi_true_deg"))
            yp = _safe_float(row.get("phi_pred_deg"))
            if yt is None or yp is None:
                continue

            if has_time:
                t = _safe_float(row.get("Time"))
                if t is None:
                    t = float(idx)
            else:
                t = float(idx)

            if display_unit == "deg":
                yt *= RAD_TO_DEG
                yp *= RAD_TO_DEG

            times.append(float(t))
            y_true.append(float(yt))
            y_pred.append(float(yp))

    if not times:
        raise ValueError(f"No usable rows in prediction CSV: {csv_path}")

    return times, y_true, y_pred


def _plot(
    run_info: WorstRunInfo,
    times: list[float],
    y_true: list[float],
    y_pred: list[float],
    display_unit: str,
    output_path: Path,
    show: bool,
) -> None:
    unit_label = "deg" if display_unit == "deg" else "rad"
    rmse_display = run_info.rmse_rad * RAD_TO_DEG if display_unit == "deg" else run_info.rmse_rad

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.plot(times, y_true, color="black", linewidth=2.0, label="Ground truth")
    ax.plot(times, y_pred, color="tab:blue", linewidth=1.6, label="SLM-LSTM prediction")

    ax.set_title(
        f"Representative Poor-Performance Run (SLM-LSTM): {run_info.run_key}",
        fontweight="normal",
        fontsize=TITLE_SIZE,
    )
    ax.set_xlabel("Time [s]", fontsize=LABEL_SIZE)
    ax.set_ylabel(rf"$\phi$ [{unit_label}]", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=LEGEND_SIZE)

    details = (
        f"split_role={run_info.split_role}\n"
        f"rmse_{unit_label}={rmse_display:.2f}\n"
        f"n_samples={run_info.n_samples if run_info.n_samples is not None else 'n/a'}"
    )
    ax.text(
        0.01,
        0.98,
        details,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=DETAIL_SIZE,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    print(f"Worst run selected: {run_info.run_key} (split_role={run_info.split_role}, rmse_rad={run_info.rmse_rad:.6f})")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Plot prediction vs ground truth for the worst-RMSE run in SLM-LSTM "
            "all-runs evaluation artifacts."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help=f"Repository root path (default: {default_repo_root}).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"SLM-LSTM run directory (default: {DEFAULT_RUN_DIR}).",
    )
    parser.add_argument(
        "--metrics-file",
        default="eval_metrics_all_runs.json",
        help="Metrics filename inside --run-dir (default: eval_metrics_all_runs.json).",
    )
    parser.add_argument(
        "--predictions-dir-name",
        default="predictions_all_runs",
        help="Predictions directory name inside --run-dir (default: predictions_all_runs).",
    )
    parser.add_argument(
        "--split-roles",
        default="train,val,eval,unseen",
        help="Comma-separated split roles to consider when selecting worst run.",
    )
    parser.add_argument(
        "--display-unit",
        choices=("deg", "rad"),
        default="deg",
        help="Display unit for y-axis (default: deg).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/ch4/slm_lstm_representative_poor_performance_run.pdf"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figure without opening an interactive window.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir if args.run_dir.is_absolute() else (repo_root / args.run_dir)
    output_path = args.output if args.output.is_absolute() else (repo_root / args.output)
    split_roles = _parse_split_roles(args.split_roles)

    worst = _find_worst_run(
        repo_root=repo_root,
        run_dir=run_dir,
        metrics_filename=args.metrics_file,
        pred_dir_name=args.predictions_dir_name,
        split_roles=split_roles,
    )
    times, y_true, y_pred = _read_prediction_series(worst.prediction_csv, args.display_unit)
    _plot(
        run_info=worst,
        times=times,
        y_true=y_true,
        y_pred=y_pred,
        display_unit=args.display_unit,
        output_path=output_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
