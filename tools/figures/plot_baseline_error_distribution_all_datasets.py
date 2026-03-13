#!/usr/bin/env python3
"""Plot baseline error distributions across datasets.

This helper uses `eval_metrics_all_runs.json` and `predictions_all_runs/*.csv`
for the four baseline models. Legacy `*_deg` prediction fields are treated as
radians in this repository history and converted to degrees by default.
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


MODEL_RUN_DIRS: tuple[tuple[str, str], ...] = (
    ("SLM-LSTM", "outputs/experiments/baseline/baseline_slm_lstm"),
    ("SLU-LSTM", "outputs/experiments/baseline/baseline_slu_lstm"),
    ("TLM-LSTM", "outputs/experiments/baseline/baseline_tlm_lstm"),
    ("TLU-LSTM", "outputs/experiments/baseline/baseline_tlu_lstm"),
)

MODEL_COLORS: dict[str, str] = {
    "SLM-LSTM": "tab:blue",
    "SLU-LSTM": "tab:orange",
    "TLM-LSTM": "tab:green",
    "TLU-LSTM": "tab:red",
}

VALID_SPLIT_ROLES = {"train", "val", "eval", "unseen"}
RAD_TO_DEG = 180.0 / math.pi
FONT_SCALE = 2.0
TITLE_SIZE = int(11 * FONT_SCALE)
LABEL_SIZE = int(10 * FONT_SCALE)
TICK_SIZE = int(10 * FONT_SCALE)


@dataclass(frozen=True)
class ErrorSeries:
    model_label: str
    run_count: int
    sample_count: int
    errors: list[float]


def _safe_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    rows = [row for row in data if isinstance(row, dict)]
    return rows


def _parse_split_roles(raw_roles: str) -> tuple[str, ...]:
    roles = tuple(
        role.strip().lower()
        for role in raw_roles.split(",")
        if role.strip()
    )
    if not roles:
        raise ValueError("No split roles selected.")
    invalid = sorted(set(roles) - VALID_SPLIT_ROLES)
    if invalid:
        raise ValueError(
            f"Invalid split roles: {invalid}. Allowed: {sorted(VALID_SPLIT_ROLES)}"
        )
    return roles


def _run_keys_from_metrics(
    metrics_rows: list[dict[str, Any]],
    split_roles: tuple[str, ...],
) -> list[str]:
    selected = set(split_roles)
    seen: set[str] = set()
    run_keys: list[str] = []
    for row in metrics_rows:
        split_role = str(row.get("split_role", "")).strip().lower()
        if split_role not in selected:
            continue
        run_key = str(row.get("run_key", "")).strip()
        if not run_key or run_key in seen:
            continue
        seen.add(run_key)
        run_keys.append(run_key)
    return run_keys


def _read_prediction_errors(
    csv_path: Path,
    display_unit: str,
) -> list[float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {csv_path}")

    errors: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"phi_true_deg", "phi_pred_deg"}
        if not required.issubset(fieldnames):
            raise ValueError(f"CSV {csv_path} missing required columns {sorted(required)}")

        for row in reader:
            y_true = _safe_float(row.get("phi_true_deg"))
            y_pred = _safe_float(row.get("phi_pred_deg"))
            if y_true is None or y_pred is None:
                continue
            err = y_pred - y_true  # stored in radians despite legacy *_deg naming
            if display_unit == "deg":
                err *= RAD_TO_DEG
            errors.append(err)

    return errors


def _load_model_errors(
    repo_root: Path,
    model_label: str,
    run_dir_rel: str,
    split_roles: tuple[str, ...],
    display_unit: str,
) -> ErrorSeries:
    run_dir = repo_root / run_dir_rel
    metrics_rows = _load_json_list(run_dir / "eval_metrics_all_runs.json")
    run_keys = _run_keys_from_metrics(metrics_rows=metrics_rows, split_roles=split_roles)
    if not run_keys:
        raise ValueError(f"No runs matched split roles {split_roles} for {model_label}")

    all_errors: list[float] = []
    for run_key in run_keys:
        pred_path = run_dir / "predictions_all_runs" / f"{run_key}.csv"
        all_errors.extend(_read_prediction_errors(pred_path, display_unit))

    if not all_errors:
        raise ValueError(f"No prediction samples loaded for {model_label}")

    return ErrorSeries(
        model_label=model_label,
        run_count=len(run_keys),
        sample_count=len(all_errors),
        errors=all_errors,
    )


def _global_error_range(series: list[ErrorSeries]) -> tuple[float, float]:
    mins = [min(s.errors) for s in series]
    maxs = [max(s.errors) for s in series]
    x_min = min(mins)
    x_max = max(maxs)
    if math.isclose(x_min, x_max):
        pad = 1.0 if x_min == 0 else abs(x_min) * 0.1
        x_min -= pad
        x_max += pad
    return x_min, x_max


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    sorted_vals = sorted(values)
    pos = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _robust_error_range(
    series: list[ErrorSeries],
    lower_pct: float,
    upper_pct: float,
) -> tuple[float, float]:
    all_errors: list[float] = []
    for s in series:
        all_errors.extend(s.errors)
    x_min = _percentile(all_errors, lower_pct)
    x_max = _percentile(all_errors, upper_pct)
    if not math.isfinite(x_min) or not math.isfinite(x_max) or math.isclose(x_min, x_max):
        return _global_error_range(series)
    return x_min, x_max


def _plot_histograms(
    series_by_model: list[ErrorSeries],
    bins: int,
    display_unit: str,
    robust_range: bool,
    lower_pct: float,
    upper_pct: float,
    output_path: Path,
    show: bool,
) -> None:
    unit_label = "deg" if display_unit == "deg" else "rad"
    if robust_range:
        x_min, x_max = _robust_error_range(series_by_model, lower_pct=lower_pct, upper_pct=upper_pct)
    else:
        x_min, x_max = _global_error_range(series_by_model)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, series in zip(axes.flatten(), series_by_model):
        ax.hist(
            series.errors,
            bins=bins,
            range=(x_min, x_max),
            color=MODEL_COLORS[series.model_label],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.25,
        )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(
            f"{series.model_label} (runs={series.run_count}, n={series.sample_count:,})",
            fontweight="normal",
            fontsize=TITLE_SIZE,
        )
        ax.set_xlabel(f"Prediction error [{unit_label}]", fontsize=LABEL_SIZE)
        ax.set_ylabel("Count", fontsize=LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        ax.grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved figure: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Create baseline error-distribution histograms across datasets using "
            "all-runs evaluation artifacts."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help=f"Repository root (default: {default_repo_root}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/baseline_error_distribution_all_datasets.pdf"),
        help="Output figure path (default: outputs/figures/baseline_error_distribution_all_datasets.pdf).",
    )
    parser.add_argument(
        "--split-roles",
        default="eval,unseen",
        help="Comma-separated split roles to include (default: eval,unseen).",
    )
    parser.add_argument(
        "--display-unit",
        choices=("deg", "rad"),
        default="deg",
        help="Display unit for error axis (default: deg).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=120,
        help="Histogram bins per subplot (default: 120).",
    )
    parser.add_argument(
        "--full-range",
        action="store_true",
        help=(
            "Use full min/max error range (includes all outliers). "
            "Default uses robust percentile range for readability."
        ),
    )
    parser.add_argument(
        "--clip-lower-pct",
        type=float,
        default=0.5,
        help="Lower percentile for robust x-range (default: 0.5).",
    )
    parser.add_argument(
        "--clip-upper-pct",
        type=float,
        default=99.5,
        help="Upper percentile for robust x-range (default: 99.5).",
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
    output_path = args.output if args.output.is_absolute() else (repo_root / args.output)
    split_roles = _parse_split_roles(args.split_roles)
    if not (0.0 <= args.clip_lower_pct < args.clip_upper_pct <= 100.0):
        raise ValueError("--clip-lower-pct and --clip-upper-pct must satisfy 0 <= lower < upper <= 100.")

    series_by_model = [
        _load_model_errors(
            repo_root=repo_root,
            model_label=model_label,
            run_dir_rel=run_dir_rel,
            split_roles=split_roles,
            display_unit=args.display_unit,
        )
        for model_label, run_dir_rel in MODEL_RUN_DIRS
    ]
    _plot_histograms(
        series_by_model=series_by_model,
        bins=args.bins,
        display_unit=args.display_unit,
        robust_range=(not args.full_range),
        lower_pct=args.clip_lower_pct,
        upper_pct=args.clip_upper_pct,
        output_path=output_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
