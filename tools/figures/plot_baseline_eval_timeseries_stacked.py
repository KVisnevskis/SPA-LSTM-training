#!/usr/bin/env python3
"""Plot stacked baseline time-series comparisons on evaluation runs.

This helper uses only configured evaluation runs (`eval_metrics.json` + `predictions/`)
from the 4 baseline model directories. Legacy `*_deg` prediction columns are treated as
radians in this repository history and converted to degrees for display by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


MODEL_RUN_DIRS: tuple[tuple[str, str], ...] = (
    ("SLM-LSTM", "outputs/experiments/baseline/baseline_slm_lstm"),
    ("SLU-LSTM", "outputs/experiments/baseline/baseline_slu_lstm"),
    ("TLM-LSTM", "outputs/experiments/baseline/baseline_tlm_lstm"),
    ("TLU-LSTM", "outputs/experiments/baseline/baseline_tlu_lstm"),
)

MODEL_GROUPS: dict[str, tuple[str, ...]] = {
    "slm_slu": ("SLM-LSTM", "SLU-LSTM"),
    "tlm_tlu": ("TLM-LSTM", "TLU-LSTM"),
}

MODEL_COLORS: dict[str, str] = {
    "SLM-LSTM": "tab:blue",
    "SLU-LSTM": "tab:orange",
    "TLM-LSTM": "tab:green",
    "TLU-LSTM": "tab:red",
}

PANEL_KEYS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
RAD_TO_DEG = 180.0 / math.pi


@dataclass(frozen=True)
class RunData:
    run_key: str
    time_s: list[float]
    y_true: list[float]
    y_pred_by_model: dict[str, list[float]]


def _safe_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _read_prediction_csv(path: Path) -> tuple[list[float], list[float], list[float]]:
    time_vals: list[float] = []
    y_true: list[float] = []
    y_pred: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Prediction CSV is empty: {path}")
        required = {"phi_true_deg", "phi_pred_deg"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Missing required columns in {path}: {sorted(required)}")

        has_time = "Time" in reader.fieldnames
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

            time_vals.append(float(t))
            y_true.append(float(yt))
            y_pred.append(float(yp))

    if not time_vals:
        raise ValueError(f"No usable samples in {path}")

    return time_vals, y_true, y_pred


def _load_eval_run_keys(repo_root: Path, ref_run_rel: str) -> list[str]:
    metrics_path = repo_root / ref_run_rel / "eval_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing eval metrics file: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {metrics_path}")

    seen: set[str] = set()
    eval_runs: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("split_role", "")).strip().lower() != "eval":
            continue
        run_key = str(row.get("run_key", "")).strip()
        if not run_key or run_key in seen:
            continue
        seen.add(run_key)
        eval_runs.append(run_key)

    if len(eval_runs) < 1:
        raise ValueError(f"No eval runs found in {metrics_path}")
    return eval_runs


def _run_duration_seconds(repo_root: Path, ref_run_rel: str, run_key: str) -> float:
    path = repo_root / ref_run_rel / "predictions" / f"{run_key}.csv"
    times, _, _ = _read_prediction_csv(path)
    return times[-1] - times[0]


def _select_six_runs(repo_root: Path, eval_run_keys: list[str], ref_run_rel: str) -> list[str]:
    ranked = sorted(
        eval_run_keys,
        key=lambda rk: _run_duration_seconds(repo_root=repo_root, ref_run_rel=ref_run_rel, run_key=rk),
        reverse=True,
    )
    if len(ranked) < 6:
        raise ValueError(
            f"Expected at least 6 eval runs for stacked plot layout; found {len(ranked)}."
        )
    # Use the two longest runs on top rows and next four below (deterministic).
    return ranked[:6]


def _convert(values: list[float], display_unit: str) -> list[float]:
    if display_unit == "deg":
        return [v * RAD_TO_DEG for v in values]
    return values


def _load_run_data(repo_root: Path, run_key: str, display_unit: str) -> RunData:
    return _load_run_data_for_models(
        repo_root=repo_root,
        run_key=run_key,
        display_unit=display_unit,
        selected_model_labels=tuple(label for label, _ in MODEL_RUN_DIRS),
    )


def _load_run_data_for_models(
    repo_root: Path,
    run_key: str,
    display_unit: str,
    selected_model_labels: tuple[str, ...],
) -> RunData:
    y_pred_by_model: dict[str, list[float]] = {}
    ref_time: list[float] | None = None
    ref_true: list[float] | None = None

    run_rel_by_label = dict(MODEL_RUN_DIRS)
    for model_label in selected_model_labels:
        run_rel = run_rel_by_label[model_label]
        pred_path = repo_root / run_rel / "predictions" / f"{run_key}.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {pred_path}")

        time_vals, y_true, y_pred = _read_prediction_csv(pred_path)
        if ref_time is None or ref_true is None:
            ref_time = time_vals
            ref_true = y_true
        else:
            n = min(len(ref_time), len(time_vals), len(ref_true), len(y_pred))
            ref_time = ref_time[:n]
            ref_true = ref_true[:n]
            y_pred = y_pred[:n]
        y_pred_by_model[model_label] = _convert(y_pred, display_unit)

    if ref_time is None or ref_true is None:
        raise RuntimeError(f"Failed to load run data for {run_key}")

    return RunData(
        run_key=run_key,
        time_s=ref_time,
        y_true=_convert(ref_true, display_unit),
        y_pred_by_model=y_pred_by_model,
    )


def _pretty_run_name(run_key: str) -> str:
    if run_key.startswith("run_"):
        return run_key.replace("_", " ")
    return run_key


def _rmse(y_true: list[float], y_pred: list[float]) -> float:
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return float("nan")
    sq_err_sum = 0.0
    for yt, yp in zip(y_true[:n], y_pred[:n]):
        err = yp - yt
        sq_err_sum += err * err
    return math.sqrt(sq_err_sum / n)


def _plot_stacked(
    run_data_list: list[RunData],
    output_path: Path,
    display_unit: str,
    selected_model_labels: tuple[str, ...],
    show: bool,
) -> None:
    mosaic = [
        ["A", "A"],
        ["B", "B"],
        ["C", "D"],
        ["E", "F"],
    ]
    # Increase vertical footprint by ~30% relative to previous 14x10 layout.
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(14, 13), constrained_layout=True)

    y_label = r"$\phi$ [$^\circ$]" if display_unit == "deg" else r"$\phi$ [rad]"
    rmse_unit = "deg" if display_unit == "deg" else "rad"
    line_handles = {}

    for panel_key, run_data in zip(PANEL_KEYS, run_data_list):
        ax = axes[panel_key]
        gt_handle = ax.plot(
            run_data.time_s,
            run_data.y_true,
            color="black",
            linewidth=2.2,
            label="Ground Truth",
        )[0]
        if "Ground Truth" not in line_handles:
            line_handles["Ground Truth"] = gt_handle

        for model_label in selected_model_labels:
            handle = ax.plot(
                run_data.time_s,
                run_data.y_pred_by_model[model_label],
                color=MODEL_COLORS[model_label],
                linewidth=1.5,
                label=model_label,
            )[0]
            if model_label not in line_handles:
                line_handles[model_label] = handle

        ax.text(
            0.01,
            0.9,
            panel_key,
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold",
            va="top",
        )
        ax.text(
            0.07,
            0.9,
            _pretty_run_name(run_data.run_key),
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            alpha=0.9,
        )
        for idx, model_label in enumerate(selected_model_labels):
            rmse_val = _rmse(run_data.y_true, run_data.y_pred_by_model[model_label])
            ax.text(
                0.07,
                0.84 - (idx * 0.055),
                f"{model_label} RMSE: {rmse_val:.3f} {rmse_unit}",
                transform=ax.transAxes,
                fontsize=9,
                color=MODEL_COLORS[model_label],
                va="top",
                alpha=0.95,
            )
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(run_data.time_s[0], run_data.time_s[-1])

    axes["B"].set_xlabel("Time [s]")
    axes["E"].set_xlabel("Time [s]")
    axes["F"].set_xlabel("Time [s]")

    legend_order = ["Ground Truth"] + list(selected_model_labels)
    axes["D"].legend(
        [line_handles[k] for k in legend_order],
        legend_order,
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )

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
            "Stacked multi-panel time-series comparison for eval runs. "
            "By default this script writes two figures: "
            "SLM+SLU+GT and TLM+TLU+GT."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help=f"Repository root path (default: {default_repo_root})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/baseline_eval_timeseries_stacked.pdf"),
        help=(
            "Base output path. If one group is selected, writes exactly this file. "
            "If multiple groups are selected, appends '_<group>' before suffix."
        ),
    )
    parser.add_argument(
        "--groups",
        default="slm_slu,tlm_tlu",
        help=(
            "Comma-separated model groups to plot. Allowed values: "
            "slm_slu, tlm_tlu (default: slm_slu,tlm_tlu)."
        ),
    )
    parser.add_argument(
        "--display-unit",
        choices=("deg", "rad"),
        default="deg",
        help=(
            "Display unit for y-axis. Input prediction values are treated as radians due to "
            "legacy `*_deg` naming."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Render and save without opening an interactive window.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    base_output_path = args.output if args.output.is_absolute() else (repo_root / args.output)

    requested_groups = tuple(
        grp.strip().lower()
        for grp in args.groups.split(",")
        if grp.strip()
    )
    if not requested_groups:
        raise ValueError("No groups selected. Use --groups slm_slu,tlm_tlu")
    unknown = [grp for grp in requested_groups if grp not in MODEL_GROUPS]
    if unknown:
        raise ValueError(f"Unknown groups {unknown}. Allowed: {sorted(MODEL_GROUPS)}")

    ref_run_rel = MODEL_RUN_DIRS[0][1]
    eval_run_keys = _load_eval_run_keys(repo_root=repo_root, ref_run_rel=ref_run_rel)
    selected_runs = _select_six_runs(
        repo_root=repo_root,
        eval_run_keys=eval_run_keys,
        ref_run_rel=ref_run_rel,
    )
    for idx, group_key in enumerate(requested_groups):
        model_labels = MODEL_GROUPS[group_key]
        run_data_list = [
            _load_run_data_for_models(
                repo_root=repo_root,
                run_key=run_key,
                display_unit=args.display_unit,
                selected_model_labels=model_labels,
            )
            for run_key in selected_runs
        ]

        if len(requested_groups) == 1:
            output_path = base_output_path
        else:
            suffix = base_output_path.suffix or ".pdf"
            stem = base_output_path.stem if base_output_path.suffix else base_output_path.name
            output_path = base_output_path.with_name(f"{stem}_{group_key}{suffix}")

        _plot_stacked(
            run_data_list=run_data_list,
            output_path=output_path,
            display_unit=args.display_unit,
            selected_model_labels=model_labels,
            show=(not args.no_show) and (idx == len(requested_groups) - 1),
        )


if __name__ == "__main__":
    main()
