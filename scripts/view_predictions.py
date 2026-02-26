#!/usr/bin/env python3
"""Desktop viewer for model prediction CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import ttk


PRED_TRUE_COL = "phi_true_deg"
PRED_PRED_COL = "phi_pred_deg"
PRED_TIME_COL = "Time"
MAX_PLOT_POINTS = 2400


@dataclass(frozen=True)
class TrialStats:
    n_samples: int
    rmse_deg: float
    mae_deg: float
    max_abs_err_deg: float
    bias_deg: float


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


def _fmt(value: Any, digits: int = 6) -> str:
    num = _safe_float(None if value is None else str(value))
    if num is None:
        return "n/a"
    return f"{num:.{digits}f}"


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _discover_runs(root: Path) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    if not root.exists():
        return runs
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        pred_dir = entry / "predictions"
        if pred_dir.is_dir() and any(pred_dir.glob("*.csv")):
            runs[entry.name] = entry
    return runs


def _trial_names(run_dir: Path) -> list[str]:
    pred_dir = run_dir / "predictions"
    if not pred_dir.exists():
        return []
    return sorted(path.stem for path in pred_dir.glob("*.csv"))


def _read_history_row(path: Path, best_epoch: int | None) -> dict[str, float | int] | None:
    if best_epoch is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = _safe_float(row.get("epoch"))
            if epoch is None or int(epoch) != best_epoch:
                continue
            out: dict[str, float | int] = {"epoch": best_epoch}
            for key in ("train_loss_mean", "val_loss_mean", "val_rmse_mean", "val_mae_mean"):
                value = _safe_float(row.get(key))
                if value is not None:
                    out[key] = value
            return out
    return None


def _weighted_eval_summary(eval_metrics: Any) -> dict[str, float | int] | None:
    if not isinstance(eval_metrics, list):
        return None

    n_trials = 0
    n_samples_total = 0
    mse_sum = 0.0
    mae_sum = 0.0
    for row in eval_metrics:
        if not isinstance(row, dict):
            continue
        n = _safe_float(str(row.get("n_samples", "")))
        rmse = _safe_float(str(row.get("rmse_deg", "")))
        mae = _safe_float(str(row.get("mae_deg", "")))
        if n is None or rmse is None or mae is None or n <= 0:
            continue
        n_int = int(n)
        n_trials += 1
        n_samples_total += n_int
        mse_sum += (rmse * rmse) * n_int
        mae_sum += mae * n_int

    if n_trials == 0 or n_samples_total == 0:
        return None
    return {
        "n_trials": n_trials,
        "n_samples": n_samples_total,
        "weighted_rmse_deg": math.sqrt(mse_sum / n_samples_total),
        "weighted_mae_deg": mae_sum / n_samples_total,
    }


def _run_metadata(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "run_manifest.json")
    training_summary = _read_json(run_dir / "training_summary.json")
    config_snapshot = _read_json(run_dir / "config_snapshot.json")
    eval_metrics = _read_json(run_dir / "eval_metrics.json")

    manifest = manifest if isinstance(manifest, dict) else {}
    training_summary = training_summary if isinstance(training_summary, dict) else {}
    config_snapshot = config_snapshot if isinstance(config_snapshot, dict) else {}

    model_cfg = config_snapshot.get("model", {}) if isinstance(config_snapshot.get("model"), dict) else {}
    training_cfg = config_snapshot.get("training", {}) if isinstance(config_snapshot.get("training"), dict) else {}
    data_cfg = config_snapshot.get("data", {}) if isinstance(config_snapshot.get("data"), dict) else {}

    raw_best_epoch = training_summary.get("best_epoch")
    best_epoch = int(raw_best_epoch) if isinstance(raw_best_epoch, (int, float)) else None

    return {
        "eval_metrics": eval_metrics,
        "best_epoch": best_epoch,
        "best_val_loss": training_summary.get("best_val_loss"),
        "epochs_completed": training_summary.get("epochs_completed"),
        "stopped_early": training_summary.get("stopped_early"),
        "best_history_row": _read_history_row(run_dir / "history.csv", best_epoch),
        "model_variant": model_cfg.get("variant", manifest.get("model_variant")),
        "learning_rate": model_cfg.get("learning_rate"),
        "features": data_cfg.get("features", manifest.get("features", [])),
        "target": data_cfg.get("target", manifest.get("target")),
        "batch_size": training_cfg.get("batch_size"),
        "stateful": training_cfg.get("stateful"),
        "patience": training_cfg.get("patience"),
        "seed": training_cfg.get("seed"),
        "overall_eval": _weighted_eval_summary(eval_metrics),
    }


def _read_prediction_csv(path: Path) -> tuple[str, list[float], list[float], list[float]]:
    x_vals: list[float] = []
    y_true: list[float] = []
    y_pred: list[float] = []
    x_label = PRED_TIME_COL

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Prediction CSV is empty: {path}")
        if PRED_TRUE_COL not in reader.fieldnames or PRED_PRED_COL not in reader.fieldnames:
            raise ValueError(
                f"CSV {path} must include '{PRED_TRUE_COL}' and '{PRED_PRED_COL}'. "
                f"Found {reader.fieldnames}"
            )

        has_time = PRED_TIME_COL in reader.fieldnames
        if not has_time:
            x_label = "index"

        for index, row in enumerate(reader):
            yt = _safe_float(row.get(PRED_TRUE_COL))
            yp = _safe_float(row.get(PRED_PRED_COL))
            if yt is None or yp is None:
                continue

            if has_time:
                xv = _safe_float(row.get(PRED_TIME_COL))
                if xv is None:
                    xv = float(index)
                    x_label = "index"
            else:
                xv = float(index)

            x_vals.append(float(xv))
            y_true.append(float(yt))
            y_pred.append(float(yp))

    return x_label, x_vals, y_true, y_pred


def _compute_trial_stats(y_true: list[float], y_pred: list[float]) -> TrialStats:
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return TrialStats(0, float("nan"), float("nan"), float("nan"), float("nan"))

    sq_sum = 0.0
    abs_sum = 0.0
    bias_sum = 0.0
    max_abs = 0.0
    for yt, yp in zip(y_true[:n], y_pred[:n]):
        err = yp - yt
        sq_sum += err * err
        abs_err = abs(err)
        abs_sum += abs_err
        bias_sum += err
        if abs_err > max_abs:
            max_abs = abs_err

    return TrialStats(
        n_samples=n,
        rmse_deg=math.sqrt(sq_sum / n),
        mae_deg=abs_sum / n,
        max_abs_err_deg=max_abs,
        bias_deg=bias_sum / n,
    )


def _find_eval_record(eval_metrics: Any, trial_name: str) -> dict[str, Any] | None:
    if not isinstance(eval_metrics, list):
        return None
    for row in eval_metrics:
        if isinstance(row, dict) and str(row.get("run_key")) == trial_name:
            return row
    return None


class PredictionViewer:
    def __init__(self, root_tk: tk.Tk, experiments_root: Path) -> None:
        self.root = root_tk
        self.experiments_root = experiments_root
        self.runs = _discover_runs(experiments_root)
        self.current_run: Path | None = None
        self.current_trial_name: str | None = None
        self.current_meta: dict[str, Any] | None = None
        self.current_trial_data: tuple[str, list[float], list[float], list[float]] | None = None

        self.model_var = tk.StringVar()
        self.trial_var = tk.StringVar()
        self.rmse_var = tk.StringVar(value="-")
        self.mae_var = tk.StringVar(value="-")
        self.max_err_var = tk.StringVar(value="-")
        self.bias_var = tk.StringVar(value="-")
        self.best_epoch_var = tk.StringVar(value="-")
        self.best_val_var = tk.StringVar(value="-")
        self.status_var = tk.StringVar(value="")

        self.model_combo: ttk.Combobox
        self.trial_combo: ttk.Combobox
        self.canvas: tk.Canvas
        self.info_text: tk.Text

        self._build_ui()
        self._populate_models()

    def _build_ui(self) -> None:
        self.root.title("SPA-LSTM Prediction Viewer")
        self.root.geometry("1400x860")
        self.root.minsize(1120, 720)

        container = ttk.Frame(self.root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(container)
        controls.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(controls, text="Model run").pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(controls, textvariable=self.model_var, state="readonly", width=38)
        self.model_combo.pack(side=tk.LEFT, padx=(8, 18))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        ttk.Label(controls, text="Trial").pack(side=tk.LEFT)
        self.trial_combo = ttk.Combobox(controls, textvariable=self.trial_var, state="readonly", width=38)
        self.trial_combo.pack(side=tk.LEFT, padx=(8, 0))
        self.trial_combo.bind("<<ComboboxSelected>>", self._on_trial_change)

        main = ttk.Frame(container)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(main, text="Predicted vs Ground Truth")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.canvas = tk.Canvas(left, background="#ffffff", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        right = ttk.LabelFrame(main, text="Metrics and Best Params")
        right.pack(side=tk.RIGHT, fill=tk.Y)

        metrics_grid = ttk.Frame(right, padding=(8, 8, 8, 4))
        metrics_grid.pack(fill=tk.X)

        self._metric_row(metrics_grid, 0, "Trial RMSE (deg)", self.rmse_var)
        self._metric_row(metrics_grid, 1, "Trial MAE (deg)", self.mae_var)
        self._metric_row(metrics_grid, 2, "Max |error| (deg)", self.max_err_var)
        self._metric_row(metrics_grid, 3, "Bias (deg)", self.bias_var)
        self._metric_row(metrics_grid, 4, "Best epoch", self.best_epoch_var)
        self._metric_row(metrics_grid, 5, "Best val loss", self.best_val_var)

        self.info_text = tk.Text(
            right,
            width=56,
            wrap=tk.WORD,
            padx=8,
            pady=8,
            state=tk.DISABLED,
        )
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 6))

        status = ttk.Label(right, textvariable=self.status_var, foreground="#4b5563")
        status.pack(fill=tk.X, padx=10, pady=(0, 8))

    def _metric_row(self, parent: ttk.Frame, row: int, label: str, value_var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=2)
        ttk.Label(parent, textvariable=value_var).grid(row=row, column=1, sticky="e", pady=2)

    def _populate_models(self) -> None:
        model_names = sorted(self.runs)
        self.model_combo["values"] = model_names

        if not model_names:
            self._set_info(
                f"No run directories with prediction CSVs found in:\n{self.experiments_root}\n\n"
                "Expected: <run_dir>/predictions/*.csv"
            )
            self._draw_message("No run data found")
            return

        first = model_names[0]
        self.model_var.set(first)
        self._load_run(first)

    def _load_run(self, run_name: str) -> None:
        run_dir = self.runs.get(run_name)
        if run_dir is None:
            return

        self.current_run = run_dir
        self.current_meta = _run_metadata(run_dir)
        trial_names = _trial_names(run_dir)
        self.trial_combo["values"] = trial_names

        if not trial_names:
            self.current_trial_data = None
            self._set_info(f"Run {run_name} has no trial CSV files.")
            self._draw_message("No trials in selected run")
            return

        first = trial_names[0]
        self.trial_var.set(first)
        self._load_trial(first)

    def _load_trial(self, trial_name: str) -> None:
        if self.current_run is None:
            return

        trial_path = self.current_run / "predictions" / f"{trial_name}.csv"
        if not trial_path.exists():
            self.current_trial_data = None
            self._set_info(f"Missing prediction file:\n{trial_path}")
            self._draw_message("Missing trial file")
            return

        try:
            self.current_trial_data = _read_prediction_csv(trial_path)
        except Exception as exc:
            self.current_trial_data = None
            self._set_info(f"Failed to read trial CSV:\n{exc}")
            self._draw_message("CSV parse error")
            return

        self.current_trial_name = trial_name
        self._refresh_trial_view()

    def _refresh_trial_view(self) -> None:
        if self.current_run is None or self.current_trial_data is None:
            return

        x_label, x_vals, y_true, y_pred = self.current_trial_data
        stats = _compute_trial_stats(y_true, y_pred)
        eval_record = _find_eval_record(
            None if self.current_meta is None else self.current_meta.get("eval_metrics"),
            self.current_trial_name or "",
        )

        self.rmse_var.set(_fmt(stats.rmse_deg, 4))
        self.mae_var.set(_fmt(stats.mae_deg, 4))
        self.max_err_var.set(_fmt(stats.max_abs_err_deg, 4))
        self.bias_var.set(_fmt(stats.bias_deg, 4))

        if self.current_meta is None:
            self.best_epoch_var.set("n/a")
            self.best_val_var.set("n/a")
        else:
            self.best_epoch_var.set(str(self.current_meta.get("best_epoch", "n/a")))
            self.best_val_var.set(_fmt(self.current_meta.get("best_val_loss"), 4))

        self._draw_plot(x_label, x_vals, y_true, y_pred, self.current_trial_name or "")
        self._set_info(self._build_detail_text(stats, eval_record))
        self.status_var.set(
            f"run={self.current_run.name} | trial={self.current_trial_name} | samples={stats.n_samples}"
        )

    def _build_detail_text(self, stats: TrialStats, eval_record: dict[str, Any] | None) -> str:
        run_name = self.current_run.name if self.current_run else "n/a"
        trial = self.current_trial_name or "n/a"
        meta = self.current_meta or {}

        lines: list[str] = []
        lines.append("Selected Trial")
        lines.append(f"- run: {run_name}")
        lines.append(f"- trial: {trial}")
        lines.append(f"- samples: {stats.n_samples}")
        lines.append(f"- rmse_deg (recomputed): {_fmt(stats.rmse_deg)}")
        lines.append(f"- mae_deg (recomputed): {_fmt(stats.mae_deg)}")
        lines.append(f"- max_abs_err_deg: {_fmt(stats.max_abs_err_deg)}")
        lines.append(f"- bias_deg: {_fmt(stats.bias_deg)}")
        lines.append("")

        lines.append("eval_metrics.json record")
        if eval_record is None:
            lines.append("- matching record: n/a")
        else:
            lines.append(f"- rmse_deg: {_fmt(eval_record.get('rmse_deg'))}")
            lines.append(f"- mae_deg: {_fmt(eval_record.get('mae_deg'))}")
            lines.append(f"- n_samples: {eval_record.get('n_samples', 'n/a')}")
        lines.append("")

        lines.append("Best Model Summary")
        lines.append(f"- best_epoch: {meta.get('best_epoch', 'n/a')}")
        lines.append(f"- best_val_loss: {_fmt(meta.get('best_val_loss'))}")
        lines.append(f"- epochs_completed: {meta.get('epochs_completed', 'n/a')}")
        lines.append(f"- stopped_early: {meta.get('stopped_early', 'n/a')}")
        best_history = meta.get("best_history_row")
        if isinstance(best_history, dict):
            lines.append(
                "- best_epoch_metrics: "
                f"train_loss={_fmt(best_history.get('train_loss_mean'))}, "
                f"val_loss={_fmt(best_history.get('val_loss_mean'))}, "
                f"val_rmse={_fmt(best_history.get('val_rmse_mean'))}, "
                f"val_mae={_fmt(best_history.get('val_mae_mean'))}"
            )
        lines.append("")

        lines.append("Model / Config Params")
        lines.append(f"- model_variant: {meta.get('model_variant', 'n/a')}")
        lines.append(f"- learning_rate: {meta.get('learning_rate', 'n/a')}")
        features = meta.get("features", [])
        lines.append(f"- features: {', '.join(map(str, features))}")
        lines.append(f"- target: {meta.get('target', 'n/a')}")
        lines.append(f"- batch_size: {meta.get('batch_size', 'n/a')}")
        lines.append(f"- stateful: {meta.get('stateful', 'n/a')}")
        lines.append(f"- patience: {meta.get('patience', 'n/a')}")
        lines.append(f"- seed: {meta.get('seed', 'n/a')}")
        lines.append("")

        overall = meta.get("overall_eval")
        lines.append("All-trial Evaluation Summary")
        if isinstance(overall, dict):
            lines.append(f"- n_trials: {overall.get('n_trials', 'n/a')}")
            lines.append(f"- n_samples: {overall.get('n_samples', 'n/a')}")
            lines.append(f"- weighted_rmse_deg: {_fmt(overall.get('weighted_rmse_deg'))}")
            lines.append(f"- weighted_mae_deg: {_fmt(overall.get('weighted_mae_deg'))}")
        else:
            lines.append("- summary: n/a")

        return "\n".join(lines)

    def _set_info(self, text: str) -> None:
        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, text)
        self.info_text.configure(state=tk.DISABLED)

    def _on_model_change(self, _event: tk.Event) -> None:
        run_name = self.model_var.get().strip()
        if run_name:
            self._load_run(run_name)

    def _on_trial_change(self, _event: tk.Event) -> None:
        trial_name = self.trial_var.get().strip()
        if trial_name:
            self._load_trial(trial_name)

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        if self.current_trial_data is None:
            self._draw_message("Select a run and trial to plot")
            return
        self._refresh_trial_view()

    def _draw_message(self, message: str) -> None:
        self.canvas.delete("all")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.create_text(
            width / 2,
            height / 2,
            text=message,
            fill="#556071",
            font=("TkDefaultFont", 14),
        )

    def _draw_plot(
        self,
        x_label: str,
        x_vals: list[float],
        y_true: list[float],
        y_pred: list[float],
        trial_name: str,
    ) -> None:
        n = min(len(x_vals), len(y_true), len(y_pred))
        if n == 0:
            self._draw_message("No numeric points to plot")
            return

        if n > MAX_PLOT_POINTS:
            idxs = [int(i * (n - 1) / (MAX_PLOT_POINTS - 1)) for i in range(MAX_PLOT_POINTS)]
            xs = [x_vals[i] for i in idxs]
            ys_true = [y_true[i] for i in idxs]
            ys_pred = [y_pred[i] for i in idxs]
        else:
            xs = x_vals[:n]
            ys_true = y_true[:n]
            ys_pred = y_pred[:n]

        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.delete("all")

        pad_left = 66
        pad_right = 24
        pad_top = 42
        pad_bottom = 52
        plot_w = max(1, width - pad_left - pad_right)
        plot_h = max(1, height - pad_top - pad_bottom)

        x_min = min(xs)
        x_max = max(xs)
        y_all = ys_true + ys_pred
        y_min = min(y_all)
        y_max = max(y_all)
        if x_max <= x_min:
            x_max = x_min + 1.0
        if y_max <= y_min:
            y_max = y_min + 1.0

        self.canvas.create_rectangle(
            pad_left,
            pad_top,
            pad_left + plot_w,
            pad_top + plot_h,
            outline="#d6dbe5",
            width=1,
        )

        for frac in (0.2, 0.4, 0.6, 0.8):
            y = pad_top + frac * plot_h
            self.canvas.create_line(
                pad_left,
                y,
                pad_left + plot_w,
                y,
                fill="#edf0f5",
                width=1,
            )

        def x_px(v: float) -> float:
            return pad_left + ((v - x_min) / (x_max - x_min)) * plot_w

        def y_px(v: float) -> float:
            return pad_top + plot_h - ((v - y_min) / (y_max - y_min)) * plot_h

        def polyline(points_x: list[float], points_y: list[float]) -> list[float]:
            coords: list[float] = []
            for vx, vy in zip(points_x, points_y):
                coords.extend([x_px(vx), y_px(vy)])
            return coords

        true_coords = polyline(xs, ys_true)
        pred_coords = polyline(xs, ys_pred)
        if len(true_coords) >= 4:
            self.canvas.create_line(*true_coords, fill="#1f77b4", width=2)
        if len(pred_coords) >= 4:
            self.canvas.create_line(*pred_coords, fill="#d62728", width=2)

        self.canvas.create_text(
            pad_left + plot_w / 2,
            18,
            text=f"{trial_name} | Ground truth vs prediction",
            fill="#1f2937",
            font=("TkDefaultFont", 12, "bold"),
        )
        self.canvas.create_text(
            pad_left + plot_w / 2,
            height - 18,
            text=f"{x_label}: {x_min:.3f} to {x_max:.3f}",
            fill="#4b5563",
        )
        self.canvas.create_text(
            12,
            pad_top + plot_h / 2,
            text=f"deg\n{y_max:.3f}\n...\n{y_min:.3f}",
            fill="#4b5563",
            justify=tk.LEFT,
            anchor="w",
        )

        legend_y = pad_top + 10
        legend_x = pad_left + 10
        self.canvas.create_line(legend_x, legend_y, legend_x + 24, legend_y, fill="#1f77b4", width=2)
        self.canvas.create_text(legend_x + 30, legend_y, text="Ground truth", anchor="w")
        self.canvas.create_line(legend_x + 138, legend_y, legend_x + 162, legend_y, fill="#d62728", width=2)
        self.canvas.create_text(legend_x + 168, legend_y, text="Prediction", anchor="w")


def main() -> int:
    parser = argparse.ArgumentParser(description="View prediction outputs in a desktop window.")
    parser.add_argument(
        "--root",
        default="outputs/experiments",
        help="Root directory containing experiment run directories.",
    )
    args = parser.parse_args()

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(
            "Failed to open Tk window. Ensure a graphical display is available (DISPLAY/X11/Wayland).",
            file=sys.stderr,
        )
        print(f"Tk error: {exc}", file=sys.stderr)
        return 1
    PredictionViewer(root, Path(args.root).resolve())
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
