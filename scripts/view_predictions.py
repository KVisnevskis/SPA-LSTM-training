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
from tkinter import filedialog, messagebox, ttk

MATPLOTLIB_IMPORT_ERROR: Exception | None = None
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover - runtime dependency
    MATPLOTLIB_IMPORT_ERROR = exc


PRED_TRUE_COL = "phi_true_deg"
PRED_PRED_COL = "phi_pred_deg"
PRED_TIME_COL = "Time"
MAX_PLOT_POINTS = 2400
RAD_TO_DEG = 180.0 / math.pi
ANGLE_UNIT_OPTIONS = ("degrees", "radians")


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


def _convert_angle(value: float | None, unit: str) -> float | None:
    if value is None:
        return None
    if unit == "deg":
        return value * RAD_TO_DEG
    return value


def _convert_angles(values: list[float], unit: str) -> list[float]:
    if unit == "deg":
        return [v * RAD_TO_DEG for v in values]
    return values


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
        # eval_metrics uses legacy *_deg names but values are currently radians.
        rmse_rad = _safe_float(str(row.get("rmse_deg", "")))
        mae_rad = _safe_float(str(row.get("mae_deg", "")))
        if n is None or rmse_rad is None or mae_rad is None or n <= 0:
            continue
        n_int = int(n)
        n_trials += 1
        n_samples_total += n_int
        mse_sum += (rmse_rad * rmse_rad) * n_int
        mae_sum += mae_rad * n_int

    if n_trials == 0 or n_samples_total == 0:
        return None
    weighted_rmse_rad = math.sqrt(mse_sum / n_samples_total)
    weighted_mae_rad = mae_sum / n_samples_total
    return {
        "n_trials": n_trials,
        "n_samples": n_samples_total,
        "weighted_rmse_rad": weighted_rmse_rad,
        "weighted_mae_rad": weighted_mae_rad,
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
        "config_name": config_snapshot.get("name", manifest.get("config_name", run_dir.name)),
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
            # Keep internal representation in radians; convert at display time.
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


def _find_eval_record(eval_metrics: Any, trial_name: str, unit: str) -> dict[str, Any] | None:
    if not isinstance(eval_metrics, list):
        return None
    for row in eval_metrics:
        if isinstance(row, dict) and str(row.get("run_key")) == trial_name:
            rmse_rad = _safe_float(str(row.get("rmse_deg", "")))
            mae_rad = _safe_float(str(row.get("mae_deg", "")))
            return {
                **row,
                "rmse_display": _convert_angle(rmse_rad, unit),
                "mae_display": _convert_angle(mae_rad, unit),
            }
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
        self.unit_var = tk.StringVar(value="degrees")
        self.rmse_var = tk.StringVar(value="-")
        self.mae_var = tk.StringVar(value="-")
        self.max_err_var = tk.StringVar(value="-")
        self.bias_var = tk.StringVar(value="-")
        self.best_epoch_var = tk.StringVar(value="-")
        self.best_val_var = tk.StringVar(value="-")
        self.status_var = tk.StringVar(value="")

        self.model_combo: ttk.Combobox
        self.trial_combo: ttk.Combobox
        self.unit_combo: ttk.Combobox
        self.rmse_label: ttk.Label
        self.mae_label: ttk.Label
        self.max_err_label: ttk.Label
        self.bias_label: ttk.Label
        self.figure: Figure
        self.ax: Any
        self.plot_canvas: FigureCanvasTkAgg
        self.toolbar: NavigationToolbar2Tk
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

        ttk.Label(controls, text="Angle unit").pack(side=tk.LEFT, padx=(18, 0))
        self.unit_combo = ttk.Combobox(
            controls,
            textvariable=self.unit_var,
            state="readonly",
            width=10,
            values=ANGLE_UNIT_OPTIONS,
        )
        self.unit_combo.pack(side=tk.LEFT, padx=(8, 0))
        self.unit_combo.bind("<<ComboboxSelected>>", self._on_unit_change)

        main = ttk.Frame(container)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(main, text="Predicted vs Ground Truth")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        plot_actions = ttk.Frame(left)
        plot_actions.pack(fill=tk.X, padx=6, pady=(6, 0))
        export_btn = ttk.Button(plot_actions, text="Export PNG", command=self._export_png)
        export_btn.pack(side=tk.RIGHT)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=left)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.toolbar = NavigationToolbar2Tk(self.plot_canvas, left, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X, padx=6, pady=(0, 6))

        right = ttk.LabelFrame(main, text="Metrics and Best Params")
        right.pack(side=tk.RIGHT, fill=tk.Y)

        metrics_grid = ttk.Frame(right, padding=(8, 8, 8, 4))
        metrics_grid.pack(fill=tk.X)

        self.rmse_label = self._metric_row(metrics_grid, 0, "Trial RMSE (deg)", self.rmse_var)
        self.mae_label = self._metric_row(metrics_grid, 1, "Trial MAE (deg)", self.mae_var)
        self.max_err_label = self._metric_row(metrics_grid, 2, "Max |error| (deg)", self.max_err_var)
        self.bias_label = self._metric_row(metrics_grid, 3, "Bias (deg)", self.bias_var)
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

        self._update_unit_labels()

    def _metric_row(self, parent: ttk.Frame, row: int, label: str, value_var: tk.StringVar) -> ttk.Label:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=2)
        ttk.Label(parent, textvariable=value_var).grid(row=row, column=1, sticky="e", pady=2)
        return label_widget

    def _selected_unit(self) -> str:
        return "rad" if self.unit_var.get().strip().lower().startswith("rad") else "deg"

    def _update_unit_labels(self) -> None:
        unit = self._selected_unit()
        suffix = "rad" if unit == "rad" else "deg"
        self.rmse_label.configure(text=f"Trial RMSE ({suffix})")
        self.mae_label.configure(text=f"Trial MAE ({suffix})")
        self.max_err_label.configure(text=f"Max |error| ({suffix})")
        self.bias_label.configure(text=f"Bias ({suffix})")

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

        unit = self._selected_unit()
        self._update_unit_labels()

        x_label, x_vals, y_true_rad, y_pred_rad = self.current_trial_data
        y_true = _convert_angles(y_true_rad, unit)
        y_pred = _convert_angles(y_pred_rad, unit)
        stats = _compute_trial_stats(y_true, y_pred)
        eval_record = _find_eval_record(
            None if self.current_meta is None else self.current_meta.get("eval_metrics"),
            self.current_trial_name or "",
            unit,
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

        config_name = "n/a" if self.current_meta is None else str(self.current_meta.get("config_name", "n/a"))
        self._draw_plot(
            x_label,
            x_vals,
            y_true,
            y_pred,
            self.current_trial_name or "",
            config_name,
            unit,
        )
        self._set_info(self._build_detail_text(stats, eval_record, unit))
        self.status_var.set(
            f"run={self.current_run.name} | trial={self.current_trial_name} | unit={unit} | samples={stats.n_samples}"
        )

    def _build_detail_text(self, stats: TrialStats, eval_record: dict[str, Any] | None, unit: str) -> str:
        run_name = self.current_run.name if self.current_run else "n/a"
        trial = self.current_trial_name or "n/a"
        meta = self.current_meta or {}
        unit_label = "deg" if unit == "deg" else "rad"

        lines: list[str] = []
        lines.append("Selected Trial")
        lines.append(f"- run: {run_name}")
        lines.append(f"- trial: {trial}")
        lines.append(f"- samples: {stats.n_samples}")
        lines.append(f"- rmse_{unit_label} (recomputed): {_fmt(stats.rmse_deg)}")
        lines.append(f"- mae_{unit_label} (recomputed): {_fmt(stats.mae_deg)}")
        lines.append(f"- max_abs_err_{unit_label}: {_fmt(stats.max_abs_err_deg)}")
        lines.append(f"- bias_{unit_label}: {_fmt(stats.bias_deg)}")
        lines.append("")

        lines.append("eval_metrics.json record")
        if eval_record is None:
            lines.append("- matching record: n/a")
        else:
            lines.append(f"- rmse_{unit_label}: {_fmt(eval_record.get('rmse_display'))}")
            lines.append(f"- mae_{unit_label}: {_fmt(eval_record.get('mae_display'))}")
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
            lines.append(
                f"- weighted_rmse_{unit_label}: {_fmt(_convert_angle(overall.get('weighted_rmse_rad'), unit))}"
            )
            lines.append(
                f"- weighted_mae_{unit_label}: {_fmt(_convert_angle(overall.get('weighted_mae_rad'), unit))}"
            )
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

    def _on_unit_change(self, _event: tk.Event) -> None:
        self._update_unit_labels()
        if self.current_trial_data is None:
            return
        self._refresh_trial_view()

    def _export_png(self) -> None:
        if self.current_run is None or self.current_trial_name is None:
            messagebox.showinfo("Export PNG", "Select a run and trial before exporting.")
            return

        default_name = f"{self.current_run.name}__{self.current_trial_name}.png"
        selected = filedialog.asksaveasfilename(
            title="Export Figure as PNG",
            defaultextension=".png",
            initialfile=default_name,
            initialdir=str(self.current_run.resolve()),
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not selected:
            return

        try:
            self.figure.savefig(selected, dpi=220, bbox_inches="tight")
        except Exception as exc:
            messagebox.showerror("Export Failed", f"Failed to save figure:\n{exc}")
            return

        self.status_var.set(f"Saved PNG: {selected}")

    def _draw_message(self, message: str) -> None:
        self.ax.clear()
        self.ax.text(
            0.5,
            0.5,
            message,
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            color="#556071",
            fontsize=12,
        )
        self.ax.set_axis_off()
        self.figure.tight_layout()
        self.plot_canvas.draw_idle()

    def _draw_plot(
        self,
        x_label: str,
        x_vals: list[float],
        y_true: list[float],
        y_pred: list[float],
        trial_name: str,
        config_name: str,
        unit: str,
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

        self.ax.clear()
        self.ax.plot(xs, ys_true, color="#1f77b4", linewidth=1.4, label="Ground truth")
        self.ax.plot(xs, ys_pred, color="#d62728", linewidth=1.4, label="Prediction")
        self.ax.set_title(f"{config_name} | {trial_name} | Ground truth vs prediction")
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(f"Bending angle ({unit})")
        self.ax.grid(True, linestyle="-", alpha=0.25)
        self.ax.legend(loc="best")
        self.figure.tight_layout()
        self.plot_canvas.draw_idle()


def main() -> int:
    parser = argparse.ArgumentParser(description="View prediction outputs in a desktop window.")
    parser.add_argument(
        "--root",
        default="outputs/experiments",
        help="Root directory containing experiment run directories.",
    )
    args = parser.parse_args()

    if MATPLOTLIB_IMPORT_ERROR is not None:
        print("Matplotlib is required for plotting. Install it with: pip install matplotlib", file=sys.stderr)
        print(f"Import error: {MATPLOTLIB_IMPORT_ERROR}", file=sys.stderr)
        return 1

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
