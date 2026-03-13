#!/usr/bin/env python3
"""Interactive helper to visualize baseline training/validation loss curves."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


BASELINE_RUNS: tuple[tuple[str, str], ...] = (
    ("SLM-LSTM", "outputs/experiments/baseline/baseline_slm_lstm"),
    ("SLU-LSTM", "outputs/experiments/baseline/baseline_slu_lstm"),
    ("TLM-LSTM", "outputs/experiments/baseline/baseline_tlm_lstm"),
    ("TLU-LSTM", "outputs/experiments/baseline/baseline_tlu_lstm"),
)


@dataclass(frozen=True)
class LossSeries:
    model_label: str
    epochs: list[int]
    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int | None
    stop_epoch: int | None
    best_val_loss: float | None


def _read_history(history_path: Path) -> tuple[list[int], list[float], list[float]]:
    epochs: list[int] = []
    train_loss: list[float] = []
    val_loss: list[float] = []

    with history_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(float(row["epoch"]))
                train = float(row["train_loss_mean"])
                val = float(row["val_loss_mean"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid history row in {history_path}: {row}") from exc

            epochs.append(epoch)
            train_loss.append(train)
            val_loss.append(val)

    if not epochs:
        raise ValueError(f"No data rows found in {history_path}")

    return epochs, train_loss, val_loss


def _read_training_summary(summary_path: Path) -> tuple[int | None, int | None, bool, float | None]:
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    if not isinstance(summary, dict):
        raise ValueError(f"Expected object in {summary_path}")

    best_epoch_raw = summary.get("best_epoch")
    stop_epoch_raw = summary.get("epochs_completed")
    stopped_early_raw = summary.get("stopped_early")
    best_val_loss_raw = summary.get("best_val_loss")

    best_epoch = int(best_epoch_raw) if isinstance(best_epoch_raw, (int, float)) else None
    stop_epoch = int(stop_epoch_raw) if isinstance(stop_epoch_raw, (int, float)) else None
    stopped_early = bool(stopped_early_raw) if isinstance(stopped_early_raw, bool) else False
    best_val_loss = (
        float(best_val_loss_raw) if isinstance(best_val_loss_raw, (int, float)) else None
    )

    if not stopped_early:
        stop_epoch = None

    return best_epoch, stop_epoch, stopped_early, best_val_loss


def _load_loss_series(repo_root: Path, model_label: str, run_dir_rel: str) -> LossSeries:
    run_dir = repo_root / run_dir_rel
    history_path = run_dir / "history.csv"
    summary_path = run_dir / "training_summary.json"

    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file: {history_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    epochs, train_loss, val_loss = _read_history(history_path)
    best_epoch, stop_epoch, _stopped_early, best_val_loss = _read_training_summary(summary_path)

    max_epoch = max(epochs)
    if best_epoch is not None:
        best_epoch = max(1, min(best_epoch, max_epoch))
    if stop_epoch is not None:
        stop_epoch = max(1, min(stop_epoch, max_epoch))

    return LossSeries(
        model_label=model_label,
        epochs=epochs,
        train_loss=train_loss,
        val_loss=val_loss,
        best_epoch=best_epoch,
        stop_epoch=stop_epoch,
        best_val_loss=best_val_loss,
    )


def _dedupe_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    ax.legend(unique_handles, unique_labels, loc="upper right", fontsize=9)


def _plot(series_by_model: Sequence[LossSeries], save_path: Path | None, show: bool) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax, series in zip(axes_flat, series_by_model):
        ax.plot(series.epochs, series.train_loss, color="tab:blue", lw=1.8, label="Train loss")
        ax.plot(series.epochs, series.val_loss, color="tab:orange", lw=1.8, label="Validation loss")

        if series.best_epoch is not None:
            ax.axvline(
                series.best_epoch,
                color="tab:green",
                lw=1.6,
                ls="--",
                label=f"Lowest val loss epoch ({series.best_epoch})",
            )
        if series.stop_epoch is not None:
            ax.axvline(
                series.stop_epoch,
                color="tab:red",
                lw=1.6,
                ls=":",
                label=f"Early stopping epoch ({series.stop_epoch})",
            )

        best_val_text = (
            f"best val loss={series.best_val_loss:.4g}" if series.best_val_loss is not None else "best val loss=n/a"
        )
        ax.set_title(f"{series.model_label} ({best_val_text})", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.7)
        _dedupe_legend(ax)

    fig.suptitle(
        "Baseline Models: Training and Validation Loss Curves",
        fontsize=15,
        fontweight="bold",
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Open an interactive matplotlib figure with training/validation loss curves for "
            "the 4 baseline models."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help=f"Repository root path (default: {default_repo_root})",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output image path (e.g., figures/ch5/baseline_loss_curves.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Prepare (and optionally save) the figure without opening a window.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()

    series_by_model = [
        _load_loss_series(repo_root=repo_root, model_label=model_label, run_dir_rel=run_dir_rel)
        for model_label, run_dir_rel in BASELINE_RUNS
    ]
    _plot(series_by_model=series_by_model, save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
