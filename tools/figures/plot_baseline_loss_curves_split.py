#!/usr/bin/env python3
"""Interactive helper to visualize baseline losses in two separate figures."""

from __future__ import annotations

import argparse
import csv
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

MODEL_COLORS: dict[str, str] = {
    "SLM-LSTM": "tab:blue",
    "SLU-LSTM": "tab:orange",
    "TLM-LSTM": "tab:green",
    "TLU-LSTM": "tab:red",
}

FONT_SCALE = 1.5
TITLE_SIZE = int(14 * FONT_SCALE)
LABEL_SIZE = int(12 * FONT_SCALE)
TICK_SIZE = int(10 * FONT_SCALE)
LEGEND_SIZE = int(10 * FONT_SCALE)


@dataclass(frozen=True)
class LossSeries:
    model_label: str
    epochs: list[int]
    train_loss: list[float]
    val_loss: list[float]


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


def _load_loss_series(repo_root: Path, model_label: str, run_dir_rel: str) -> LossSeries:
    run_dir = repo_root / run_dir_rel
    history_path = run_dir / "history.csv"

    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file: {history_path}")

    epochs, train_loss, val_loss = _read_history(history_path)
    return LossSeries(
        model_label=model_label,
        epochs=epochs,
        train_loss=train_loss,
        val_loss=val_loss,
    )


def _plot_one_metric(series_by_model: Sequence[LossSeries], metric: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)

    for series in series_by_model:
        y = series.train_loss if metric == "train" else series.val_loss
        ax.plot(
            series.epochs,
            y,
            lw=2.0,
            label=series.model_label,
            color=MODEL_COLORS[series.model_label],
        )

    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="normal")
    ax.set_xlabel("Epoch", fontsize=LABEL_SIZE)
    ax.set_ylabel("Loss (MSE, normalized units²)", fontsize=LABEL_SIZE)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.7)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.legend(loc="upper right", fontsize=LEGEND_SIZE)
    return fig


def _plot(series_by_model: Sequence[LossSeries], save_dir: Path, show: bool) -> None:
    fig_train = _plot_one_metric(
        series_by_model=series_by_model,
        metric="train",
        title="Baseline Models: Training Loss vs Epoch",
    )
    fig_val = _plot_one_metric(
        series_by_model=series_by_model,
        metric="val",
        title="Baseline Models: Validation Loss vs Epoch",
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    train_path = save_dir / "baseline_training_loss_curves.pdf"
    val_path = save_dir / "baseline_validation_loss_curves.pdf"
    fig_train.savefig(train_path, bbox_inches="tight")
    fig_val.savefig(val_path, bbox_inches="tight")
    print(f"Saved figure: {train_path}")
    print(f"Saved figure: {val_path}")

    if show:
        plt.show()
    else:
        plt.close(fig_train)
        plt.close(fig_val)


def _parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Open two interactive matplotlib figures for the 4 baseline models: "
            "one for training loss and one for validation loss."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help=f"Repository root path (default: {default_repo_root})",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Output directory for PDF files (default: outputs/figures).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Prepare (and optionally save) the figures without opening windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    save_dir = args.save_dir if args.save_dir.is_absolute() else (repo_root / args.save_dir)

    series_by_model = [
        _load_loss_series(repo_root=repo_root, model_label=model_label, run_dir_rel=run_dir_rel)
        for model_label, run_dir_rel in BASELINE_RUNS
    ]
    _plot(series_by_model=series_by_model, save_dir=save_dir, show=not args.no_show)


if __name__ == "__main__":
    main()
