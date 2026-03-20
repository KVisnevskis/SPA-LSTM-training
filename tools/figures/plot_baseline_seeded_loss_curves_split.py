#!/usr/bin/env python3
"""Interactive helper to visualize seeded SLM-LSTM losses in two figures."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


SEEDED_RUN_ROOT = Path("outputs/experiments/baseline_slm_seeded_replication")
BASE_RUN_DIR = Path("outputs/experiments/baseline/baseline_slm_lstm")

SEED_COLORS: tuple[str, ...] = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:brown",
    "tab:purple",
)

FONT_SCALE = 1.5
TITLE_SIZE = int(14 * FONT_SCALE)
LABEL_SIZE = int(12 * FONT_SCALE)
TICK_SIZE = int(10 * FONT_SCALE)
LEGEND_SIZE = int(10 * FONT_SCALE)


@dataclass(frozen=True)
class LossSeries:
    seed_label: str
    epochs: list[int]
    train_loss: list[float]
    val_loss: list[float]


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return raw


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


def _load_seed_info(run_dir: Path) -> tuple[int, str, Path] | None:
    snapshot_path = run_dir / "config_snapshot.json"
    history_path = run_dir / "history.csv"
    if not snapshot_path.exists() or not history_path.exists():
        return None

    snapshot = _load_json(snapshot_path)
    training_cfg = snapshot.get("training")
    if not isinstance(training_cfg, dict):
        return None
    raw_seed = training_cfg.get("seed")
    if not isinstance(raw_seed, (int, float)):
        return None

    seed = int(raw_seed)
    return seed, f"Seed {seed}", run_dir


def _discover_seeded_runs(
    repo_root: Path,
    seed_run_root: Path,
    base_run_dir: Path,
    include_base_run: bool,
) -> list[tuple[int, str, Path]]:
    discovered_by_seed: dict[int, tuple[int, str, Path]] = {}

    candidate_dirs: list[Path] = []
    if include_base_run:
        candidate_dirs.append(base_run_dir if base_run_dir.is_absolute() else (repo_root / base_run_dir))
    run_root = seed_run_root if seed_run_root.is_absolute() else (repo_root / seed_run_root)
    candidate_dirs.extend(sorted(path for path in run_root.iterdir() if path.is_dir()))

    for run_dir in candidate_dirs:
        info = _load_seed_info(run_dir)
        if info is None:
            continue
        seed = info[0]
        if seed in discovered_by_seed:
            raise ValueError(f"Duplicate seed discovered for seed {seed}: {discovered_by_seed[seed][2]} and {run_dir}")
        discovered_by_seed[seed] = info

    if not discovered_by_seed:
        raise ValueError("No seeded runs with history/config snapshot were found.")
    return [discovered_by_seed[seed] for seed in sorted(discovered_by_seed)]


def _parse_seed_list(raw: str) -> set[int]:
    out: set[int] = set()
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            out.add(int(text))
        except ValueError as exc:
            raise ValueError(f"Invalid seed value '{text}' in seed list '{raw}'") from exc
    return out


def _load_loss_series(run_dir: Path, seed_label: str) -> LossSeries:
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file: {history_path}")

    epochs, train_loss, val_loss = _read_history(history_path)
    return LossSeries(
        seed_label=seed_label,
        epochs=epochs,
        train_loss=train_loss,
        val_loss=val_loss,
    )


def _plot_one_metric(series_by_seed: Sequence[LossSeries], metric: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)

    for idx, series in enumerate(series_by_seed):
        y = series.train_loss if metric == "train" else series.val_loss
        ax.plot(
            series.epochs,
            y,
            lw=2.0,
            label=series.seed_label,
            color=SEED_COLORS[idx % len(SEED_COLORS)],
        )

    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="normal")
    ax.set_xlabel("Epoch", fontsize=LABEL_SIZE)
    ax.set_ylabel("Loss (MSE, normalized units²)", fontsize=LABEL_SIZE)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.7)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.legend(loc="upper right", fontsize=LEGEND_SIZE)
    return fig


def _plot(series_by_seed: Sequence[LossSeries], save_dir: Path, show: bool) -> None:
    fig_train = _plot_one_metric(
        series_by_seed=series_by_seed,
        metric="train",
        title="Seeded SLM-LSTM Runs: Training Loss vs Epoch",
    )
    fig_val = _plot_one_metric(
        series_by_seed=series_by_seed,
        metric="val",
        title="Seeded SLM-LSTM Runs: Validation Loss vs Epoch",
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    train_path = save_dir / "baseline_seeded_training_loss_curves.pdf"
    val_path = save_dir / "baseline_seeded_validation_loss_curves.pdf"
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
            "Open two interactive matplotlib figures for seeded SLM-LSTM runs: "
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
        "--seed-run-root",
        type=Path,
        default=SEEDED_RUN_ROOT,
        help="Directory containing seeded run subdirectories.",
    )
    parser.add_argument(
        "--base-run-dir",
        type=Path,
        default=BASE_RUN_DIR,
        help="Original baseline SLM-LSTM run directory to include as Seed 42.",
    )
    parser.add_argument(
        "--include-base-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the original baseline SLM-LSTM run as Seed 42 (default: true).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("outputs/figures/ch4"),
        help="Output directory for PDF files (default: outputs/figures/ch4).",
    )
    parser.add_argument(
        "--exclude-seeds",
        default="",
        help="Comma-separated seed values to omit from the plots (for example: 97).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Prepare the figures without opening windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    save_dir = args.save_dir if args.save_dir.is_absolute() else (repo_root / args.save_dir)
    seeded_runs = _discover_seeded_runs(
        repo_root=repo_root,
        seed_run_root=args.seed_run_root,
        base_run_dir=args.base_run_dir,
        include_base_run=bool(args.include_base_run),
    )
    excluded_seeds = _parse_seed_list(args.exclude_seeds)
    if excluded_seeds:
        seeded_runs = [item for item in seeded_runs if item[0] not in excluded_seeds]
    if not seeded_runs:
        raise ValueError("No seeded runs remain after applying --exclude-seeds.")
    series_by_seed = [
        _load_loss_series(run_dir=run_dir, seed_label=seed_label)
        for _, seed_label, run_dir in seeded_runs
    ]
    _plot(series_by_seed=series_by_seed, save_dir=save_dir, show=not args.no_show)


if __name__ == "__main__":
    main()
