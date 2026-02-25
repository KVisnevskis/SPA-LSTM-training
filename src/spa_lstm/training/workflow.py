"""End-to-end training workflow from config."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from spa_lstm.config import ExperimentConfig
from spa_lstm.data.constants import REQUIRED_BASE_COLUMNS
from spa_lstm.data.hdf5_loader import load_runs_as_dataframes
from spa_lstm.data.scaling import resolve_scaling_bounds, save_bounds_json, scale_dataframe
from spa_lstm.data.splits import assert_disjoint_splits
from spa_lstm.models.factory import build_lstm_model
from spa_lstm.training.stateful import train_stateful


def _set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def _to_xy(df, features: list[str], target: str) -> tuple[np.ndarray, np.ndarray]:
    x = df[features].to_numpy(dtype=np.float32)
    y = df[[target]].to_numpy(dtype=np.float32)
    return x, y


def run_training(cfg: ExperimentConfig) -> Path:
    """Execute training run and return output directory path."""

    cfg.validate()
    assert_disjoint_splits(cfg.data.train_runs, cfg.data.val_runs, cfg.data.eval_runs)
    _set_reproducible_seed(cfg.training.seed)

    output_dir = Path(cfg.runtime.output_dir) / cfg.runtime.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    required_columns = tuple(dict.fromkeys(list(REQUIRED_BASE_COLUMNS) + cfg.data.features + [cfg.data.target]))

    train_raw = load_runs_as_dataframes(cfg.data.h5_path, cfg.data.train_runs, required_columns)
    val_raw = load_runs_as_dataframes(cfg.data.h5_path, cfg.data.val_runs, required_columns)

    bounds = resolve_scaling_bounds(cfg.data.scaling, train_raw, cfg.data.features, cfg.data.target)
    if bounds:
        save_bounds_json(output_dir / cfg.runtime.bounds_path, bounds)

    train_scaled = {
        k: scale_dataframe(v, cfg.data.scaling, bounds, cfg.data.features, cfg.data.target)
        for k, v in train_raw.items()
    }
    val_scaled = {
        k: scale_dataframe(v, cfg.data.scaling, bounds, cfg.data.features, cfg.data.target)
        for k, v in val_raw.items()
    }

    if len(cfg.data.train_runs) != len(cfg.data.val_runs):
        raise ValueError("Stateful paired validation requires equal train_runs and val_runs length.")

    train_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for train_key, val_key in zip(cfg.data.train_runs, cfg.data.val_runs):
        x_tr, y_tr = _to_xy(train_scaled[train_key], cfg.data.features, cfg.data.target)
        x_va, y_va = _to_xy(val_scaled[val_key], cfg.data.features, cfg.data.target)
        train_pairs.append((x_tr, y_tr, x_va, y_va))

    model = build_lstm_model(cfg.model, cfg.training, num_features=len(cfg.data.features))

    history = train_stateful(
        model=model,
        train_pairs=train_pairs,
        epochs=cfg.training.epochs,
        patience=cfg.training.patience,
    )

    best_model_path = output_dir / cfg.runtime.save_best_path
    final_model_path = output_dir / cfg.runtime.save_final_path
    model.save(best_model_path)
    model.save(final_model_path)

    history_path = output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss_mean", "val_loss_mean"])
        writer.writeheader()
        for epoch in history:
            writer.writerow(asdict(epoch))

    manifest = {
        "config_name": cfg.name,
        "h5_path": cfg.data.h5_path,
        "model_variant": cfg.model.variant,
        "best_model": str(best_model_path),
        "final_model": str(final_model_path),
        "history": str(history_path),
        "epochs_completed": len(history),
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return output_dir

