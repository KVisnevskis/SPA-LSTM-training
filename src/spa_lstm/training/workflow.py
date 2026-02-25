"""End-to-end training workflow from config."""

from __future__ import annotations

import csv
import json
import platform
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from spa_lstm.config import ExperimentConfig
from spa_lstm.data.constants import REQUIRED_BASE_COLUMNS
from spa_lstm.data.hdf5_loader import load_runs_as_dataframes
from spa_lstm.data.scaling import load_hdf5_scaler_bounds
from spa_lstm.data.splits import assert_disjoint_splits, assert_no_duplicate_runs
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


def _require_dataset_exists(h5_path: str) -> None:
    dataset_path = Path(h5_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"HDF5 dataset not found at '{dataset_path}'. "
            "Generate it first (see context/lstm_baseline_handoff.md)."
        )


def _validate_split_config(cfg: ExperimentConfig) -> None:
    if not cfg.data.train_runs:
        raise ValueError("data.train_runs is empty.")
    if not cfg.data.val_runs:
        raise ValueError("data.val_runs is empty.")
    if len(cfg.data.train_runs) != len(cfg.data.val_runs):
        raise ValueError("Stateful paired validation requires equal train_runs and val_runs length.")


def _validate_run_arrays(
    runs: dict[str, Any], run_keys: list[str], feature_columns: list[str], target_column: str, split_name: str
) -> None:
    columns = list(dict.fromkeys(feature_columns + [target_column]))
    for run_key in run_keys:
        df = runs[run_key]
        if len(df) == 0:
            raise ValueError(f"{split_name} run '{run_key}' has zero rows.")
        try:
            arr = df[columns].to_numpy(dtype=np.float64)
        except Exception as exc:
            raise ValueError(
                f"{split_name} run '{run_key}' has non-numeric values in required columns {columns}."
            ) from exc
        if arr.ndim != 2 or arr.shape[1] != len(columns):
            raise ValueError(
                f"{split_name} run '{run_key}' has unexpected array shape {arr.shape} for columns {columns}."
            )
        if not np.isfinite(arr).all():
            raise ValueError(
                f"{split_name} run '{run_key}' has non-finite values (NaN/Inf) in required columns {columns}."
            )


def _collect_environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import tensorflow as tf

        info["tensorflow"] = tf.__version__
        info["gpu_devices"] = [device.name for device in tf.config.list_physical_devices("GPU")]
    except Exception:
        info["tensorflow"] = None
        info["gpu_devices"] = []
    return info


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_training(cfg: ExperimentConfig) -> Path:
    """Execute training run and return output directory path."""

    cfg.validate()
    _require_dataset_exists(cfg.data.h5_path)
    _validate_split_config(cfg)
    assert_no_duplicate_runs(cfg.data.train_runs, cfg.data.val_runs, cfg.data.eval_runs)
    assert_disjoint_splits(cfg.data.train_runs, cfg.data.val_runs, cfg.data.eval_runs)
    _set_reproducible_seed(cfg.training.seed)

    output_dir = Path(cfg.runtime.output_dir) / cfg.runtime.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    required_columns = tuple(
        dict.fromkeys(
            list(REQUIRED_BASE_COLUMNS)
            + cfg.data.features
            + [cfg.data.target]
        )
    )

    train_raw = load_runs_as_dataframes(cfg.data.h5_path, cfg.data.train_runs, required_columns)
    val_raw = load_runs_as_dataframes(cfg.data.h5_path, cfg.data.val_runs, required_columns)
    _validate_run_arrays(train_raw, cfg.data.train_runs, cfg.data.features, cfg.data.target, "train")
    _validate_run_arrays(val_raw, cfg.data.val_runs, cfg.data.features, cfg.data.target, "val")

    # Training uses already-scaled values from the preprocessed HDF5.
    train_scaled = train_raw
    val_scaled = val_raw

    scale_columns = list(dict.fromkeys(cfg.data.features + [cfg.data.target]))
    bounds = load_hdf5_scaler_bounds(cfg.data.h5_path, columns=scale_columns)
    _write_json(output_dir / cfg.runtime.bounds_path, {column: asdict(value) for column, value in bounds.items()})

    train_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for train_key, val_key in zip(cfg.data.train_runs, cfg.data.val_runs):
        x_tr, y_tr = _to_xy(train_scaled[train_key], cfg.data.features, cfg.data.target)
        x_va, y_va = _to_xy(val_scaled[val_key], cfg.data.features, cfg.data.target)
        train_pairs.append((x_tr, y_tr, x_va, y_va))

    model = build_lstm_model(cfg.model, cfg.training, num_features=len(cfg.data.features))

    train_result = train_stateful(
        model=model,
        train_pairs=train_pairs,
        epochs=cfg.training.epochs,
        patience=cfg.training.patience,
    )

    best_model_path = output_dir / cfg.runtime.save_best_path
    final_model_path = output_dir / cfg.runtime.save_final_path
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(final_model_path)
    model.set_weights(train_result.best_weights)
    model.save(best_model_path)

    history_path = output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss_mean",
                "train_rmse_mean",
                "train_mae_mean",
                "val_loss_mean",
                "val_rmse_mean",
                "val_mae_mean",
                "learning_rate",
            ],
        )
        writer.writeheader()
        for epoch in train_result.history:
            writer.writerow(asdict(epoch))

    config_snapshot_path = output_dir / "config_snapshot.json"
    _write_json(config_snapshot_path, asdict(cfg))

    training_summary = {
        "epochs_completed": len(train_result.history),
        "best_epoch": train_result.best_epoch,
        "best_val_loss": train_result.best_val_loss,
        "stopped_early": train_result.stopped_early,
    }
    training_summary_path = output_dir / "training_summary.json"
    _write_json(training_summary_path, training_summary)

    manifest = {
        "config_name": cfg.name,
        "h5_path": cfg.data.h5_path,
        "features": cfg.data.features,
        "target": cfg.data.target,
        "scaling_mode": cfg.data.scaling.mode,
        "model_variant": cfg.model.variant,
        "best_model": str(best_model_path),
        "final_model": str(final_model_path),
        "history": str(history_path),
        "bounds": str(output_dir / cfg.runtime.bounds_path),
        "config_snapshot": str(config_snapshot_path),
        "training_summary": str(training_summary_path),
        "epochs_completed": len(train_result.history),
        "split_counts": {
            "train_runs": len(cfg.data.train_runs),
            "val_runs": len(cfg.data.val_runs),
            "eval_runs": len(cfg.data.eval_runs),
        },
        "environment": _collect_environment_info(),
    }
    _write_json(output_dir / "run_manifest.json", manifest)

    return output_dir
