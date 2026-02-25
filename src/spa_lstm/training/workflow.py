"""End-to-end training workflow from config."""

from __future__ import annotations

import csv
import json
import platform
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from spa_lstm.config import ExperimentConfig
from spa_lstm.data.constants import REQUIRED_BASE_COLUMNS
from spa_lstm.data.hdf5_loader import load_runs_as_dataframes
from spa_lstm.data.scaling import load_hdf5_scaler_bounds
from spa_lstm.data.splits import assert_disjoint_splits, assert_no_duplicate_runs
from spa_lstm.models.factory import build_lstm_model
from spa_lstm.training.resource_monitor import ResourceMonitor
from spa_lstm.training.stateful import EpochSummary, TrainingResult, train_stateful


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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in '{path}', got {type(raw).__name__}.")
    return raw


def _load_keras_model(model_path: Path):
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("TensorFlow is required to load a saved model for resume.") from exc
    return tf.keras.models.load_model(str(model_path))


def _parse_epoch_summary(raw: dict[str, Any]) -> EpochSummary:
    return EpochSummary(
        epoch=int(raw["epoch"]),
        train_loss_mean=float(raw["train_loss_mean"]),
        train_rmse_mean=float(raw["train_rmse_mean"]),
        train_mae_mean=float(raw["train_mae_mean"]),
        val_loss_mean=float(raw["val_loss_mean"]),
        val_rmse_mean=float(raw["val_rmse_mean"]),
        val_mae_mean=float(raw["val_mae_mean"]),
        learning_rate=float(raw["learning_rate"]),
    )


def run_training(cfg: ExperimentConfig, resume: bool = False) -> Path:
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

    best_model_path = output_dir / cfg.runtime.save_best_path
    final_model_path = output_dir / cfg.runtime.save_final_path
    latest_model_path = output_dir / "latest.keras"
    resume_state_path = output_dir / "resume_state.json"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_lstm_model(cfg.model, cfg.training, num_features=len(cfg.data.features))
    history_seed: list[EpochSummary] = []
    start_epoch = 1
    best_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improve = 0
    best_weights = model.get_weights()
    resumed_from_checkpoint = False

    has_resume_artifacts = resume_state_path.exists() and latest_model_path.exists()
    if resume and has_resume_artifacts:
        raw_state = _load_json(resume_state_path)
        raw_history = raw_state.get("history", [])
        if not isinstance(raw_history, list):
            raise ValueError(f"Invalid resume state history in '{resume_state_path}'.")
        history_seed = [_parse_epoch_summary(item) for item in raw_history]
        start_epoch = int(raw_state.get("next_epoch", len(history_seed) + 1))
        best_epoch = int(raw_state.get("best_epoch", 0))
        best_val_loss = float(raw_state.get("best_val_loss", float("inf")))
        epochs_without_improve = int(raw_state.get("epochs_without_improve", 0))

        model = _load_keras_model(latest_model_path)
        if best_model_path.exists():
            best_weights = _load_keras_model(best_model_path).get_weights()
        else:
            best_weights = model.get_weights()
        resumed_from_checkpoint = True
    elif resume and not has_resume_artifacts:
        print("Resume requested but no resume artifacts found; starting a fresh training run.")

    def _persist_epoch_state(
        summary: EpochSummary,
        history: list[EpochSummary],
        best_epoch_now: int,
        best_val_loss_now: float,
        epochs_without_improve_now: int,
    ) -> None:
        model.save(latest_model_path)
        if summary.epoch == best_epoch_now:
            model.save(best_model_path)
        _write_json(
            resume_state_path,
            {
                "next_epoch": summary.epoch + 1,
                "best_epoch": best_epoch_now,
                "best_val_loss": best_val_loss_now,
                "epochs_without_improve": epochs_without_improve_now,
                "history": [asdict(epoch) for epoch in history],
                "latest_model": str(latest_model_path),
                "best_model": str(best_model_path),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    resource_csv_path = output_dir / "resource_usage.csv"
    monitor = ResourceMonitor(resource_csv_path, interval_seconds=15.0)
    resource_info: dict[str, Any] = {
        "resource_usage_csv": str(resource_csv_path),
        "resource_samples": 0,
        "resource_interval_seconds": 15.0,
        "gpu_metrics_observed": False,
    }
    monitor_started = False

    try:
        monitor.start()
        monitor_started = True
    except Exception as exc:
        print(f"Resource monitor disabled: {exc}")

    try:
        if start_epoch > cfg.training.epochs:
            train_result = TrainingResult(
                history=history_seed,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                stopped_early=False,
                best_weights=best_weights,
            )
        else:
            train_result = train_stateful(
                model=model,
                train_pairs=train_pairs,
                epochs=cfg.training.epochs,
                patience=cfg.training.patience,
                start_epoch=start_epoch,
                initial_history=history_seed,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                best_weights=best_weights,
                epochs_without_improve=epochs_without_improve,
                on_epoch_end=_persist_epoch_state,
            )
    finally:
        if monitor_started:
            resource_info = monitor.stop()

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
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "resume_state_path": str(resume_state_path),
        "latest_model_path": str(latest_model_path),
        **resource_info,
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
        "resume_state": str(resume_state_path),
        "latest_model": str(latest_model_path),
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "resource_usage": resource_info,
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
