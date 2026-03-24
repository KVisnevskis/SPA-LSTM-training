"""MLP training workflow support."""

from __future__ import annotations

import csv
import json
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlp.config import ExperimentConfig
from mlp.data import prepare_training_data
from mlp.model import build_mlp_model

__all__ = ["EpochSummary", "TrainingResult", "run_training"]


@dataclass
class EpochSummary:
    """Per-epoch aggregate metrics captured by the MLP trainer."""

    epoch: int
    train_loss_mean: float
    train_rmse_mean: float
    train_mae_mean: float
    val_loss_mean: float
    val_rmse_mean: float
    val_mae_mean: float
    learning_rate: float


@dataclass
class TrainingResult:
    """MLP training loop output with best-model state metadata."""

    history: list[EpochSummary]
    best_epoch: int
    best_val_loss: float
    stopped_early: bool
    best_weights: list[Any]


def _set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def _current_learning_rate(model) -> float:
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        return float("nan")

    lr = getattr(optimizer, "learning_rate", None)
    if lr is None:
        return float("nan")

    try:
        return float(lr.numpy())  # type: ignore[call-arg]
    except Exception:
        pass

    try:
        import tensorflow as tf

        return float(tf.keras.backend.get_value(lr))
    except Exception:
        pass

    try:
        return float(lr)
    except Exception:
        return float("nan")


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


def _history_metric(history: dict[str, list[float]], key: str) -> float:
    values = history.get(key)
    if not values:
        return float("nan")
    return float(values[-1])


def _train_model(model, prepared_data, cfg: ExperimentConfig) -> TrainingResult:
    history_rows: list[EpochSummary] = []
    best_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improve = 0
    best_weights = model.get_weights()
    stopped_early = False

    for epoch in range(1, cfg.training.epochs + 1):
        fit_result = model.fit(
            prepared_data.x_train,
            prepared_data.y_train,
            validation_data=(prepared_data.x_val, prepared_data.y_val),
            epochs=1,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            verbose=cfg.training.verbose,
        )
        history = getattr(fit_result, "history", {})

        summary = EpochSummary(
            epoch=epoch,
            train_loss_mean=_history_metric(history, "loss"),
            train_rmse_mean=_history_metric(history, "rmse"),
            train_mae_mean=_history_metric(history, "mae"),
            val_loss_mean=_history_metric(history, "val_loss"),
            val_rmse_mean=_history_metric(history, "val_rmse"),
            val_mae_mean=_history_metric(history, "val_mae"),
            learning_rate=_current_learning_rate(model),
        )
        history_rows.append(summary)

        if cfg.training.verbose:
            print(
                f"Epoch {epoch:04d} | "
                f"train_loss={summary.train_loss_mean:.6f} | "
                f"train_rmse={summary.train_rmse_mean:.6f} | "
                f"train_mae={summary.train_mae_mean:.6f} | "
                f"val_loss={summary.val_loss_mean:.6f} | "
                f"val_rmse={summary.val_rmse_mean:.6f} | "
                f"val_mae={summary.val_mae_mean:.6f}"
            )

        if summary.val_loss_mean < best_val_loss:
            best_val_loss = summary.val_loss_mean
            best_epoch = epoch
            epochs_without_improve = 0
            best_weights = model.get_weights()
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= cfg.training.patience:
            if cfg.training.verbose:
                print(f"Early stopping: no improvement for {cfg.training.patience} epoch(s).")
            stopped_early = True
            break

    return TrainingResult(
        history=history_rows,
        best_epoch=best_epoch,
        best_val_loss=float(best_val_loss),
        stopped_early=stopped_early,
        best_weights=best_weights,
    )


def run_training(cfg: ExperimentConfig) -> Path:
    """Execute an MLP training run and return the output directory path."""

    cfg.validate()
    _set_reproducible_seed(cfg.training.seed)

    output_dir = Path(cfg.runtime.output_dir) / cfg.runtime.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_data = prepare_training_data(cfg)
    bounds_payload = {column: asdict(bounds) for column, bounds in prepared_data.bounds.items()}
    _write_json(output_dir / cfg.runtime.bounds_path, bounds_payload)

    model = build_mlp_model(cfg.model, num_features=len(cfg.data.features))
    train_result = _train_model(model, prepared_data, cfg)

    final_model_path = output_dir / cfg.runtime.save_final_path
    best_model_path = output_dir / cfg.runtime.save_best_path
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
        for row in train_result.history:
            writer.writerow(asdict(row))

    config_snapshot_path = output_dir / "config_snapshot.json"
    _write_json(config_snapshot_path, asdict(cfg))

    training_summary = {
        "epochs_completed": len(train_result.history),
        "best_epoch": train_result.best_epoch,
        "best_val_loss": train_result.best_val_loss,
        "stopped_early": train_result.stopped_early,
        "resumed_from_checkpoint": False,
    }
    training_summary_path = output_dir / "training_summary.json"
    _write_json(training_summary_path, training_summary)

    manifest = {
        "config_name": cfg.name,
        "h5_path": cfg.data.h5_path,
        "features": cfg.data.features,
        "target": cfg.data.target,
        "scaling_mode": cfg.data.scaling.mode,
        "model_family": "mlp",
        "hidden_layers": cfg.model.hidden_layers,
        "activation": cfg.model.activation,
        "dropout": cfg.model.dropout,
        "best_model": str(best_model_path),
        "final_model": str(final_model_path),
        "history": str(history_path),
        "bounds": str(output_dir / cfg.runtime.bounds_path),
        "config_snapshot": str(config_snapshot_path),
        "training_summary": str(training_summary_path),
        "epochs_completed": len(train_result.history),
        "row_counts": {
            "train": int(prepared_data.x_train.shape[0]),
            "val": int(prepared_data.x_val.shape[0]),
        },
        "split_counts": {
            "train_runs": len(cfg.data.train_runs),
            "val_runs": len(cfg.data.val_runs),
            "eval_runs": len(cfg.data.eval_runs),
        },
        "environment": _collect_environment_info(),
    }
    _write_json(output_dir / "run_manifest.json", manifest)

    return output_dir
