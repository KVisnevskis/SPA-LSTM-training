"""Model evaluation workflow using configured eval runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spa_lstm.config import ColumnBounds, ExperimentConfig
from spa_lstm.data.constants import REQUIRED_BASE_COLUMNS
from spa_lstm.data.hdf5_loader import load_runs_as_dataframes
from spa_lstm.data.scaling import denormalize_target, load_hdf5_scaler_bounds
from spa_lstm.evaluation.metrics import mae, rmse
from spa_lstm.training.stateful import as_sequence_batch, reset_recurrent_states


def load_bounds_json(path: Path) -> dict[str, ColumnBounds]:
    """Load scaler bounds JSON written by training workflow."""

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected object at scaler bounds path '{path}'.")

    bounds: dict[str, ColumnBounds] = {}
    for column, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"Invalid bounds entry for column '{column}'.")
        bounds[column] = ColumnBounds(lo=float(value["lo"]), hi=float(value["hi"]))
    return bounds


def evaluate_model(cfg: ExperimentConfig, model_path: str, run_dir: str | None = None) -> Path:
    """Evaluate model on configured eval runs and return output path."""

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("TensorFlow is required for model evaluation.") from exc

    output_dir = Path(run_dir) if run_dir else Path(cfg.runtime.output_dir) / cfg.runtime.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    required_columns = tuple(dict.fromkeys(list(REQUIRED_BASE_COLUMNS) + cfg.data.features + [cfg.data.target]))
    eval_raw = load_runs_as_dataframes(cfg.data.h5_path, cfg.data.eval_runs, required_columns)
    scale_columns = list(dict.fromkeys(cfg.data.features + [cfg.data.target]))

    bounds_file = output_dir / cfg.runtime.bounds_path
    if bounds_file.exists():
        bounds = load_bounds_json(bounds_file)
    else:
        bounds = load_hdf5_scaler_bounds(cfg.data.h5_path, columns=scale_columns)

    model = tf.keras.models.load_model(model_path)

    records: list[dict[str, float | str | int]] = []
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for run_key, df in eval_raw.items():
        x = df[cfg.data.features].to_numpy(dtype=np.float32)
        y_scaled_2d = df[[cfg.data.target]].to_numpy(dtype=np.float32)
        x_stream, _ = as_sequence_batch(x, y_scaled_2d)
        y_scaled = y_scaled_2d.reshape(-1)

        reset_recurrent_states(model)
        y_hat_scaled = model.predict(x_stream, batch_size=1, verbose=0).reshape(-1)

        y_true = denormalize_target(
            y_scaled,
            cfg.data.target,
            bounds,
            out_min=cfg.data.scaling.output_min,
            out_max=cfg.data.scaling.output_max,
        )
        y_pred = denormalize_target(
            y_hat_scaled,
            cfg.data.target,
            bounds,
            out_min=cfg.data.scaling.output_min,
            out_max=cfg.data.scaling.output_max,
        )

        out_df = pd.DataFrame(
            {
                "index": np.arange(len(y_true), dtype=np.int64),
                "phi_true_deg": y_true,
                "phi_pred_deg": y_pred,
            }
        )
        if "Time" in df.columns:
            out_df["Time"] = df["Time"].to_numpy()

        out_path = predictions_dir / f"{run_key}.csv"
        out_df.to_csv(out_path, index=False)

        records.append(
            {
                "run_key": run_key,
                "n_samples": int(len(y_true)),
                "rmse_deg": rmse(y_true, y_pred),
                "mae_deg": mae(y_true, y_pred),
            }
        )

    metrics_path = output_dir / "eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    return metrics_path
