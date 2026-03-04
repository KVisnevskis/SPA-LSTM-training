"""Evaluation workflow for configured eval runs or all dataset runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from spa_lstm.config import ColumnBounds, ExperimentConfig
from spa_lstm.data.constants import REQUIRED_BASE_COLUMNS
from spa_lstm.data.hdf5_loader import list_run_keys, load_runs_as_dataframes
from spa_lstm.data.scaling import denormalize_target, load_hdf5_scaler_bounds
from spa_lstm.evaluation.metrics import mae, rmse
from spa_lstm.training.stateful import as_sequence_batch, reset_recurrent_states

EvaluationScope = Literal["eval", "all"]


def load_bounds_json(path: Path) -> dict[str, ColumnBounds]:
    """Load scaler bounds JSON written by training workflow."""

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object at '{path}'.")

    bounds: dict[str, ColumnBounds] = {}
    for column, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"Invalid bounds entry for column '{column}'.")
        bounds[column] = ColumnBounds(lo=float(value["lo"]), hi=float(value["hi"]))
    return bounds


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve_run_keys(cfg: ExperimentConfig, scope: EvaluationScope) -> list[str]:
    if scope == "eval":
        return list(cfg.data.eval_runs)
    if scope == "all":
        return list_run_keys(cfg.data.h5_path)
    raise ValueError(f"Unsupported evaluation scope '{scope}'. Expected 'eval' or 'all'.")


def _resolve_bounds(cfg: ExperimentConfig, output_dir: Path) -> dict[str, ColumnBounds]:
    scale_columns = list(dict.fromkeys(cfg.data.features + [cfg.data.target]))
    bounds_file = output_dir / cfg.runtime.bounds_path
    if bounds_file.exists():
        return load_bounds_json(bounds_file)
    return load_hdf5_scaler_bounds(cfg.data.h5_path, columns=scale_columns)


def _split_role(run_key: str, cfg: ExperimentConfig) -> tuple[str, bool, bool, bool, bool]:
    is_train = run_key in set(cfg.data.train_runs)
    is_val = run_key in set(cfg.data.val_runs)
    is_eval = run_key in set(cfg.data.eval_runs)
    memberships = [is_train, is_val, is_eval]
    if sum(memberships) > 1:
        role = "overlap"
    elif is_train:
        role = "train"
    elif is_val:
        role = "val"
    elif is_eval:
        role = "eval"
    else:
        role = "unseen"
    return role, is_train, is_val, is_eval, role == "unseen"


def _motion_type(run_key: str) -> str:
    return "static" if "static" in run_key.lower() else "dynamic"


def _aggregate(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        return {"n_runs": 0, "n_samples": 0, "weighted_rmse": float("nan"), "weighted_mae": float("nan")}

    n_samples = sum(int(row["n_samples"]) for row in records)
    if n_samples <= 0:
        return {"n_runs": len(records), "n_samples": 0, "weighted_rmse": float("nan"), "weighted_mae": float("nan")}

    rmse_weighted = np.sqrt(sum((float(row["rmse"]) ** 2) * int(row["n_samples"]) for row in records) / n_samples)
    mae_weighted = sum(float(row["mae"]) * int(row["n_samples"]) for row in records) / n_samples
    return {
        "n_runs": len(records),
        "n_samples": int(n_samples),
        "weighted_rmse": float(rmse_weighted),
        "weighted_mae": float(mae_weighted),
    }


def evaluate_model(
    cfg: ExperimentConfig,
    model_path: str,
    run_dir: str | None = None,
    scope: EvaluationScope = "eval",
) -> Path:
    """Run model inference and write metrics for the requested scope.

    Parameters
    ----------
    cfg:
        Experiment config.
    model_path:
        Saved Keras model path.
    run_dir:
        Output directory override. Defaults to `<cfg.runtime.output_dir>/<cfg.runtime.run_name>`.
    scope:
        `eval` to evaluate configured `data.eval_runs`; `all` to evaluate every run under `/runs/*`.
    """

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("TensorFlow is required for model evaluation.") from exc

    output_dir = Path(run_dir) if run_dir else Path(cfg.runtime.output_dir) / cfg.runtime.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    run_keys = _resolve_run_keys(cfg, scope)
    if not run_keys:
        raise ValueError(f"No run keys found for evaluation scope '{scope}'.")

    required_columns = tuple(dict.fromkeys(list(REQUIRED_BASE_COLUMNS) + cfg.data.features + [cfg.data.target]))
    runs = load_runs_as_dataframes(cfg.data.h5_path, run_keys, required_columns)
    bounds = _resolve_bounds(cfg, output_dir)

    model = tf.keras.models.load_model(model_path)

    predictions_dir_name = "predictions" if scope == "eval" else "predictions_all_runs"
    metrics_name = "eval_metrics.json" if scope == "eval" else "eval_metrics_all_runs.json"
    summary_name = "eval_summary.json" if scope == "eval" else "eval_summary_all_runs.json"
    predictions_dir = output_dir / predictions_dir_name
    predictions_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for run_key in run_keys:
        df = runs[run_key]
        x_2d = df[cfg.data.features].to_numpy(dtype=np.float32)
        y_scaled_2d = df[[cfg.data.target]].to_numpy(dtype=np.float32)
        x_stream, _ = as_sequence_batch(x_2d, y_scaled_2d)
        y_scaled = y_scaled_2d.reshape(-1)

        reset_recurrent_states(model)
        y_hat_scaled = model.predict(x_stream, batch_size=1, verbose=0).reshape(-1)
        if len(y_hat_scaled) != len(y_scaled):
            raise ValueError(
                f"Prediction length mismatch for run '{run_key}': "
                f"pred={len(y_hat_scaled)} target={len(y_scaled)}."
            )

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

        # Keep historical column names used elsewhere in this repo.
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

        split_role, is_train, is_val, is_eval, is_unseen = _split_role(run_key, cfg)
        run_rmse = rmse(y_true, y_pred)
        run_mae = mae(y_true, y_pred)
        records.append(
            {
                "run_key": run_key,
                "scope": scope,
                "split_role": split_role,
                "motion_type": _motion_type(run_key),
                "is_train_run": is_train,
                "is_val_run": is_val,
                "is_eval_run": is_eval,
                "is_unseen_run": is_unseen,
                "n_samples": int(len(y_true)),
                "rmse": float(run_rmse),
                "mae": float(run_mae),
                # Legacy names kept for compatibility with existing viewer code.
                "rmse_deg": float(run_rmse),
                "mae_deg": float(run_mae),
                "prediction_csv": str(out_path),
            }
        )

    metrics_path = output_dir / metrics_name
    _write_json(metrics_path, records)

    by_split: dict[str, list[dict[str, Any]]] = {}
    by_motion: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_split.setdefault(str(row["split_role"]), []).append(row)
        by_motion.setdefault(str(row["motion_type"]), []).append(row)

    summary = {
        "scope": scope,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "predictions_dir": str(predictions_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall": _aggregate(records),
        "by_split_role": {key: _aggregate(rows) for key, rows in sorted(by_split.items())},
        "by_motion_type": {key: _aggregate(rows) for key, rows in sorted(by_motion.items())},
    }
    _write_json(output_dir / summary_name, summary)

    return metrics_path
