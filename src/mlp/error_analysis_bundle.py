"""Build a samplewise error-analysis bundle from archived MLP artifacts."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from mlp.config import ExperimentConfig, load_experiment_config

README_TEMPLATE = """# MLP Error Analysis Bundle

This bundle packages samplewise prediction outputs for archived MLP experiment artifacts in a form intended for downstream comparative error analysis.

## Bundle Contents

- `{store_name}`: HDF5 store containing:
  - `/meta/models`: one row per packaged model
  - `/meta/runs`: one row per `(model, run)` pair
  - `/coefficients/<model_name>`: empty compatibility tables because MLP models do not expose linear coefficient tables
  - `/predictions/<model_name>/<run_id>`: samplewise prediction tables
- `models.csv`: flat model catalog including architecture, hyperparameters, convergence metadata, held-out metrics, and inferred training duration
- `runs.csv`: flat `(model, run)` catalog with split labels and per-run metrics
- `coefficients_all_models.csv`: empty compatibility table with the same header used by the basic-estimator bundle
- `source_snapshots/`: copied config and summary files from the original artifact directories, including `history.csv`
- `manifest.json`: generation metadata and traceability information

## Prediction Table Schema

Each table under `/predictions/<model_name>/<run_id>` contains:

- `model_name`
- `run_id`
- `Time`
- `split`
- `sample_index`
- `phi_true`
- `phi_prediction`
- `phi_error`
- `phi_true_deg`
- `phi_prediction_deg`
- `phi_error_deg`

The source prediction CSVs in this repository retain legacy `phi_true_deg` and `phi_pred_deg` names even though the stored values are radians. This bundle normalizes those traces into explicit radian and degree columns for downstream analysis.

## Traceability

Traceability is preserved through three mechanisms:

1. `models.csv` and `/meta/models` record the original artifact directory, source config path, selected hyperparameters, convergence metadata, and exported summary-file locations.
2. `runs.csv` and `/meta/runs` record the source prediction CSV path and packaged HDF5 key for every `(model, run)` pair.
3. `source_snapshots/` contains copies of the config, history, run manifest, training summary, scaler bounds, and evaluation summaries used to produce each exported model entry.

## Training Duration Note

The original MLP trainer did not log wall-clock training time. The `training_*_utc_inferred` and `training_duration_seconds_inferred` fields are therefore inferred from the modification timestamps of `scaler_bounds.json` and `history.csv` in each artifact directory.

## Included Models

{included_models}

## Exclusions

{exclusions}
"""

MODEL_COLUMNS = [
    "model_name",
    "artifact_dir",
    "prediction_store_path",
    "run_summary_path",
    "data_config_path",
    "model_config_path",
    "fit_intercept",
    "selected_alpha",
    "selected_lag",
    "degree",
    "feature_count",
    "raw_feature_count",
    "coefficient_count",
    "run_count",
    "prediction_row_count",
    "packaged_prediction_root",
    "packaged_coefficient_key",
    "has_validation_search",
    "validation_rmse_stored_units",
    "held_out_rmse_stored_units",
    "validation_mae_stored_units",
    "held_out_mae_stored_units",
    "validation_r2",
    "held_out_r2",
    "validation_pearson_r",
    "held_out_pearson_r",
    "estimator_family",
    "feature_expansion",
    "temporal_mode",
    "input_group",
    "source_prediction_format",
    "source_prediction_glob",
    "model_variant",
    "config_name",
    "feature_columns",
    "target_column",
    "train_runs",
    "val_runs",
    "eval_runs",
    "activation",
    "dropout",
    "scaling_mode",
    "output_min",
    "output_max",
    "learning_rate",
    "hidden_units",
    "layer_count",
    "layer_units",
    "parameter_count",
    "stateful",
    "batch_size",
    "seed",
    "training_max_epochs",
    "patience",
    "epochs_completed",
    "best_epoch",
    "best_val_loss",
    "best_val_rmse",
    "best_val_mae",
    "final_val_loss",
    "final_val_rmse",
    "final_val_mae",
    "stopped_early",
    "train_row_count",
    "val_row_count",
    "held_out_row_count",
    "unseen_row_count",
    "validation_rmse_deg",
    "held_out_rmse_deg",
    "validation_mae_deg",
    "held_out_mae_deg",
    "unseen_rmse_stored_units",
    "unseen_mae_stored_units",
    "training_start_utc_inferred",
    "training_end_utc_inferred",
    "training_duration_seconds_inferred",
    "training_duration_source",
    "config_snapshot_path",
    "history_path",
    "training_summary_path",
    "eval_summary_path",
    "eval_summary_all_runs_path",
    "source_snapshot_dir",
]

RUN_COLUMNS = [
    "model_name",
    "run_id",
    "split",
    "rows_saved",
    "source_hdf5_key",
    "source_prediction_hdf5_key",
    "packaged_prediction_hdf5_key",
    "artifact_dir",
    "prediction_store_path",
    "lag_length",
    "trimmed_initial_rows",
    "source_prediction_csv_path",
    "scope",
    "split_role",
    "motion_type",
    "is_train_run",
    "is_val_run",
    "is_eval_run",
    "is_unseen_run",
    "n_samples",
    "rmse_stored_units",
    "mae_stored_units",
    "rmse_deg",
    "mae_deg",
    "prediction_csv",
]

COEFFICIENT_COLUMNS = [
    "model_name",
    "feature",
    "feature_group",
    "coefficient",
    "abs_coefficient",
    "term_type",
    "source_features",
    "source_feature",
    "lag",
    "source_lags",
]

PREDICTION_COLUMNS = [
    "model_name",
    "run_id",
    "Time",
    "split",
    "sample_index",
    "phi_true",
    "phi_prediction",
    "phi_error",
    "phi_true_deg",
    "phi_prediction_deg",
    "phi_error_deg",
]

SNAPSHOT_FILENAMES = [
    "config_snapshot.json",
    "history.csv",
    "run_manifest.json",
    "training_summary.json",
    "scaler_bounds.json",
    "eval_metrics.json",
    "eval_metrics_all_runs.json",
    "eval_summary.json",
    "eval_summary_all_runs.json",
]


@dataclass(frozen=True)
class BundleBuildResult:
    """Summary of a generated bundle."""

    bundle_root: Path
    bundle_store_path: Path
    models_csv_path: Path
    runs_csv_path: Path
    coefficients_csv_path: Path
    manifest_path: Path
    included_model_count: int
    included_run_table_count: int
    included_prediction_row_count: int


@dataclass(frozen=True)
class _BundleArtifacts:
    bundle_root: Path
    bundle_store_path: Path
    models_csv_path: Path
    runs_csv_path: Path
    coefficients_csv_path: Path
    manifest_path: Path
    readme_path: Path
    source_snapshots_dir: Path


def build_error_analysis_bundle(
    config_paths: Iterable[str | Path],
    bundle_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    store_name: str = "mlp_error_analysis.h5",
) -> BundleBuildResult:
    """Build an MLP error-analysis bundle from experiment configs."""

    repo_root_path = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    artifacts = _prepare_bundle_artifacts(bundle_dir=bundle_dir, store_name=store_name)

    model_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    manifest_models: list[dict[str, Any]] = []
    excluded_outputs: list[dict[str, str]] = []
    included_prediction_row_count = 0

    config_path_list = sorted(Path(path).resolve() for path in config_paths)
    if not config_path_list:
        raise ValueError("No config paths were provided.")

    with pd.HDFStore(artifacts.bundle_store_path, mode="w") as store:
        for config_path in config_path_list:
            cfg = load_experiment_config(config_path)
            artifact_dir = _resolve_artifact_dir(cfg=cfg, config_path=config_path, repo_root=repo_root_path)

            try:
                model_row, model_run_rows, snapshot_files = _package_model(
                    cfg=cfg,
                    config_path=config_path,
                    artifact_dir=artifact_dir,
                    repo_root=repo_root_path,
                    source_snapshots_dir=artifacts.source_snapshots_dir,
                    store=store,
                )
            except FileNotFoundError as exc:
                excluded_outputs.append({"name": cfg.name, "reason": str(exc)})
                continue

            model_rows.append(model_row)
            run_rows.extend(model_run_rows)
            manifest_models.append(
                {
                    "model_name": cfg.name,
                    "artifact_dir": _display_path(artifact_dir, repo_root_path),
                    "prediction_store_path": _display_path(artifact_dir / "predictions_all_runs", repo_root_path),
                    "snapshot_files": [_display_path(path, repo_root_path) for path in snapshot_files],
                }
            )
            included_prediction_row_count += int(model_row["prediction_row_count"])
            _store_empty_coefficients(store=store, model_name=cfg.name)

        models_df = _ordered_dataframe(model_rows, MODEL_COLUMNS)
        runs_df = _ordered_dataframe(run_rows, RUN_COLUMNS)

        _store_table(
            store=store,
            key="/meta/models",
            frame=models_df,
            data_columns=["model_name"],
        )
        _store_table(
            store=store,
            key="/meta/runs",
            frame=runs_df,
            data_columns=["model_name", "run_id", "split"],
        )

    models_df = _ordered_dataframe(model_rows, MODEL_COLUMNS)
    runs_df = _ordered_dataframe(run_rows, RUN_COLUMNS)
    coefficients_df = _ordered_dataframe(coefficient_rows, COEFFICIENT_COLUMNS)

    models_df.to_csv(artifacts.models_csv_path, index=False)
    runs_df.to_csv(artifacts.runs_csv_path, index=False)
    coefficients_df.to_csv(artifacts.coefficients_csv_path, index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root_path),
        "bundle_root": _display_path(artifacts.bundle_root, repo_root_path),
        "bundle_store_path": _display_path(artifacts.bundle_store_path, repo_root_path),
        "included_model_count": int(len(model_rows)),
        "included_run_table_count": int(len(run_rows)),
        "included_prediction_row_count": int(included_prediction_row_count),
        "prediction_angle_units": {
            "phi_true": "radians",
            "phi_prediction": "radians",
            "phi_error": "radians",
            "phi_true_deg": "degrees",
            "phi_prediction_deg": "degrees",
            "phi_error_deg": "degrees",
        },
        "training_duration_note": (
            "Durations are inferred from scaler_bounds.json to history.csv "
            "modification timestamps because the MLP trainer did not persist wall-clock timers."
        ),
        "excluded_outputs": excluded_outputs,
        "models": manifest_models,
    }
    with artifacts.manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    artifacts.readme_path.write_text(
        README_TEMPLATE.format(
            store_name=artifacts.bundle_store_path.name,
            included_models=_format_readme_models([row["model_name"] for row in model_rows]),
            exclusions=_format_readme_exclusions(excluded_outputs),
        ),
        encoding="utf-8",
    )

    return BundleBuildResult(
        bundle_root=artifacts.bundle_root,
        bundle_store_path=artifacts.bundle_store_path,
        models_csv_path=artifacts.models_csv_path,
        runs_csv_path=artifacts.runs_csv_path,
        coefficients_csv_path=artifacts.coefficients_csv_path,
        manifest_path=artifacts.manifest_path,
        included_model_count=len(model_rows),
        included_run_table_count=len(run_rows),
        included_prediction_row_count=included_prediction_row_count,
    )


def discover_experiment_configs(config_dir: str | Path, *, glob_pattern: str = "*.yaml") -> list[Path]:
    """Return sorted experiment config paths from a directory."""

    config_dir_path = Path(config_dir)
    return sorted(path.resolve() for path in config_dir_path.glob(glob_pattern) if path.is_file())


def discover_experiment_configs_many(config_dirs: Iterable[str | Path], *, glob_pattern: str = "*.yaml") -> list[Path]:
    """Return sorted unique experiment configs collected from multiple directories."""

    seen: set[Path] = set()
    ordered: list[Path] = []
    for config_dir in config_dirs:
        for path in discover_experiment_configs(config_dir, glob_pattern=glob_pattern):
            if path in seen:
                continue
            seen.add(path)
            ordered.append(path)
    return sorted(ordered)


def _prepare_bundle_artifacts(bundle_dir: str | Path, store_name: str) -> _BundleArtifacts:
    bundle_root = Path(bundle_dir).resolve()
    bundle_root.mkdir(parents=True, exist_ok=True)

    bundle_store_path = bundle_root / store_name
    if bundle_store_path.exists():
        bundle_store_path.unlink()

    source_snapshots_dir = bundle_root / "source_snapshots"
    if source_snapshots_dir.exists():
        shutil.rmtree(source_snapshots_dir)
    source_snapshots_dir.mkdir(parents=True, exist_ok=True)

    return _BundleArtifacts(
        bundle_root=bundle_root,
        bundle_store_path=bundle_store_path,
        models_csv_path=bundle_root / "models.csv",
        runs_csv_path=bundle_root / "runs.csv",
        coefficients_csv_path=bundle_root / "coefficients_all_models.csv",
        manifest_path=bundle_root / "manifest.json",
        readme_path=bundle_root / "README.md",
        source_snapshots_dir=source_snapshots_dir,
    )


def _package_model(
    *,
    cfg: ExperimentConfig,
    config_path: Path,
    artifact_dir: Path,
    repo_root: Path,
    source_snapshots_dir: Path,
    store: pd.HDFStore,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]]:
    bounds_path = artifact_dir / cfg.runtime.bounds_path
    history_path = artifact_dir / "history.csv"
    metrics_path = artifact_dir / "eval_metrics_all_runs.json"
    predictions_dir = artifact_dir / "predictions_all_runs"
    summary_all_runs_path = artifact_dir / "eval_summary_all_runs.json"
    summary_eval_path = artifact_dir / "eval_summary.json"
    training_summary_path = artifact_dir / "training_summary.json"
    run_manifest_path = artifact_dir / "run_manifest.json"
    config_snapshot_path = artifact_dir / "config_snapshot.json"

    required_paths = [
        bounds_path,
        history_path,
        metrics_path,
        predictions_dir,
        summary_all_runs_path,
        training_summary_path,
        run_manifest_path,
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required artifact(s): {', '.join(_display_path(path, repo_root) for path in missing)}"
        )

    metrics = _load_json(metrics_path)
    if not isinstance(metrics, list):
        raise ValueError(f"Expected a list in '{metrics_path}'.")

    summary_all_runs = _load_json(summary_all_runs_path)
    summary_eval = _load_json(summary_eval_path) if summary_eval_path.exists() else {}
    training_summary = _load_json(training_summary_path)
    run_manifest = _load_json(run_manifest_path)
    history_df = pd.read_csv(history_path)

    snapshot_files = _copy_source_snapshots(
        model_name=cfg.name,
        config_path=config_path,
        artifact_dir=artifact_dir,
        source_snapshots_dir=source_snapshots_dir,
    )

    run_rows: list[dict[str, Any]] = []
    prediction_row_count = 0
    for metric_row in metrics:
        if not isinstance(metric_row, dict):
            raise ValueError(f"Expected object rows in '{metrics_path}'.")
        packaged_df, run_row = _package_prediction_run(
            cfg=cfg,
            metric_row=metric_row,
            artifact_dir=artifact_dir,
            predictions_dir=predictions_dir,
            repo_root=repo_root,
        )
        packaged_key = str(run_row["packaged_prediction_hdf5_key"])
        _store_table(
            store=store,
            key=packaged_key,
            frame=packaged_df,
            data_columns=["model_name", "run_id", "split"],
        )
        prediction_row_count += int(len(packaged_df))
        run_rows.append(run_row)

    split_summary = summary_all_runs.get("by_split_role", {}) if isinstance(summary_all_runs, dict) else {}
    validation_summary = split_summary.get("val", {})
    held_out_summary = summary_eval.get("overall", split_summary.get("eval", {})) if isinstance(summary_eval, dict) else {}
    unseen_summary = split_summary.get("unseen", {})

    best_epoch = _json_value(training_summary, "best_epoch")
    best_history_row = _history_row_for_best_epoch(history_df, best_epoch)
    final_history_row = history_df.iloc[-1] if not history_df.empty else pd.Series(dtype=float)

    timing = _infer_training_timing(bounds_path=bounds_path, history_path=history_path)
    row_counts = run_manifest.get("row_counts", {}) if isinstance(run_manifest, dict) else {}
    layer_units = list(cfg.model.hidden_layers)

    model_row: dict[str, Any] = {
        "model_name": cfg.name,
        "artifact_dir": _display_path(artifact_dir, repo_root),
        "prediction_store_path": _display_path(predictions_dir, repo_root),
        "run_summary_path": _display_path(run_manifest_path, repo_root),
        "data_config_path": _display_path(config_path, repo_root),
        "model_config_path": _display_path(config_path, repo_root),
        "fit_intercept": pd.NA,
        "selected_alpha": pd.NA,
        "selected_lag": pd.NA,
        "degree": pd.NA,
        "feature_count": len(cfg.data.features),
        "raw_feature_count": len(cfg.data.features),
        "coefficient_count": 0,
        "run_count": len(run_rows),
        "prediction_row_count": prediction_row_count,
        "packaged_prediction_root": f"/predictions/{cfg.name}",
        "packaged_coefficient_key": f"/coefficients/{cfg.name}",
        "has_validation_search": False,
        "validation_rmse_stored_units": _summary_value(validation_summary, "weighted_rmse"),
        "held_out_rmse_stored_units": _summary_value(held_out_summary, "weighted_rmse"),
        "validation_mae_stored_units": _summary_value(validation_summary, "weighted_mae"),
        "held_out_mae_stored_units": _summary_value(held_out_summary, "weighted_mae"),
        "validation_r2": pd.NA,
        "held_out_r2": pd.NA,
        "validation_pearson_r": pd.NA,
        "held_out_pearson_r": pd.NA,
        "estimator_family": "mlp",
        "feature_expansion": "none",
        "temporal_mode": "row_wise",
        "input_group": _infer_input_group(cfg.data.features),
        "source_prediction_format": "csv_directory",
        "source_prediction_glob": "*.csv",
        "model_variant": "mlp",
        "config_name": cfg.name,
        "feature_columns": ", ".join(cfg.data.features),
        "target_column": cfg.data.target,
        "train_runs": ", ".join(cfg.data.train_runs),
        "val_runs": ", ".join(cfg.data.val_runs),
        "eval_runs": ", ".join(cfg.data.eval_runs),
        "activation": cfg.model.activation,
        "dropout": cfg.model.dropout,
        "scaling_mode": cfg.data.scaling.mode,
        "output_min": cfg.data.scaling.output_min,
        "output_max": cfg.data.scaling.output_max,
        "learning_rate": cfg.model.learning_rate,
        "hidden_units": layer_units[0],
        "layer_count": len(layer_units),
        "layer_units": ", ".join(str(unit) for unit in layer_units),
        "parameter_count": _mlp_parameter_count(len(cfg.data.features), layer_units),
        "stateful": False,
        "batch_size": cfg.training.batch_size,
        "seed": cfg.training.seed,
        "training_max_epochs": cfg.training.epochs,
        "patience": cfg.training.patience,
        "epochs_completed": _json_value(training_summary, "epochs_completed"),
        "best_epoch": best_epoch,
        "best_val_loss": _json_value(training_summary, "best_val_loss"),
        "best_val_rmse": _history_value(best_history_row, "val_rmse_mean"),
        "best_val_mae": _history_value(best_history_row, "val_mae_mean"),
        "final_val_loss": _history_value(final_history_row, "val_loss_mean"),
        "final_val_rmse": _history_value(final_history_row, "val_rmse_mean"),
        "final_val_mae": _history_value(final_history_row, "val_mae_mean"),
        "stopped_early": _json_value(training_summary, "stopped_early"),
        "train_row_count": _json_value(row_counts, "train"),
        "val_row_count": _json_value(row_counts, "val"),
        "held_out_row_count": _summary_value(held_out_summary, "n_samples"),
        "unseen_row_count": _summary_value(unseen_summary, "n_samples"),
        "validation_rmse_deg": _degrees_value(_summary_value(validation_summary, "weighted_rmse")),
        "held_out_rmse_deg": _degrees_value(_summary_value(held_out_summary, "weighted_rmse")),
        "validation_mae_deg": _degrees_value(_summary_value(validation_summary, "weighted_mae")),
        "held_out_mae_deg": _degrees_value(_summary_value(held_out_summary, "weighted_mae")),
        "unseen_rmse_stored_units": _summary_value(unseen_summary, "weighted_rmse"),
        "unseen_mae_stored_units": _summary_value(unseen_summary, "weighted_mae"),
        "training_start_utc_inferred": timing["start"],
        "training_end_utc_inferred": timing["end"],
        "training_duration_seconds_inferred": timing["duration_seconds"],
        "training_duration_source": timing["source"],
        "config_snapshot_path": (
            _display_path(config_snapshot_path, repo_root) if config_snapshot_path.exists() else pd.NA
        ),
        "history_path": _display_path(history_path, repo_root),
        "training_summary_path": _display_path(training_summary_path, repo_root),
        "eval_summary_path": _display_path(summary_eval_path, repo_root) if summary_eval_path.exists() else pd.NA,
        "eval_summary_all_runs_path": _display_path(summary_all_runs_path, repo_root),
        "source_snapshot_dir": _display_path(source_snapshots_dir / cfg.name, repo_root),
    }
    return model_row, run_rows, snapshot_files


def _package_prediction_run(
    *,
    cfg: ExperimentConfig,
    metric_row: dict[str, Any],
    artifact_dir: Path,
    predictions_dir: Path,
    repo_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_id = str(metric_row["run_key"])
    prediction_csv = _resolve_prediction_csv(metric_row=metric_row, predictions_dir=predictions_dir, repo_root=repo_root)
    raw_df = pd.read_csv(prediction_csv)
    split_role = str(metric_row.get("split_role", "unseen"))
    split = _bundle_split_label(split_role)
    packaged_df = _normalize_prediction_frame(
        raw_df=raw_df,
        model_name=cfg.name,
        run_id=run_id,
        split=split,
    )
    packaged_prediction_key = f"/predictions/{cfg.name}/{run_id}"

    rmse_value = metric_row.get("rmse")
    mae_value = metric_row.get("mae")
    run_row = {
        "model_name": cfg.name,
        "run_id": run_id,
        "split": split,
        "rows_saved": int(len(packaged_df)),
        "source_hdf5_key": f"/runs/{run_id}",
        "source_prediction_hdf5_key": pd.NA,
        "packaged_prediction_hdf5_key": packaged_prediction_key,
        "artifact_dir": _display_path(artifact_dir, repo_root),
        "prediction_store_path": _display_path(predictions_dir, repo_root),
        "lag_length": pd.NA,
        "trimmed_initial_rows": pd.NA,
        "source_prediction_csv_path": _display_path(prediction_csv, repo_root),
        "scope": metric_row.get("scope", "all"),
        "split_role": split_role,
        "motion_type": metric_row.get("motion_type", pd.NA),
        "is_train_run": metric_row.get("is_train_run", False),
        "is_val_run": metric_row.get("is_val_run", False),
        "is_eval_run": metric_row.get("is_eval_run", False),
        "is_unseen_run": metric_row.get("is_unseen_run", False),
        "n_samples": int(metric_row.get("n_samples", len(packaged_df))),
        "rmse_stored_units": rmse_value,
        "mae_stored_units": mae_value,
        "rmse_deg": _degrees_value(rmse_value),
        "mae_deg": _degrees_value(mae_value),
        "prediction_csv": metric_row.get("prediction_csv", _display_path(prediction_csv, repo_root)),
    }
    return packaged_df, run_row


def _resolve_artifact_dir(*, cfg: ExperimentConfig, config_path: Path, repo_root: Path) -> Path:
    config_dir = config_path.parent
    output_dir = Path(cfg.runtime.output_dir)
    if not output_dir.is_absolute():
        candidate = (config_dir / output_dir / cfg.runtime.run_name).resolve()
        if candidate.exists():
            return candidate
        return (repo_root / output_dir / cfg.runtime.run_name).resolve()
    return (output_dir / cfg.runtime.run_name).resolve()


def _resolve_prediction_csv(*, metric_row: dict[str, Any], predictions_dir: Path, repo_root: Path) -> Path:
    prediction_csv = metric_row.get("prediction_csv")
    if isinstance(prediction_csv, str) and prediction_csv.strip():
        candidate = Path(prediction_csv)
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        if candidate.exists():
            return candidate

    fallback = predictions_dir / f"{metric_row['run_key']}.csv"
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"Missing prediction CSV for run '{metric_row['run_key']}'.")


def _normalize_prediction_frame(*, raw_df: pd.DataFrame, model_name: str, run_id: str, split: str) -> pd.DataFrame:
    required = {"phi_true_deg", "phi_pred_deg"}
    missing = required - set(raw_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Prediction CSV for '{run_id}' is missing column(s): {missing_str}.")

    phi_true = raw_df["phi_true_deg"].astype(float).to_numpy()
    phi_prediction = raw_df["phi_pred_deg"].astype(float).to_numpy()
    phi_error = phi_prediction - phi_true

    if "index" in raw_df.columns:
        sample_index = raw_df["index"].astype(int).to_numpy()
    else:
        sample_index = np.arange(len(raw_df), dtype=np.int64)

    if "Time" in raw_df.columns:
        time_values = raw_df["Time"].astype(float).to_numpy()
    else:
        time_values = np.full(len(raw_df), np.nan, dtype=float)

    frame = pd.DataFrame(
        {
            "model_name": model_name,
            "run_id": run_id,
            "Time": time_values,
            "split": split,
            "sample_index": sample_index,
            "phi_true": phi_true,
            "phi_prediction": phi_prediction,
            "phi_error": phi_error,
            "phi_true_deg": np.degrees(phi_true),
            "phi_prediction_deg": np.degrees(phi_prediction),
            "phi_error_deg": np.degrees(phi_error),
        }
    )
    return frame[PREDICTION_COLUMNS]


def _bundle_split_label(split_role: str) -> str:
    mapping = {
        "train": "train",
        "val": "val",
        "eval": "held_out",
        "unseen": "not_in_split",
        "overlap": "overlap",
    }
    return mapping.get(split_role, split_role)


def _copy_source_snapshots(
    *,
    model_name: str,
    config_path: Path,
    artifact_dir: Path,
    source_snapshots_dir: Path,
) -> list[Path]:
    snapshot_dir = source_snapshots_dir / model_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    config_copy = snapshot_dir / config_path.name
    shutil.copy2(config_path, config_copy)
    copied.append(config_copy)

    for filename in SNAPSHOT_FILENAMES:
        src = artifact_dir / filename
        if not src.exists():
            continue
        dst = snapshot_dir / filename
        shutil.copy2(src, dst)
        copied.append(dst)

    return copied


def _store_empty_coefficients(*, store: pd.HDFStore, model_name: str) -> None:
    frame = pd.DataFrame(columns=COEFFICIENT_COLUMNS)
    store.put(f"/coefficients/{model_name}", frame, format="fixed")


def _store_table(
    *,
    store: pd.HDFStore,
    key: str,
    frame: pd.DataFrame,
    data_columns: list[str] | bool,
    min_itemsize: dict[str, int] | None = None,
) -> None:
    store.put(
        key,
        frame,
        format="table",
        data_columns=data_columns,
        min_itemsize=min_itemsize,
    )


def _ordered_dataframe(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path.resolve())


def _summary_value(summary: Any, key: str) -> Any:
    if isinstance(summary, dict):
        return summary.get(key, pd.NA)
    return pd.NA


def _json_value(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, pd.NA)
    return pd.NA


def _history_row_for_best_epoch(history_df: pd.DataFrame, best_epoch: Any) -> pd.Series:
    if history_df.empty:
        return pd.Series(dtype=float)
    if best_epoch is pd.NA or best_epoch is None:
        return history_df.iloc[int(history_df["val_loss_mean"].astype(float).idxmin())]

    epoch_mask = history_df["epoch"].astype(int) == int(best_epoch)
    if epoch_mask.any():
        return history_df.loc[epoch_mask].iloc[0]
    return history_df.iloc[int(history_df["val_loss_mean"].astype(float).idxmin())]


def _history_value(row: pd.Series, key: str) -> Any:
    if key not in row:
        return pd.NA
    value = row[key]
    if pd.isna(value):
        return pd.NA
    return float(value)


def _infer_training_timing(*, bounds_path: Path, history_path: Path) -> dict[str, Any]:
    start_ts = bounds_path.stat().st_mtime
    end_ts = history_path.stat().st_mtime
    duration = max(0.0, end_ts - start_ts)
    return {
        "start": datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
        "end": datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
        "duration_seconds": float(duration),
        "source": "filesystem_mtime:scaler_bounds.json->history.csv",
    }


def _degrees_value(value: Any) -> Any:
    try:
        return float(np.degrees(float(value)))
    except Exception:
        return pd.NA


def _mlp_parameter_count(num_features: int, hidden_layers: list[int]) -> int:
    total = 0
    input_width = num_features
    for units in hidden_layers:
        total += (input_width * units) + units
        input_width = units
    total += input_width + 1
    return int(total)


def _infer_input_group(features: list[str]) -> str:
    normalized = tuple(features)
    if normalized == ("pressure",):
        return "pressure"
    if normalized == ("pressure", "acc_x", "acc_y", "acc_z"):
        return "pressure_accel"
    if all(feature.startswith("acc_") for feature in normalized):
        return "accel"
    return "custom"


def _format_readme_models(model_names: list[str]) -> str:
    if not model_names:
        return "- None"
    return "\n".join(f"- `{name}`" for name in model_names)


def _format_readme_exclusions(excluded_outputs: list[dict[str, str]]) -> str:
    if not excluded_outputs:
        return "- None"
    lines = []
    for item in excluded_outputs:
        lines.append(f"- `{item['name']}`: {item['reason']}")
    return "\n".join(lines)
