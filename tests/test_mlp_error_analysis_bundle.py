from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tables")

from mlp.error_analysis_bundle import (  # noqa: E402
    COEFFICIENT_COLUMNS,
    PREDICTION_COLUMNS,
    build_error_analysis_bundle,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def test_build_mlp_error_analysis_bundle_exports_metadata_and_predictions(tmp_path: Path) -> None:
    repo_root = tmp_path
    config_dir = repo_root / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "demo_mlp.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: demo_mlp",
                "data:",
                "  h5_path: outputs/preprocessed_all_trials.h5",
                "  features: [pressure, acc_x, acc_y, acc_z]",
                "  target: phi",
                "  train_runs: [train_a]",
                "  val_runs: [val_a]",
                "  eval_runs: [eval_a]",
                "  scaling:",
                "    mode: prescaled",
                "    output_min: -1.0",
                "    output_max: 1.0",
                "model:",
                "  hidden_layers: [64, 32]",
                "  activation: relu",
                "  dropout: 0.0",
                "  learning_rate: 0.001",
                "training:",
                "  epochs: 200",
                "  patience: 15",
                "  batch_size: 128",
                "  seed: 42",
                "runtime:",
                "  output_dir: artifacts",
                "  run_name: demo_mlp",
                "  bounds_path: scaler_bounds.json",
            ]
        ),
        encoding="utf-8",
    )

    artifact_dir = repo_root / "artifacts" / "demo_mlp"
    predictions_dir = artifact_dir / "predictions_all_runs"
    predictions_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "index": [0, 1],
            "phi_true_deg": [0.0, 0.1],
            "phi_pred_deg": [0.05, 0.2],
            "Time": [0.0, 0.02],
        }
    ).to_csv(predictions_dir / "eval_a.csv", index=False)
    pd.DataFrame(
        {
            "index": [0, 1],
            "phi_true_deg": [-0.2, -0.1],
            "phi_pred_deg": [-0.1, 0.0],
            "Time": [0.0, 0.02],
        }
    ).to_csv(predictions_dir / "extra_a.csv", index=False)

    history_path = artifact_dir / "history.csv"
    pd.DataFrame(
        {
            "epoch": [1, 2],
            "val_loss_mean": [0.02, 0.01],
            "val_rmse_mean": [0.12, 0.08],
            "val_mae_mean": [0.10, 0.06],
        }
    ).to_csv(history_path, index=False)

    bounds_path = artifact_dir / "scaler_bounds.json"
    _write_json(bounds_path, {"phi": {"lo": -1.0, "hi": 1.0}})
    os.utime(bounds_path, (1_700_000_000, 1_700_000_000))
    os.utime(history_path, (1_700_000_030, 1_700_000_030))

    metrics = [
        {
            "run_key": "eval_a",
            "scope": "all",
            "split_role": "eval",
            "motion_type": "dynamic",
            "is_train_run": False,
            "is_val_run": False,
            "is_eval_run": True,
            "is_unseen_run": False,
            "n_samples": 2,
            "rmse": 0.09,
            "mae": 0.07,
            "prediction_csv": "artifacts/demo_mlp/predictions_all_runs/eval_a.csv",
        },
        {
            "run_key": "extra_a",
            "scope": "all",
            "split_role": "unseen",
            "motion_type": "static",
            "is_train_run": False,
            "is_val_run": False,
            "is_eval_run": False,
            "is_unseen_run": True,
            "n_samples": 2,
            "rmse": 0.22,
            "mae": 0.20,
            "prediction_csv": "artifacts/demo_mlp/predictions_all_runs/extra_a.csv",
        },
    ]
    _write_json(artifact_dir / "eval_metrics_all_runs.json", metrics)
    _write_json(
        artifact_dir / "eval_summary_all_runs.json",
        {
            "by_split_role": {
                "val": {"weighted_rmse": 0.08, "weighted_mae": 0.06},
                "eval": {"weighted_rmse": 0.09, "weighted_mae": 0.07, "n_samples": 2},
                "unseen": {"weighted_rmse": 0.22, "weighted_mae": 0.20, "n_samples": 2},
            }
        },
    )
    _write_json(
        artifact_dir / "eval_summary.json",
        {"overall": {"weighted_rmse": 0.09, "weighted_mae": 0.07, "n_samples": 2}},
    )
    _write_json(
        artifact_dir / "training_summary.json",
        {
            "epochs_completed": 25,
            "best_epoch": 2,
            "best_val_loss": 0.01,
            "stopped_early": True,
        },
    )
    _write_json(
        artifact_dir / "run_manifest.json",
        {
            "config_name": "demo_mlp",
            "row_counts": {"train": 10, "val": 4},
        },
    )
    _write_json(artifact_dir / "config_snapshot.json", {"name": "demo_mlp"})

    bundle_dir = repo_root / "bundle"
    result = build_error_analysis_bundle(
        config_paths=[config_path],
        bundle_dir=bundle_dir,
        repo_root=repo_root,
        store_name="demo_mlp_error_analysis.h5",
    )

    assert result.included_model_count == 1
    assert result.included_run_table_count == 2
    assert result.included_prediction_row_count == 4
    assert (bundle_dir / "README.md").exists()
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "models.csv").exists()
    assert (bundle_dir / "runs.csv").exists()
    assert (bundle_dir / "coefficients_all_models.csv").exists()

    with pd.HDFStore(result.bundle_store_path, mode="r") as store:
        assert "/meta/models" in store.keys()
        assert "/meta/runs" in store.keys()
        assert "/coefficients/demo_mlp" in store.keys()
        assert "/predictions/demo_mlp/eval_a" in store.keys()
        pred = store["/predictions/demo_mlp/eval_a"]
        meta_models = store["/meta/models"]
        meta_runs = store["/meta/runs"]

    assert list(pred.columns) == PREDICTION_COLUMNS
    assert pred["split"].tolist() == ["held_out", "held_out"]
    assert pred["phi_true"].tolist() == [0.0, 0.1]
    assert pred["phi_prediction"].tolist() == [0.05, 0.2]
    assert pred["phi_error"].tolist() == [0.05, 0.1]
    np.testing.assert_allclose(pred["phi_prediction_deg"].to_numpy(), np.degrees([0.05, 0.2]))

    assert meta_models.loc[0, "validation_rmse_stored_units"] == 0.08
    assert meta_models.loc[0, "held_out_rmse_stored_units"] == 0.09
    assert meta_models.loc[0, "best_val_rmse"] == 0.08
    assert meta_models.loc[0, "best_val_mae"] == 0.06
    assert meta_models.loc[0, "estimator_family"] == "mlp"
    assert meta_models.loc[0, "input_group"] == "pressure_accel"
    assert meta_models.loc[0, "training_duration_seconds_inferred"] == 30.0
    assert meta_models.loc[0, "packaged_coefficient_key"] == "/coefficients/demo_mlp"
    assert meta_runs["split"].tolist() == ["held_out", "not_in_split"]
    assert meta_runs["packaged_prediction_hdf5_key"].tolist() == [
        "/predictions/demo_mlp/eval_a",
        "/predictions/demo_mlp/extra_a",
    ]

    coefficient_frame = pd.read_csv(bundle_dir / "coefficients_all_models.csv")
    assert list(coefficient_frame.columns) == COEFFICIENT_COLUMNS
    assert coefficient_frame.empty

    snapshot_dir = bundle_dir / "source_snapshots" / "demo_mlp"
    assert (snapshot_dir / "demo_mlp.yaml").exists()
    assert (snapshot_dir / "history.csv").exists()
    assert (snapshot_dir / "training_summary.json").exists()
