from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tables")

from spa_lstm.evaluation.error_analysis_bundle import (  # noqa: E402
    COEFFICIENT_COLUMNS,
    PREDICTION_COLUMNS,
    build_error_analysis_bundle,
    discover_experiment_configs_many,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def test_build_error_analysis_bundle_normalizes_prediction_schema(tmp_path: Path) -> None:
    repo_root = tmp_path
    config_dir = repo_root / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "demo_slu_lstm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: demo_slu_lstm",
                "data:",
                "  h5_path: outputs/preprocessed_all_trials.h5",
                "  features: [pressure]",
                "  target: phi",
                "  train_runs: [train_a]",
                "  val_runs: [val_a]",
                "  eval_runs: [eval_a]",
                "  scaling:",
                "    mode: prescaled",
                "    output_min: -1.0",
                "    output_max: 1.0",
                "model:",
                "  variant: slu_lstm",
                "  learning_rate: 0.001",
                "training:",
                "  epochs: 10",
                "  patience: 2",
                "  batch_size: 1",
                "  stateful: true",
                "  seed: 7",
                "runtime:",
                "  output_dir: artifacts",
                "  run_name: demo_slu_lstm",
            ]
        ),
        encoding="utf-8",
    )

    artifact_dir = repo_root / "artifacts" / "demo_slu_lstm"
    predictions_dir = artifact_dir / "predictions_all_runs"
    predictions_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "index": [0, 1],
            "phi_true_deg": [0.0, 0.1],
            "phi_pred_deg": [0.1, 0.2],
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
            "rmse": 0.11,
            "mae": 0.10,
            "prediction_csv": "artifacts/demo_slu_lstm/predictions_all_runs/eval_a.csv",
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
            "prediction_csv": "artifacts/demo_slu_lstm/predictions_all_runs/extra_a.csv",
        },
    ]
    _write_json(artifact_dir / "eval_metrics_all_runs.json", metrics)
    _write_json(
        artifact_dir / "eval_summary_all_runs.json",
        {
            "scope": "all",
            "by_split_role": {
                "val": {"weighted_rmse": 0.33, "weighted_mae": 0.21},
                "eval": {"weighted_rmse": 0.11, "weighted_mae": 0.10},
            },
        },
    )
    _write_json(artifact_dir / "eval_summary.json", {"scope": "eval"})
    _write_json(artifact_dir / "eval_metrics.json", [{"run_key": "eval_a"}])
    _write_json(
        artifact_dir / "training_summary.json",
        {
            "epochs_completed": 8,
            "best_epoch": 6,
            "best_val_loss": 0.0123,
            "stopped_early": True,
        },
    )
    _write_json(artifact_dir / "run_manifest.json", {"config_name": "demo_slu_lstm"})
    _write_json(artifact_dir / "config_snapshot.json", {"name": "demo_slu_lstm"})
    _write_json(artifact_dir / "scaler_bounds.json", {"phi": {"lo": -1.0, "hi": 1.0}})

    bundle_dir = repo_root / "bundle"
    result = build_error_analysis_bundle(
        config_paths=[config_path],
        bundle_dir=bundle_dir,
        repo_root=repo_root,
        store_name="demo_error_analysis.h5",
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
        assert "/coefficients/demo_slu_lstm" in store.keys()
        assert "/predictions/demo_slu_lstm/eval_a" in store.keys()
        pred = store["/predictions/demo_slu_lstm/eval_a"]
        meta_models = store["/meta/models"]
        meta_runs = store["/meta/runs"]

    assert list(pred.columns) == PREDICTION_COLUMNS
    assert pred["split"].tolist() == ["held_out", "held_out"]
    assert pred["phi_true"].tolist() == [0.0, 0.1]
    assert pred["phi_prediction"].tolist() == [0.1, 0.2]
    assert pred["phi_error"].tolist() == [0.1, 0.1]
    np.testing.assert_allclose(pred["phi_prediction_deg"].to_numpy(), np.degrees([0.1, 0.2]))
    assert meta_models.loc[0, "validation_rmse_stored_units"] == 0.33
    assert meta_models.loc[0, "held_out_rmse_stored_units"] == 0.11
    assert meta_models.loc[0, "estimator_family"] == "lstm"
    assert meta_models.loc[0, "input_group"] == "pressure"
    assert meta_models.loc[0, "packaged_coefficient_key"] == "/coefficients/demo_slu_lstm"
    assert meta_runs["split"].tolist() == ["held_out", "not_in_split"]
    assert meta_runs["packaged_prediction_hdf5_key"].tolist() == [
        "/predictions/demo_slu_lstm/eval_a",
        "/predictions/demo_slu_lstm/extra_a",
    ]

    coefficient_frame = pd.read_csv(bundle_dir / "coefficients_all_models.csv")
    assert list(coefficient_frame.columns) == COEFFICIENT_COLUMNS
    assert coefficient_frame.empty

    snapshot_dir = bundle_dir / "source_snapshots" / "demo_slu_lstm"
    assert (snapshot_dir / "demo_slu_lstm.yaml").exists()
    assert (snapshot_dir / "run_manifest.json").exists()
    assert (snapshot_dir / "training_summary.json").exists()


def test_discover_experiment_configs_many_deduplicates_and_sorts(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "zeta.yaml").write_text("name: zeta\n", encoding="utf-8")
    (dir_a / "alpha.yaml").write_text("name: alpha\n", encoding="utf-8")
    (dir_b / "beta.yaml").write_text("name: beta\n", encoding="utf-8")

    paths = discover_experiment_configs_many([dir_a, dir_b, dir_a])

    assert paths == [
        (dir_a / "alpha.yaml").resolve(),
        (dir_a / "zeta.yaml").resolve(),
        (dir_b / "beta.yaml").resolve(),
    ]
