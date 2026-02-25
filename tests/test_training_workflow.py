from __future__ import annotations

import json

import pandas as pd

from spa_lstm.config import (
    ColumnBounds,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RuntimeConfig,
    ScalingConfig,
    TrainingConfig,
)
from spa_lstm.training.stateful import EpochSummary, TrainingResult
from spa_lstm.training.workflow import run_training


class _FakeModel:
    def __init__(self) -> None:
        self.saved_paths: list[str] = []

    def save(self, path) -> None:  # noqa: ANN001
        path = str(path)
        self.saved_paths.append(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write("fake-model")

    def set_weights(self, weights) -> None:  # noqa: ANN001, ARG002
        return None


def _sample_runs() -> dict[str, pd.DataFrame]:
    df = pd.DataFrame(
        {
            "pressure": [500.0, 700.0, 900.0],
            "acc_x": [0.1, 0.2, 0.3],
            "acc_y": [0.2, 0.1, 0.0],
            "acc_z": [1.0, 1.1, 1.2],
            "phi": [0.0, 1.0, 2.0],
            "Time": [0.0, 0.02, 0.04],
        }
    )
    return {"train_a": df.copy(), "val_a": df.copy()}


def test_run_training_writes_required_artifacts(tmp_path, monkeypatch) -> None:
    h5_path = tmp_path / "dataset.h5"
    h5_path.write_text("", encoding="utf-8")

    cfg = ExperimentConfig(
        name="test_run",
        data=DataConfig(
            h5_path=str(h5_path),
            features=["pressure", "acc_x", "acc_y", "acc_z"],
            target="phi",
            train_runs=["train_a"],
            val_runs=["val_a"],
            eval_runs=[],
            scaling=ScalingConfig(mode="prescaled", output_min=-1.0, output_max=1.0),
        ),
        model=ModelConfig(variant="slm_lstm", learning_rate=1e-3),
        training=TrainingConfig(epochs=1, patience=1, batch_size=1, stateful=True, seed=7),
        runtime=RuntimeConfig(output_dir=str(tmp_path / "outputs"), run_name="test_run"),
    )

    fake_model = _FakeModel()
    sample_runs = _sample_runs()
    monkeypatch.setattr(
        "spa_lstm.training.workflow.load_runs_as_dataframes",
        lambda _h5, run_keys, _required: {run_key: sample_runs[run_key].copy() for run_key in run_keys},
    )
    monkeypatch.setattr(
        "spa_lstm.training.workflow.load_hdf5_scaler_bounds",
        lambda _h5, columns: {
            "pressure": ColumnBounds(lo=-100.0, hi=900.0),
            "acc_x": ColumnBounds(lo=-10.0, hi=10.0),
            "acc_y": ColumnBounds(lo=-10.0, hi=10.0),
            "acc_z": ColumnBounds(lo=-10.0, hi=10.0),
            "phi": ColumnBounds(lo=-3.0, hi=1.0),
        },
    )
    monkeypatch.setattr("spa_lstm.training.workflow.build_lstm_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(
        "spa_lstm.training.workflow.train_stateful",
        lambda **kwargs: TrainingResult(
            history=[
                EpochSummary(
                    epoch=1,
                    train_loss_mean=0.1,
                    train_rmse_mean=0.11,
                    train_mae_mean=0.12,
                    val_loss_mean=0.2,
                    val_rmse_mean=0.21,
                    val_mae_mean=0.22,
                    learning_rate=1e-3,
                )
            ],
            best_epoch=1,
            best_val_loss=0.2,
            stopped_early=False,
            best_weights=[],
        ),
    )

    out_dir = run_training(cfg)

    assert out_dir.exists()
    assert (out_dir / "best.keras").exists()
    assert (out_dir / "final.keras").exists()
    assert (out_dir / "history.csv").exists()
    assert (out_dir / "run_manifest.json").exists()
    assert (out_dir / "config_snapshot.json").exists()
    assert (out_dir / "training_summary.json").exists()
    assert (out_dir / "scaler_bounds.json").exists()

    with (out_dir / "run_manifest.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["config_name"] == "test_run"
    assert manifest["epochs_completed"] == 1
    assert manifest["scaling_mode"] == "prescaled"


def test_run_training_prescaled_skips_forward_scaling(tmp_path, monkeypatch) -> None:
    h5_path = tmp_path / "dataset.h5"
    h5_path.write_text("", encoding="utf-8")

    cfg = ExperimentConfig(
        name="test_inverse_only",
        data=DataConfig(
            h5_path=str(h5_path),
            features=["pressure", "acc_x", "acc_y", "acc_z"],
            target="phi",
            train_runs=["train_a"],
            val_runs=["val_a"],
            eval_runs=[],
            scaling=ScalingConfig(mode="prescaled", output_min=-1.0, output_max=1.0),
        ),
        model=ModelConfig(variant="slm_lstm", learning_rate=1e-3),
        training=TrainingConfig(epochs=1, patience=1, batch_size=1, stateful=True, seed=7),
        runtime=RuntimeConfig(output_dir=str(tmp_path / "outputs"), run_name="test_inverse_only"),
    )

    sample_runs = _sample_runs()
    fake_model = _FakeModel()

    monkeypatch.setattr(
        "spa_lstm.training.workflow.load_runs_as_dataframes",
        lambda _h5, run_keys, _required: {run_key: sample_runs[run_key].copy() for run_key in run_keys},
    )
    monkeypatch.setattr(
        "spa_lstm.training.workflow.load_hdf5_scaler_bounds",
        lambda _h5, columns: {
            "pressure": ColumnBounds(lo=-100.0, hi=900.0),
            "acc_x": ColumnBounds(lo=-10.0, hi=10.0),
            "acc_y": ColumnBounds(lo=-10.0, hi=10.0),
            "acc_z": ColumnBounds(lo=-10.0, hi=10.0),
            "phi": ColumnBounds(lo=-3.0, hi=1.0),
        },
    )
    monkeypatch.setattr("spa_lstm.training.workflow.build_lstm_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(
        "spa_lstm.training.workflow.train_stateful",
        lambda **kwargs: TrainingResult(
            history=[
                EpochSummary(
                    epoch=1,
                    train_loss_mean=0.1,
                    train_rmse_mean=0.11,
                    train_mae_mean=0.12,
                    val_loss_mean=0.2,
                    val_rmse_mean=0.21,
                    val_mae_mean=0.22,
                    learning_rate=1e-3,
                )
            ],
            best_epoch=1,
            best_val_loss=0.2,
            stopped_early=False,
            best_weights=[],
        ),
    )

    out_dir = run_training(cfg)
    assert out_dir.exists()
