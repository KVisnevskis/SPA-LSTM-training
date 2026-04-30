from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np

from mlp.config import DataConfig, ExperimentConfig, ModelConfig, RuntimeConfig, ScalingConfig, TrainingConfig
from mlp.data import PreparedTrainingData
from mlp.training import run_training
from spa_lstm.config import ColumnBounds


class _FakeModel:
    def __init__(self, epochs: list[dict[str, float]]) -> None:
        self._epochs = list(epochs)
        self._fit_index = 0
        self._weight = 0.0
        self.saved_paths: list[str] = []
        self.optimizer = SimpleNamespace(learning_rate=0.001)

    def fit(self, x, y, *, validation_data, epochs, batch_size, shuffle, verbose):  # noqa: ANN001, ANN201
        _ = (x, y, validation_data, epochs, batch_size, shuffle, verbose)
        epoch_metrics = self._epochs[self._fit_index]
        self._fit_index += 1
        self._weight = float(self._fit_index)
        return SimpleNamespace(
            history={
                "loss": [epoch_metrics["loss"]],
                "rmse": [epoch_metrics["rmse"]],
                "mae": [epoch_metrics["mae"]],
                "val_loss": [epoch_metrics["val_loss"]],
                "val_rmse": [epoch_metrics["val_rmse"]],
                "val_mae": [epoch_metrics["val_mae"]],
            }
        )

    def get_weights(self):  # noqa: ANN201
        return [np.array([self._weight], dtype=np.float32)]

    def set_weights(self, weights) -> None:  # noqa: ANN001
        self._weight = float(weights[0][0])

    def save(self, path) -> None:  # noqa: ANN001
        path = str(path)
        self.saved_paths.append(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{self._weight:.1f}")


def _make_cfg(tmp_path, *, epochs: int = 3, patience: int = 5) -> ExperimentConfig:  # noqa: ANN001
    return ExperimentConfig(
        name="baseline_mlp",
        data=DataConfig(
            h5_path=str(tmp_path / "dataset.h5"),
            features=["pressure", "acc_x", "acc_y", "acc_z"],
            target="phi",
            train_runs=["train_a"],
            val_runs=["val_a"],
            eval_runs=["eval_a"],
            scaling=ScalingConfig(mode="prescaled", output_min=-1.0, output_max=1.0),
        ),
        model=ModelConfig(hidden_layers=[64], activation="relu", dropout=0.0, learning_rate=1e-3),
        training=TrainingConfig(epochs=epochs, patience=patience, batch_size=8, seed=42, verbose=0),
        runtime=RuntimeConfig(output_dir=str(tmp_path / "outputs"), run_name="baseline_mlp"),
    )


def _prepared_data() -> PreparedTrainingData:
    return PreparedTrainingData(
        x_train=np.zeros((4, 4), dtype=np.float32),
        y_train=np.zeros((4, 1), dtype=np.float32),
        x_val=np.zeros((2, 4), dtype=np.float32),
        y_val=np.zeros((2, 1), dtype=np.float32),
        bounds={
            "pressure": ColumnBounds(lo=0.0, hi=1.0),
            "acc_x": ColumnBounds(lo=0.0, hi=1.0),
            "acc_y": ColumnBounds(lo=0.0, hi=1.0),
            "acc_z": ColumnBounds(lo=0.0, hi=1.0),
            "phi": ColumnBounds(lo=0.0, hi=1.0),
        },
        train_row_counts={"train_a": 4},
        val_row_counts={"val_a": 2},
    )


def test_run_training_writes_required_artifacts_and_tracks_best_epoch(tmp_path, monkeypatch) -> None:
    cfg = _make_cfg(tmp_path, epochs=3, patience=5)
    fake_model = _FakeModel(
        [
            {"loss": 0.5, "rmse": 0.6, "mae": 0.7, "val_loss": 0.4, "val_rmse": 0.5, "val_mae": 0.6},
            {"loss": 0.3, "rmse": 0.4, "mae": 0.5, "val_loss": 0.2, "val_rmse": 0.3, "val_mae": 0.4},
            {"loss": 0.25, "rmse": 0.35, "mae": 0.45, "val_loss": 0.25, "val_rmse": 0.33, "val_mae": 0.43},
        ]
    )

    monkeypatch.setattr("mlp.training.prepare_training_data", lambda _cfg: _prepared_data())
    monkeypatch.setattr("mlp.training.build_mlp_model", lambda *_args, **_kwargs: fake_model)

    out_dir = run_training(cfg)

    assert out_dir.exists()
    assert (out_dir / "best.keras").exists()
    assert (out_dir / "final.keras").exists()
    assert (out_dir / "history.csv").exists()
    assert (out_dir / "run_manifest.json").exists()
    assert (out_dir / "config_snapshot.json").exists()
    assert (out_dir / "training_summary.json").exists()
    assert (out_dir / "scaler_bounds.json").exists()

    assert (out_dir / "final.keras").read_text(encoding="utf-8") == "3.0"
    assert (out_dir / "best.keras").read_text(encoding="utf-8") == "2.0"

    with (out_dir / "training_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["epochs_completed"] == 3
    assert summary["best_epoch"] == 2
    assert summary["best_val_loss"] == 0.2
    assert summary["stopped_early"] is False

    with (out_dir / "run_manifest.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["config_name"] == "baseline_mlp"
    assert manifest["model_family"] == "mlp"
    assert manifest["row_counts"] == {"train": 4, "val": 2}

    with (out_dir / "history.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[1]["val_loss_mean"] == "0.2"


def test_run_training_stops_early_when_patience_is_exhausted(tmp_path, monkeypatch) -> None:
    cfg = _make_cfg(tmp_path, epochs=5, patience=1)
    fake_model = _FakeModel(
        [
            {"loss": 0.5, "rmse": 0.6, "mae": 0.7, "val_loss": 0.4, "val_rmse": 0.5, "val_mae": 0.6},
            {"loss": 0.45, "rmse": 0.55, "mae": 0.65, "val_loss": 0.45, "val_rmse": 0.52, "val_mae": 0.62},
        ]
    )

    monkeypatch.setattr("mlp.training.prepare_training_data", lambda _cfg: _prepared_data())
    monkeypatch.setattr("mlp.training.build_mlp_model", lambda *_args, **_kwargs: fake_model)

    out_dir = run_training(cfg)

    with (out_dir / "training_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["epochs_completed"] == 2
    assert summary["best_epoch"] == 1
    assert summary["stopped_early"] is True
    assert (out_dir / "final.keras").read_text(encoding="utf-8") == "2.0"
    assert (out_dir / "best.keras").read_text(encoding="utf-8") == "1.0"
