from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlp.config import DataConfig, ExperimentConfig, ModelConfig, RuntimeConfig, ScalingConfig, TrainingConfig
from mlp.data import prepare_training_data
from spa_lstm.config import ColumnBounds


def _make_cfg(tmp_path, *, train_runs=None, val_runs=None, eval_runs=None) -> ExperimentConfig:  # noqa: ANN001
    h5_path = tmp_path / "dataset.h5"
    h5_path.write_text("", encoding="utf-8")

    return ExperimentConfig(
        name="mlp_stage3_test",
        data=DataConfig(
            h5_path=str(h5_path),
            features=["pressure", "acc_x", "acc_y", "acc_z"],
            target="phi",
            train_runs=list(train_runs or ["train_a", "train_b"]),
            val_runs=list(val_runs or ["val_a"]),
            eval_runs=list(eval_runs or ["eval_a"]),
            scaling=ScalingConfig(mode="prescaled", output_min=-1.0, output_max=1.0),
        ),
        model=ModelConfig(hidden_layers=[64], activation="relu", dropout=0.0, learning_rate=1e-3),
        training=TrainingConfig(epochs=10, patience=2, batch_size=8, seed=42, verbose=0),
        runtime=RuntimeConfig(output_dir=str(tmp_path / "outputs"), run_name="mlp_stage3_test"),
    )


def _make_run_df(base_value: float, rows: int) -> pd.DataFrame:
    idx = np.arange(rows, dtype=np.float32)
    return pd.DataFrame(
        {
            "pressure": base_value + idx,
            "acc_x": base_value + 10.0 + idx,
            "acc_y": base_value + 20.0 + idx,
            "acc_z": base_value + 30.0 + idx,
            "phi": base_value + 40.0 + idx,
        }
    )


def test_prepare_training_data_concatenates_runs_in_config_order(tmp_path, monkeypatch) -> None:
    cfg = _make_cfg(tmp_path, train_runs=["train_b", "train_a"], val_runs=["val_a", "val_b"], eval_runs=["eval_a"])
    sample_runs = {
        "train_a": _make_run_df(100.0, 2),
        "train_b": _make_run_df(10.0, 3),
        "val_a": _make_run_df(200.0, 1),
        "val_b": _make_run_df(300.0, 2),
    }

    monkeypatch.setattr(
        "mlp.data.load_runs_as_dataframes",
        lambda _h5, run_keys, _required: {run_key: sample_runs[run_key].copy() for run_key in run_keys},
    )
    monkeypatch.setattr(
        "mlp.data.load_hdf5_scaler_bounds",
        lambda _h5, columns: {column: ColumnBounds(lo=-1.0, hi=1.0) for column in columns},
    )

    prepared = prepare_training_data(cfg)

    assert prepared.x_train.shape == (5, 4)
    assert prepared.y_train.shape == (5, 1)
    assert prepared.x_val.shape == (3, 4)
    assert prepared.y_val.shape == (3, 1)
    assert prepared.x_train.dtype == np.float32
    assert prepared.y_train.dtype == np.float32
    assert prepared.train_row_counts == {"train_b": 3, "train_a": 2}
    assert prepared.val_row_counts == {"val_a": 1, "val_b": 2}

    np.testing.assert_allclose(prepared.x_train[:3, 0], np.array([10.0, 11.0, 12.0], dtype=np.float32))
    np.testing.assert_allclose(prepared.x_train[3:, 0], np.array([100.0, 101.0], dtype=np.float32))
    np.testing.assert_allclose(prepared.y_val[:, 0], np.array([240.0, 340.0, 341.0], dtype=np.float32))
    assert set(prepared.bounds) == {"pressure", "acc_x", "acc_y", "acc_z", "phi"}


def test_prepare_training_data_rejects_zero_row_run(tmp_path, monkeypatch) -> None:
    cfg = _make_cfg(tmp_path, train_runs=["train_a"], val_runs=["val_a"], eval_runs=["eval_a"])
    sample_runs = {
        "train_a": _make_run_df(10.0, 0),
        "val_a": _make_run_df(20.0, 2),
    }

    monkeypatch.setattr(
        "mlp.data.load_runs_as_dataframes",
        lambda _h5, run_keys, _required: {run_key: sample_runs[run_key].copy() for run_key in run_keys},
    )
    monkeypatch.setattr(
        "mlp.data.load_hdf5_scaler_bounds",
        lambda _h5, columns: {column: ColumnBounds(lo=-1.0, hi=1.0) for column in columns},
    )

    with pytest.raises(ValueError, match="train run 'train_a' has zero rows"):
        prepare_training_data(cfg)


def test_prepare_training_data_rejects_non_finite_values(tmp_path, monkeypatch) -> None:
    cfg = _make_cfg(tmp_path, train_runs=["train_a"], val_runs=["val_a"], eval_runs=["eval_a"])
    bad_train = _make_run_df(10.0, 2)
    bad_train.loc[1, "acc_y"] = np.nan
    sample_runs = {
        "train_a": bad_train,
        "val_a": _make_run_df(20.0, 2),
    }

    monkeypatch.setattr(
        "mlp.data.load_runs_as_dataframes",
        lambda _h5, run_keys, _required: {run_key: sample_runs[run_key].copy() for run_key in run_keys},
    )
    monkeypatch.setattr(
        "mlp.data.load_hdf5_scaler_bounds",
        lambda _h5, columns: {column: ColumnBounds(lo=-1.0, hi=1.0) for column in columns},
    )

    with pytest.raises(ValueError, match="non-finite values"):
        prepare_training_data(cfg)


def test_prepare_training_data_rejects_split_overlap(tmp_path) -> None:
    cfg = _make_cfg(tmp_path, train_runs=["shared_run"], val_runs=["shared_run"], eval_runs=["eval_a"])

    with pytest.raises(ValueError, match="Split overlap detected"):
        prepare_training_data(cfg)
