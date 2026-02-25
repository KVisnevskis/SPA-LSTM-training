from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spa_lstm.config import ColumnBounds, ScalingConfig
from spa_lstm.data.scaling import (
    denormalize_target,
    load_hdf5_scaler_bounds,
    minmax_inverse,
    minmax_scale,
    resolve_scaling_bounds,
    scale_dataframe,
)


def test_scale_and_inverse_round_trip() -> None:
    bounds = ColumnBounds(lo=-2.0, hi=2.0)
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    scaled = minmax_scale(values, bounds, -1.0, 1.0)
    recovered = minmax_inverse(scaled, bounds, -1.0, 1.0)

    assert np.allclose(values, recovered)


def test_fit_train_bounds_respect_accelerometer_unit_conversion() -> None:
    train_runs = {
        "run_a": pd.DataFrame(
            {
                "acc_x": [-1.0, 1.0],  # values in g
                "pressure": [500.0, 700.0],
                "phi": [-20.0, 30.0],
            }
        )
    }
    scaling_cfg = ScalingConfig(mode="fit_train_only_minmax", accelerometer_in_g=True)
    bounds = resolve_scaling_bounds(
        scaling_cfg=scaling_cfg,
        train_runs=train_runs,
        feature_columns=["acc_x", "pressure"],
        target_column="phi",
    )

    assert bounds["acc_x"].lo == -9.81
    assert bounds["acc_x"].hi == 9.81


def test_fixed_bounds_mode_requires_explicit_bounds() -> None:
    cfg = ScalingConfig(mode="fixed_bounds_thesis")
    with pytest.raises(ValueError):
        resolve_scaling_bounds(
            scaling_cfg=cfg,
            train_runs={},
            feature_columns=["pressure"],
            target_column="phi",
        )


def test_prescaled_does_not_apply_forward_scaling() -> None:
    df = pd.DataFrame({"acc_x": [0.5], "pressure": [0.25], "phi": [-0.1]})
    cfg = ScalingConfig(mode="prescaled", accelerometer_in_g=True)
    out = scale_dataframe(df, cfg, bounds={}, feature_columns=["acc_x", "pressure"], target_column="phi")

    assert out.equals(df)


def test_prescaled_denormalizes_target_values() -> None:
    cfg = ScalingConfig(mode="prescaled", output_min=-1.0, output_max=1.0)
    bounds = {"phi": ColumnBounds(lo=-3.0, hi=1.0)}
    scaled = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    unscaled = denormalize_target(scaled, "phi", cfg, bounds)
    assert np.allclose(unscaled, np.array([-3.0, -1.0, 1.0]))


class _FakeStore:
    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        self._tables = tables

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ANN201
        _ = (exc_type, exc, tb)
        return False

    def keys(self) -> list[str]:
        return list(self._tables)

    def __getitem__(self, key: str) -> pd.DataFrame:
        return self._tables[key]


def test_load_hdf5_scaler_bounds_from_meta_table(monkeypatch) -> None:
    scaler_df = pd.DataFrame(
        [
            {"column": "pressure", "min": -100.0, "max": 900.0, "range": 1000.0, "is_constant": False},
            {"column": "phi", "min": -3.0, "max": 1.0, "range": 4.0, "is_constant": False},
        ]
    )
    monkeypatch.setattr(
        "spa_lstm.data.scaling.pd.HDFStore",
        lambda *_args, **_kwargs: _FakeStore({"/meta/scaler_parameters": scaler_df}),
    )

    bounds = load_hdf5_scaler_bounds("dummy.h5", columns=["pressure", "phi"])
    assert bounds["pressure"].lo == -100.0
    assert bounds["pressure"].hi == 900.0
    assert bounds["phi"].lo == -3.0
    assert bounds["phi"].hi == 1.0


def test_load_hdf5_scaler_bounds_missing_columns_are_rejected(monkeypatch) -> None:
    scaler_df = pd.DataFrame([{"column": "pressure", "min": -1.0, "max": 1.0}])
    monkeypatch.setattr(
        "spa_lstm.data.scaling.pd.HDFStore",
        lambda *_args, **_kwargs: _FakeStore({"/meta/scaler_parameters": scaler_df}),
    )

    with pytest.raises(KeyError):
        load_hdf5_scaler_bounds("dummy.h5", columns=["phi"])
