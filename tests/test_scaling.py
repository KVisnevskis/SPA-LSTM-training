from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spa_lstm.config import ColumnBounds
from spa_lstm.data.scaling import denormalize_target, load_hdf5_scaler_bounds, minmax_inverse


def test_minmax_inverse_maps_back_to_physical_range() -> None:
    bounds = ColumnBounds(lo=-3.0, hi=1.0)
    scaled = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    unscaled = minmax_inverse(scaled, bounds, out_min=-1.0, out_max=1.0)
    assert np.allclose(unscaled, np.array([-3.0, -1.0, 1.0], dtype=np.float64))


def test_denormalize_target_uses_named_target_bounds() -> None:
    bounds = {"phi": ColumnBounds(lo=-30.0, hi=30.0)}
    scaled = np.array([0.0, 0.5, 1.0], dtype=np.float32)

    unscaled = denormalize_target(scaled, "phi", bounds, out_min=0.0, out_max=1.0)
    assert np.allclose(unscaled, np.array([-30.0, 0.0, 30.0], dtype=np.float64))


def test_denormalize_target_requires_target_bounds() -> None:
    with pytest.raises(KeyError):
        denormalize_target(np.array([0.0], dtype=np.float32), "phi", bounds={})


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


def test_load_hdf5_scaler_bounds_requires_meta_table(monkeypatch) -> None:
    monkeypatch.setattr(
        "spa_lstm.data.scaling.pd.HDFStore",
        lambda *_args, **_kwargs: _FakeStore({}),
    )

    with pytest.raises(KeyError):
        load_hdf5_scaler_bounds("dummy.h5", columns=["phi"])


def test_load_hdf5_scaler_bounds_requires_expected_columns(monkeypatch) -> None:
    scaler_df = pd.DataFrame([{"column": "phi", "min": -1.0}])
    monkeypatch.setattr(
        "spa_lstm.data.scaling.pd.HDFStore",
        lambda *_args, **_kwargs: _FakeStore({"/meta/scaler_parameters": scaler_df}),
    )

    with pytest.raises(KeyError):
        load_hdf5_scaler_bounds("dummy.h5", columns=["phi"])


def test_load_hdf5_scaler_bounds_missing_requested_column_is_rejected(monkeypatch) -> None:
    scaler_df = pd.DataFrame([{"column": "pressure", "min": -1.0, "max": 1.0}])
    monkeypatch.setattr(
        "spa_lstm.data.scaling.pd.HDFStore",
        lambda *_args, **_kwargs: _FakeStore({"/meta/scaler_parameters": scaler_df}),
    )

    with pytest.raises(KeyError):
        load_hdf5_scaler_bounds("dummy.h5", columns=["phi"])
