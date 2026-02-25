from __future__ import annotations

import pandas as pd
import pytest

from spa_lstm.data.hdf5_loader import list_run_keys, load_runs_as_dataframes


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


def _fake_tables() -> dict[str, pd.DataFrame]:
    df = pd.DataFrame(
        {
            "pressure": [500.0, 501.0],
            "acc_x": [0.1, 0.2],
            "acc_y": [0.0, 0.1],
            "acc_z": [1.0, 1.1],
            "phi": [0.0, 1.0],
        }
    )
    return {
        "/runs/freehand_tt_1": df,
        "/runs/run_0roll_0pitch_tt_1": df,
        "/meta/runs": pd.DataFrame({"run_key": ["freehand_tt_1", "run_0roll_0pitch_tt_1"]}),
    }


def test_list_run_keys_returns_runs_namespace_keys(monkeypatch) -> None:
    monkeypatch.setattr(
        "spa_lstm.data.hdf5_loader.pd.HDFStore",
        lambda *_args, **_kwargs: _FakeStore(_fake_tables()),
    )

    keys = list_run_keys("dummy.h5")
    assert keys == ["freehand_tt_1", "run_0roll_0pitch_tt_1"]


def test_load_runs_requires_exact_run_keys(monkeypatch) -> None:
    monkeypatch.setattr(
        "spa_lstm.data.hdf5_loader.pd.HDFStore",
        lambda *_args, **_kwargs: _FakeStore(_fake_tables()),
    )

    runs = load_runs_as_dataframes(
        "dummy.h5",
        run_keys=["freehand_tt_1", "run_0roll_0pitch_tt_1"],
        required_columns=["pressure", "acc_x", "acc_y", "acc_z", "phi"],
    )
    assert set(runs) == {"freehand_tt_1", "run_0roll_0pitch_tt_1"}
    assert len(runs["freehand_tt_1"]) == 2


def test_load_runs_missing_key_error_contains_expected_layout_hint(monkeypatch) -> None:
    monkeypatch.setattr(
        "spa_lstm.data.hdf5_loader.pd.HDFStore",
        lambda *_args, **_kwargs: _FakeStore(_fake_tables()),
    )

    with pytest.raises(KeyError) as exc:
        load_runs_as_dataframes(
            "dummy.h5",
            run_keys=["sfreehand_tt_9"],
            required_columns=["pressure", "acc_x", "acc_y", "acc_z", "phi"],
        )

    message = str(exc.value)
    assert "Expected key '/runs/sfreehand_tt_9' under /runs/" in message
    assert "Available /runs keys (first 10)" in message
    assert "freehand_tt_1" in message


def test_missing_pytables_dependency_raises_clear_error(monkeypatch) -> None:
    def _raise_import_error(*_args, **_kwargs):  # noqa: ANN202
        raise ImportError("missing tables")

    monkeypatch.setattr("spa_lstm.data.hdf5_loader.pd.HDFStore", _raise_import_error)

    with pytest.raises(RuntimeError) as exc:
        list_run_keys("dummy.h5")

    assert "PyTables dependency is missing" in str(exc.value)
