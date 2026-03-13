from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spa_lstm.config import DataConfig, ExperimentConfig, ModelConfig, RuntimeConfig, ScalingConfig
from spa_lstm.evaluation import workflow as wf


def _make_cfg(*, eval_runs: list[str] | None = None) -> ExperimentConfig:
    return ExperimentConfig(
        name="eval_helper_cfg",
        data=DataConfig(
            h5_path="dummy.h5",
            features=["pressure", "acc_x", "acc_y", "acc_z"],
            target="phi",
            train_runs=["train_a"],
            val_runs=["val_a"],
            eval_runs=list(eval_runs or ["eval_a"]),
            scaling=ScalingConfig(mode="prescaled", output_min=-1.0, output_max=1.0),
        ),
        model=ModelConfig(variant="slm_lstm", learning_rate=1e-3),
        runtime=RuntimeConfig(output_dir="outputs/experiments", run_name="eval_helper_cfg"),
    )


def test_load_bounds_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "scaler_bounds.json"
    path.write_text(json.dumps({"phi": {"lo": -30.0, "hi": 30.0}}), encoding="utf-8")

    bounds = wf.load_bounds_json(path)
    assert bounds["phi"].lo == -30.0
    assert bounds["phi"].hi == 30.0


def test_load_bounds_json_rejects_invalid_entry(tmp_path: Path) -> None:
    path = tmp_path / "bad_bounds.json"
    path.write_text(json.dumps({"phi": 123}), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid bounds entry"):
        wf.load_bounds_json(path)


def test_split_role_marks_overlap() -> None:
    cfg = _make_cfg(eval_runs=["shared_run"])
    cfg.data.train_runs = ["shared_run"]
    role, is_train, is_val, is_eval, is_unseen = wf._split_role("shared_run", cfg)
    assert role == "overlap"
    assert is_train is True
    assert is_eval is True
    assert is_unseen is False


def test_resolve_bounds_prefers_bounds_json(monkeypatch, tmp_path: Path) -> None:
    cfg = _make_cfg()
    output_dir = tmp_path / "run_out"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / cfg.runtime.bounds_path).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(wf, "load_bounds_json", lambda _p: {"phi": SimpleNamespace(lo=-1.0, hi=1.0)})
    called = {"h5": 0}

    def _fake_h5_bounds(_h5_path, columns):  # noqa: ANN001, ANN202
        _ = columns
        called["h5"] += 1
        return {}

    monkeypatch.setattr(wf, "load_hdf5_scaler_bounds", _fake_h5_bounds)
    bounds = wf._resolve_bounds(cfg, output_dir)
    assert "phi" in bounds
    assert called["h5"] == 0


def test_resolve_bounds_falls_back_to_hdf5(monkeypatch, tmp_path: Path) -> None:
    cfg = _make_cfg()
    output_dir = tmp_path / "run_out"
    output_dir.mkdir(parents=True, exist_ok=True)

    called = {"h5": 0}

    def _fake_h5_bounds(_h5_path, columns):  # noqa: ANN001, ANN202
        called["h5"] += 1
        assert set(columns) == {"pressure", "acc_x", "acc_y", "acc_z", "phi"}
        return {"phi": SimpleNamespace(lo=-2.0, hi=2.0)}

    monkeypatch.setattr(wf, "load_hdf5_scaler_bounds", _fake_h5_bounds)
    bounds = wf._resolve_bounds(cfg, output_dir)
    assert bounds["phi"].lo == -2.0
    assert called["h5"] == 1


def test_resolve_run_keys_rejects_invalid_scope() -> None:
    cfg = _make_cfg()
    with pytest.raises(ValueError, match="Unsupported evaluation scope"):
        wf._resolve_run_keys(cfg, "invalid_scope")  # type: ignore[arg-type]


def test_aggregate_handles_empty_and_zero_samples() -> None:
    empty = wf._aggregate([])
    assert empty["n_runs"] == 0
    assert empty["n_samples"] == 0

    zero_samples = wf._aggregate([{"n_samples": 0, "rmse": 1.0, "mae": 1.0}])
    assert zero_samples["n_runs"] == 1
    assert zero_samples["n_samples"] == 0

