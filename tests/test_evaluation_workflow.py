from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from spa_lstm.config import DataConfig, ExperimentConfig, ModelConfig, RuntimeConfig, ScalingConfig
from spa_lstm.evaluation.workflow import evaluate_model


class _FakeModel:
    def __init__(self) -> None:
        self.layers: list[object] = []
        self.predict_calls: list[tuple[tuple[int, ...], int, int]] = []

    def predict(self, x, batch_size, verbose):  # noqa: ANN001
        self.predict_calls.append((tuple(x.shape), int(batch_size), int(verbose)))
        return np.zeros((x.shape[0], x.shape[1], 1), dtype=np.float32)


def test_evaluate_model_uses_prescaled_bounds_and_stream_shape(tmp_path, monkeypatch) -> None:
    h5_path = tmp_path / "dataset.h5"
    h5_path.write_text("", encoding="utf-8")

    cfg = ExperimentConfig(
        name="eval_smoke",
        data=DataConfig(
            h5_path=str(h5_path),
            features=["pressure", "acc_x", "acc_y", "acc_z"],
            target="phi",
            train_runs=[],
            val_runs=[],
            eval_runs=["eval_a"],
            scaling=ScalingConfig(mode="prescaled", output_min=-1.0, output_max=1.0),
        ),
        model=ModelConfig(variant="slm_lstm", learning_rate=1e-3),
        runtime=RuntimeConfig(output_dir=str(tmp_path / "outputs"), run_name="eval_smoke"),
    )

    eval_df = pd.DataFrame(
        {
            "pressure": [0.0, 0.2, 0.3],
            "acc_x": [0.1, 0.2, 0.3],
            "acc_y": [0.2, 0.1, 0.0],
            "acc_z": [1.0, 1.1, 1.2],
            "phi": [-1.0, 0.0, 1.0],
            "Time": [0.0, 0.02, 0.04],
        }
    )
    monkeypatch.setattr(
        "spa_lstm.evaluation.workflow.load_runs_as_dataframes",
        lambda _h5, run_keys, _required: {run_key: eval_df.copy() for run_key in run_keys},
    )
    monkeypatch.setattr(
        "spa_lstm.evaluation.workflow.load_hdf5_scaler_bounds",
        lambda _h5, columns: {
            "pressure": SimpleNamespace(lo=-1.0, hi=1.0),
            "acc_x": SimpleNamespace(lo=-1.0, hi=1.0),
            "acc_y": SimpleNamespace(lo=-1.0, hi=1.0),
            "acc_z": SimpleNamespace(lo=-1.0, hi=1.0),
            "phi": SimpleNamespace(lo=-30.0, hi=30.0),
        },
    )

    fake_model = _FakeModel()
    fake_tf = SimpleNamespace(keras=SimpleNamespace(models=SimpleNamespace(load_model=lambda _path: fake_model)))
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    out_dir = tmp_path / "outputs" / "eval_smoke"
    metrics_path = evaluate_model(cfg, model_path="dummy.keras", run_dir=str(out_dir))

    assert metrics_path.exists()
    assert (out_dir / "predictions" / "eval_a.csv").exists()
    assert fake_model.predict_calls == [((3, 1, 4), 1, 0)]

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert len(metrics) == 1
    assert metrics[0]["run_key"] == "eval_a"
    assert metrics[0]["n_samples"] == 3
