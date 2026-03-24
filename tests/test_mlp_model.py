from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from mlp.config import ModelConfig
from mlp.model import build_mlp_model


class _FakeModel:
    def __init__(self, inputs, outputs, name: str) -> None:  # noqa: ANN001
        self.inputs = inputs
        self.outputs = outputs
        self.name = name
        self.compile_kwargs: dict[str, object] = {}

    def compile(self, **kwargs) -> None:  # noqa: ANN003
        self.compile_kwargs = kwargs


class _FakeKeras:
    def __init__(self) -> None:
        self.input_calls: list[dict[str, object]] = []
        self.dense_defs: list[dict[str, object]] = []
        self.dropout_defs: list[dict[str, object]] = []

        self.layers = SimpleNamespace(Dense=self._dense, Dropout=self._dropout)
        self.optimizers = SimpleNamespace(Adam=self._adam)
        self.metrics = SimpleNamespace(
            RootMeanSquaredError=self._rmse_metric,
            MeanAbsoluteError=self._mae_metric,
        )
        self.Model = _FakeModel

    def Input(self, *, shape: tuple[int, ...] | None = None, name: str | None = None):  # noqa: N802, ANN201
        call = {"shape": shape, "name": name}
        self.input_calls.append(call)
        return {"kind": "input", **call}

    def _dense(self, units: int, *, activation: str | None = None, name: str):  # noqa: ANN202
        layer_def = {"units": units, "activation": activation, "name": name}
        self.dense_defs.append(layer_def)

        def _apply(x):  # noqa: ANN001, ANN202
            return {"kind": "dense", "input": x, "def": layer_def}

        return _apply

    def _dropout(self, rate: float, *, name: str):  # noqa: ANN202
        layer_def = {"rate": rate, "name": name}
        self.dropout_defs.append(layer_def)

        def _apply(x):  # noqa: ANN001, ANN202
            return {"kind": "dropout", "input": x, "def": layer_def}

        return _apply

    @staticmethod
    def _adam(*, learning_rate: float) -> dict[str, float]:
        return {"learning_rate": float(learning_rate)}

    @staticmethod
    def _rmse_metric(*, name: str) -> dict[str, str]:
        return {"metric": "rmse", "name": name}

    @staticmethod
    def _mae_metric(*, name: str) -> dict[str, str]:
        return {"metric": "mae", "name": name}


def _install_fake_tensorflow(monkeypatch: pytest.MonkeyPatch) -> _FakeKeras:
    fake_keras = _FakeKeras()
    fake_tf = SimpleNamespace(keras=fake_keras)
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    return fake_keras


def test_build_mlp_model_single_hidden_layer(monkeypatch) -> None:
    fake_keras = _install_fake_tensorflow(monkeypatch)

    model_cfg = ModelConfig(hidden_layers=[64], activation="relu", dropout=0.0, learning_rate=1e-3)
    model = build_mlp_model(model_cfg, num_features=4)

    assert fake_keras.input_calls == [{"shape": (4,), "name": "sensor_features"}]
    assert fake_keras.dense_defs == [
        {"units": 64, "activation": "relu", "name": "dense_1"},
        {"units": 1, "activation": None, "name": "phi_hat"},
    ]
    assert fake_keras.dropout_defs == []
    assert model.name == "slm_mlp"
    assert model.compile_kwargs["loss"] == "mse"
    assert model.compile_kwargs["optimizer"] == {"learning_rate": 1e-3}


def test_build_mlp_model_applies_multiple_hidden_layers_and_dropout(monkeypatch) -> None:
    fake_keras = _install_fake_tensorflow(monkeypatch)

    model_cfg = ModelConfig(hidden_layers=[128, 64], activation="tanh", dropout=0.2, learning_rate=5e-4)
    model = build_mlp_model(model_cfg, num_features=3)

    assert fake_keras.input_calls == [{"shape": (3,), "name": "sensor_features"}]
    assert fake_keras.dense_defs == [
        {"units": 128, "activation": "tanh", "name": "dense_1"},
        {"units": 64, "activation": "tanh", "name": "dense_2"},
        {"units": 1, "activation": None, "name": "phi_hat"},
    ]
    assert fake_keras.dropout_defs == [
        {"rate": 0.2, "name": "dropout_1"},
        {"rate": 0.2, "name": "dropout_2"},
    ]
    assert model.name == "slm_mlp"
    assert model.compile_kwargs["optimizer"] == {"learning_rate": 5e-4}


def test_build_mlp_model_rejects_non_positive_num_features(monkeypatch) -> None:
    _install_fake_tensorflow(monkeypatch)

    model_cfg = ModelConfig(hidden_layers=[64], activation="relu", dropout=0.0, learning_rate=1e-3)

    with pytest.raises(ValueError, match="num_features must be > 0"):
        build_mlp_model(model_cfg, num_features=0)
