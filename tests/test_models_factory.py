from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from spa_lstm.config import ModelConfig, TrainingConfig
from spa_lstm.models.factory import build_lstm_model


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
        self.lstm_defs: list[dict[str, object]] = []
        self.dense_defs: list[dict[str, object]] = []

        self.layers = SimpleNamespace(LSTM=self._lstm, Dense=self._dense)
        self.optimizers = SimpleNamespace(Adam=self._adam)
        self.metrics = SimpleNamespace(
            RootMeanSquaredError=self._rmse_metric,
            MeanAbsoluteError=self._mae_metric,
        )
        self.Model = _FakeModel

    def Input(  # noqa: N802
        self,
        *,
        batch_shape: tuple[int, None, int] | None = None,
        shape: tuple[None, int] | None = None,
        name: str | None = None,
    ) -> dict[str, object]:
        call = {"batch_shape": batch_shape, "shape": shape, "name": name}
        self.input_calls.append(call)
        return {"kind": "input", **call}

    def _lstm(  # noqa: ANN202
        self,
        units: int,
        *,
        activation: str,
        return_sequences: bool,
        stateful: bool,
        name: str,
    ):
        layer_def = {
            "units": units,
            "activation": activation,
            "return_sequences": return_sequences,
            "stateful": stateful,
            "name": name,
        }
        self.lstm_defs.append(layer_def)

        def _apply(x):  # noqa: ANN001, ANN202
            return {"kind": "lstm", "input": x, "def": layer_def}

        return _apply

    def _dense(self, units: int, *, name: str):  # noqa: ANN202
        layer_def = {"units": units, "name": name}
        self.dense_defs.append(layer_def)

        def _apply(x):  # noqa: ANN001, ANN202
            return {"kind": "dense", "input": x, "def": layer_def}

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


def test_build_lstm_model_single_layer_stateful_uses_batch_shape(monkeypatch) -> None:
    fake_keras = _install_fake_tensorflow(monkeypatch)

    model_cfg = ModelConfig(variant="slm_lstm", learning_rate=1e-3)
    train_cfg = TrainingConfig(stateful=True, batch_size=1)
    model = build_lstm_model(model_cfg, train_cfg, num_features=4)

    assert fake_keras.input_calls == [
        {"batch_shape": (1, None, 4), "shape": None, "name": "sensor_sequence"}
    ]
    assert len(fake_keras.lstm_defs) == 1
    assert fake_keras.lstm_defs[0]["units"] == 512
    assert fake_keras.lstm_defs[0]["return_sequences"] is True
    assert fake_keras.lstm_defs[0]["stateful"] is True
    assert fake_keras.dense_defs == [{"units": 1, "name": "phi_hat"}]
    assert model.name == "slm_lstm"
    assert model.compile_kwargs["loss"] == "mse"
    assert model.compile_kwargs["optimizer"] == {"learning_rate": 1e-3}


def test_build_lstm_model_two_layer_non_stateful_uses_shape(monkeypatch) -> None:
    fake_keras = _install_fake_tensorflow(monkeypatch)

    model_cfg = ModelConfig(variant="tlm_lstm", learning_rate=5e-4)
    train_cfg = TrainingConfig(stateful=False, batch_size=16)
    model = build_lstm_model(model_cfg, train_cfg, num_features=3)

    assert fake_keras.input_calls == [
        {"batch_shape": None, "shape": (None, 3), "name": "sensor_sequence"}
    ]
    assert [layer["units"] for layer in fake_keras.lstm_defs] == [256, 256]
    assert all(layer["return_sequences"] for layer in fake_keras.lstm_defs)
    assert all(layer["stateful"] is False for layer in fake_keras.lstm_defs)
    assert model.name == "tlm_lstm"


def test_build_lstm_model_rejects_unknown_variant(monkeypatch) -> None:
    _install_fake_tensorflow(monkeypatch)

    model_cfg = ModelConfig(variant="slm_lstm", learning_rate=1e-3)
    model_cfg.variant = "unknown_variant"
    train_cfg = TrainingConfig(stateful=True, batch_size=1)

    with pytest.raises(ValueError, match="Unknown model variant"):
        build_lstm_model(model_cfg, train_cfg, num_features=4)

