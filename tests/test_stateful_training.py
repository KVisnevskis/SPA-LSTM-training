from __future__ import annotations

import numpy as np

from spa_lstm.training.stateful import train_stateful


class _FakeHistory:
    def __init__(self, loss: float) -> None:
        self.history = {"loss": [loss], "rmse": [loss + 0.1], "mae": [loss + 0.2]}


class _FakeLayer:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset_states(self) -> None:
        self.reset_count += 1


class _FakeModel:
    def __init__(self, train_losses: list[float], val_losses: list[float]) -> None:
        self.layers = [_FakeLayer()]
        self._train_losses = iter(train_losses)
        self._val_losses = iter(val_losses)
        self._weight = 0.0
        self.optimizer = type("Optimizer", (), {"learning_rate": 1e-3})()

    def fit(self, x, y, epochs, batch_size, shuffle, verbose):  # noqa: ANN001
        _ = (x, y, epochs, batch_size, shuffle, verbose)
        self._weight += 1.0
        return _FakeHistory(next(self._train_losses))

    def evaluate(self, x, y, batch_size, verbose, return_dict=True):  # noqa: ANN001, ARG002
        _ = (x, y, batch_size, verbose)
        loss = next(self._val_losses)
        return {"loss": loss, "rmse": loss + 0.1, "mae": loss + 0.2} if return_dict else [loss]

    def get_weights(self):  # noqa: ANN201
        return [np.array([self._weight], dtype=np.float32)]

    def set_weights(self, weights):  # noqa: ANN001
        self._weight = float(weights[0][0])


def test_stateful_training_resets_on_train_and_validation_boundaries() -> None:
    model = _FakeModel(train_losses=[1.0, 0.9], val_losses=[0.5, 0.6])
    x = np.ones((3, 2), dtype=np.float32)
    y = np.ones((3, 1), dtype=np.float32)
    train_pairs = [(x, y, x, y)]

    result = train_stateful(model, train_pairs=train_pairs, epochs=5, patience=1, verbose=0)

    assert len(result.history) == 2
    assert result.history[0].train_rmse_mean == 1.1
    assert result.history[0].train_mae_mean == 1.2
    assert result.history[0].val_rmse_mean == 0.6
    assert result.history[0].val_mae_mean == 0.7
    assert result.best_epoch == 1
    assert result.stopped_early is True
    assert result.best_weights[0][0] == 1.0
    assert model.layers[0].reset_count == 4
