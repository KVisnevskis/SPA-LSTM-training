from __future__ import annotations

import numpy as np

from spa_lstm.evaluation.metrics import mae, rmse


def test_rmse_zero_for_identical_arrays() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert rmse(y_true, y_pred) == 0.0


def test_mae_expected_value() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert mae(y_true, y_pred) == 1.0

