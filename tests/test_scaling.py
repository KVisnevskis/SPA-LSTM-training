from __future__ import annotations

import numpy as np

from spa_lstm.config import ColumnBounds
from spa_lstm.data.scaling import minmax_inverse, minmax_scale


def test_scale_and_inverse_round_trip() -> None:
    bounds = ColumnBounds(lo=-2.0, hi=2.0)
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    scaled = minmax_scale(values, bounds, -1.0, 1.0)
    recovered = minmax_inverse(scaled, bounds, -1.0, 1.0)

    assert np.allclose(values, recovered)

