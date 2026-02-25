"""Common dataset constants for SPA-LSTM training."""

from __future__ import annotations

from spa_lstm.config import ColumnBounds

THESIS_FIXED_BOUNDS: dict[str, ColumnBounds] = {
    "acc_x": ColumnBounds(-12.0, 12.0),
    "acc_y": ColumnBounds(-12.0, 12.0),
    "acc_z": ColumnBounds(-12.0, 12.0),
    "pressure": ColumnBounds(400.0, 1800.0),
    "phi": ColumnBounds(-200.0, 50.0),
}

REQUIRED_BASE_COLUMNS: tuple[str, ...] = (
    "pressure",
    "acc_x",
    "acc_y",
    "acc_z",
    "phi",
)

