"""Common dataset constants for SPA-LSTM training."""

from __future__ import annotations

REQUIRED_BASE_COLUMNS: tuple[str, ...] = (
    "pressure",
    "acc_x",
    "acc_y",
    "acc_z",
    "phi",
)
