"""Stateful sequence training loop for thesis-aligned baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EpochSummary:
    """Per-epoch aggregate metrics captured by trainer."""

    epoch: int
    train_loss_mean: float
    val_loss_mean: float


def reset_recurrent_states(model) -> None:
    """Reset states for all recurrent layers that expose reset_states()."""

    for layer in getattr(model, "layers", []):
        if hasattr(layer, "reset_states"):
            layer.reset_states()


def as_sequence_batch(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert [T,F], [T,1] to batch-first [1,T,F], [1,T,1]."""

    if x.ndim != 2:
        raise ValueError(f"Expected x with shape [T,F], got {x.shape}")
    if y.ndim != 2:
        raise ValueError(f"Expected y with shape [T,1], got {y.shape}")
    return x[None, ...], y[None, ...]


def train_stateful(
    model,
    train_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    epochs: int,
    patience: int,
    verbose: int = 1,
) -> list[EpochSummary]:
    """Train with run-wise state resets and early stopping.

    Parameters
    ----------
    model:
        Compiled Keras model.
    train_pairs:
        List of `(x_train, y_train, x_val, y_val)` arrays, each with shapes
        `[T, F]` and `[T, 1]`.
    """

    if not train_pairs:
        raise ValueError("train_pairs is empty.")

    history: list[EpochSummary] = []
    best_val = float("inf")
    epochs_without_improve = 0
    best_weights = model.get_weights()

    for epoch in range(1, epochs + 1):
        train_losses: list[float] = []
        val_losses: list[float] = []

        for x_train, y_train, x_val, y_val in train_pairs:
            reset_recurrent_states(model)

            x_train_b, y_train_b = as_sequence_batch(x_train, y_train)
            x_val_b, y_val_b = as_sequence_batch(x_val, y_val)

            result = model.fit(
                x_train_b,
                y_train_b,
                validation_data=(x_val_b, y_val_b),
                epochs=1,
                batch_size=1,
                shuffle=False,
                verbose=0,
            )
            train_losses.append(float(result.history["loss"][-1]))
            val_losses.append(float(result.history["val_loss"][-1]))

        summary = EpochSummary(
            epoch=epoch,
            train_loss_mean=float(np.mean(train_losses)),
            val_loss_mean=float(np.mean(val_losses)),
        )
        history.append(summary)

        if verbose:
            print(
                f"Epoch {epoch:04d} | "
                f"train_loss={summary.train_loss_mean:.6f} | "
                f"val_loss={summary.val_loss_mean:.6f}"
            )

        if summary.val_loss_mean < best_val:
            best_val = summary.val_loss_mean
            epochs_without_improve = 0
            best_weights = model.get_weights()
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= patience:
            if verbose:
                print(f"Early stopping: no improvement for {patience} epoch(s).")
            break

    model.set_weights(best_weights)
    return history

