"""Stateful sequence training loop for thesis-aligned baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class EpochSummary:
    """Per-epoch aggregate metrics captured by trainer."""

    epoch: int
    train_loss_mean: float
    train_rmse_mean: float
    train_mae_mean: float
    val_loss_mean: float
    val_rmse_mean: float
    val_mae_mean: float
    learning_rate: float


@dataclass
class TrainingResult:
    """Training loop output with best-model state metadata."""

    history: list[EpochSummary]
    best_epoch: int
    best_val_loss: float
    stopped_early: bool
    best_weights: list[Any]


EpochEndCallback = Callable[[EpochSummary, list[EpochSummary], int, float, int], None]


def reset_recurrent_states(model) -> None:
    """Reset states for all recurrent layers that expose reset_states()."""

    for layer in getattr(model, "layers", []):
        if hasattr(layer, "reset_states"):
            layer.reset_states()


def as_sequence_batch(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert [T,F], [T,1] to sample-stream batches [T,1,F], [T,1,1]."""

    if x.ndim != 2:
        raise ValueError(f"Expected x with shape [T,F], got {x.shape}")
    if y.ndim != 2:
        raise ValueError(f"Expected y with shape [T,1], got {y.shape}")
    return x[:, None, :], y[:, None, :]


def _current_learning_rate(model) -> float:
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        return float("nan")

    lr = getattr(optimizer, "learning_rate", None)
    if lr is None:
        return float("nan")

    try:
        import tensorflow as tf

        return float(tf.keras.backend.get_value(lr))
    except Exception:
        try:
            return float(lr)
        except Exception:
            return float("nan")


def train_stateful(
    model,
    train_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    epochs: int,
    patience: int,
    verbose: int = 1,
    fit_verbose: int = 0,
    eval_verbose: int = 0,
    pair_labels: list[str] | None = None,
    log_each_fit: bool = False,
    start_epoch: int = 1,
    initial_history: list[EpochSummary] | None = None,
    best_epoch: int = 0,
    best_val_loss: float = float("inf"),
    best_weights: list[Any] | None = None,
    epochs_without_improve: int = 0,
    on_epoch_end: EpochEndCallback | None = None,
) -> TrainingResult:
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
    if pair_labels is not None and len(pair_labels) != len(train_pairs):
        raise ValueError(
            f"pair_labels length ({len(pair_labels)}) must match train_pairs length ({len(train_pairs)})."
        )

    history: list[EpochSummary] = list(initial_history or [])
    best_val = float(best_val_loss)
    best_weights = model.get_weights() if best_weights is None else best_weights
    stopped_early = False

    for epoch in range(start_epoch, epochs + 1):
        train_losses: list[float] = []
        train_rmses: list[float] = []
        train_maes: list[float] = []
        val_losses: list[float] = []
        val_rmses: list[float] = []
        val_maes: list[float] = []

        for pair_index, (x_train, y_train, x_val, y_val) in enumerate(train_pairs):
            pair_label = (
                pair_labels[pair_index]
                if pair_labels is not None
                else f"pair_{pair_index + 1}"
            )
            x_train_b, y_train_b = as_sequence_batch(x_train, y_train)
            x_val_b, y_val_b = as_sequence_batch(x_val, y_val)

            if log_each_fit:
                print(
                    f"Epoch {epoch:04d} | fit-start | {pair_label} | "
                    f"train_shape={x_train_b.shape}->{y_train_b.shape}"
                )
            reset_recurrent_states(model)
            fit_result = model.fit(
                x_train_b,
                y_train_b,
                epochs=1,
                batch_size=1,
                shuffle=False,
                verbose=fit_verbose,
            )
            train_loss = float(fit_result.history["loss"][-1])
            train_rmse = float(fit_result.history.get("rmse", [float("nan")])[-1])
            train_mae = float(fit_result.history.get("mae", [float("nan")])[-1])
            train_losses.append(train_loss)
            train_rmses.append(train_rmse)
            train_maes.append(train_mae)
            if log_each_fit:
                print(
                    f"Epoch {epoch:04d} | fit-end | {pair_label} | "
                    f"loss={train_loss:.6f} | rmse={train_rmse:.6f} | mae={train_mae:.6f}"
                )

            if log_each_fit:
                print(
                    f"Epoch {epoch:04d} | eval-start | {pair_label} | "
                    f"val_shape={x_val_b.shape}->{y_val_b.shape}"
                )
            reset_recurrent_states(model)
            eval_result = model.evaluate(
                x_val_b,
                y_val_b,
                batch_size=1,
                verbose=eval_verbose,
                return_dict=True,
            )
            val_loss = float(eval_result["loss"])
            val_rmse = float(eval_result.get("rmse", float("nan")))
            val_mae = float(eval_result.get("mae", float("nan")))
            val_losses.append(val_loss)
            val_rmses.append(val_rmse)
            val_maes.append(val_mae)
            if log_each_fit:
                print(
                    f"Epoch {epoch:04d} | eval-end | {pair_label} | "
                    f"loss={val_loss:.6f} | rmse={val_rmse:.6f} | mae={val_mae:.6f}"
                )

        summary = EpochSummary(
            epoch=epoch,
            train_loss_mean=float(np.mean(train_losses)),
            train_rmse_mean=float(np.mean(train_rmses)),
            train_mae_mean=float(np.mean(train_maes)),
            val_loss_mean=float(np.mean(val_losses)),
            val_rmse_mean=float(np.mean(val_rmses)),
            val_mae_mean=float(np.mean(val_maes)),
            learning_rate=_current_learning_rate(model),
        )
        history.append(summary)

        if verbose:
            print(
                f"Epoch {epoch:04d} | "
                f"train_loss={summary.train_loss_mean:.6f} | "
                f"train_rmse={summary.train_rmse_mean:.6f} | "
                f"train_mae={summary.train_mae_mean:.6f} | "
                f"val_loss={summary.val_loss_mean:.6f} | "
                f"val_rmse={summary.val_rmse_mean:.6f} | "
                f"val_mae={summary.val_mae_mean:.6f}"
            )

        if summary.val_loss_mean < best_val:
            best_val = summary.val_loss_mean
            best_epoch = epoch
            epochs_without_improve = 0
            best_weights = model.get_weights()
        else:
            epochs_without_improve += 1

        if on_epoch_end is not None:
            on_epoch_end(summary, history, best_epoch, float(best_val), epochs_without_improve)

        if epochs_without_improve >= patience:
            if verbose:
                print(f"Early stopping: no improvement for {patience} epoch(s).")
            stopped_early = True
            break

    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_val_loss=float(best_val),
        stopped_early=stopped_early,
        best_weights=best_weights,
    )
