"""Keras model factory for thesis LSTM variants."""

from __future__ import annotations

from spa_lstm.config import ModelConfig, TrainingConfig


def build_lstm_model(model_cfg: ModelConfig, train_cfg: TrainingConfig, num_features: int):
    """Build and compile a Keras LSTM model for the selected variant.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model.
    """

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("TensorFlow is required for model construction.") from exc

    variant = model_cfg.variant

    if variant in {"slm_lstm", "slu_lstm"}:
        units = [512]
    elif variant in {"tlm_lstm", "tlu_lstm"}:
        units = [256, 256]
    else:
        raise ValueError(f"Unknown model variant: {variant}")

    if train_cfg.stateful:
        inputs = tf.keras.Input(batch_shape=(train_cfg.batch_size, None, num_features), name="sensor_sequence")
    else:
        inputs = tf.keras.Input(shape=(None, num_features), name="sensor_sequence")

    x = inputs
    for i, unit in enumerate(units):
        x = tf.keras.layers.LSTM(
            unit,
            activation="tanh",
            return_sequences=True,
            stateful=train_cfg.stateful,
            name=f"lstm_{i + 1}",
        )(x)

    outputs = tf.keras.layers.Dense(1, name="phi_hat")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=variant)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=model_cfg.learning_rate),
        loss="mse",
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )
    return model

