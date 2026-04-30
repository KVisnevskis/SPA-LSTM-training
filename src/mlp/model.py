"""MLP model construction helpers."""

from __future__ import annotations

from mlp.config import ModelConfig

__all__ = ["build_mlp_model"]


def build_mlp_model(model_cfg: ModelConfig, num_features: int):
    """Build and compile a Keras MLP model for row-wise regression."""

    model_cfg.validate()
    if num_features <= 0:
        raise ValueError("num_features must be > 0.")

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("TensorFlow is required for model construction.") from exc

    inputs = tf.keras.Input(shape=(num_features,), name="sensor_features")
    x = inputs

    for index, units in enumerate(model_cfg.hidden_layers, start=1):
        x = tf.keras.layers.Dense(
            units,
            activation=model_cfg.activation,
            name=f"dense_{index}",
        )(x)
        if model_cfg.dropout > 0.0:
            x = tf.keras.layers.Dropout(model_cfg.dropout, name=f"dropout_{index}")(x)

    outputs = tf.keras.layers.Dense(1, name="phi_hat")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="slm_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=model_cfg.learning_rate),
        loss="mse",
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )
    return model
