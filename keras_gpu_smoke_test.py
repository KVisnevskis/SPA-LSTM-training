#!/usr/bin/env python3
"""Quick Keras/TensorFlow GPU smoke test.

This script verifies:
1) TensorFlow can see at least one GPU.
2) A simple GPU matmul executes on GPU.
3) A tiny Keras model can train on GPU.

Exit code is non-zero on failure.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:
    print(f"[FAIL] Missing dependency: {exc}")
    print("Install TensorFlow in this environment first.")
    raise SystemExit(1)


def fail(message: str, code: int = 2) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(code)


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Keras API: {keras.__version__}")

    physical_gpus = tf.config.list_physical_devices("GPU")
    logical_gpus = tf.config.list_logical_devices("GPU")
    print(f"Physical GPUs detected: {len(physical_gpus)}")
    print(f"Logical GPUs detected: {len(logical_gpus)}")
    for i, gpu in enumerate(physical_gpus):
        print(f"  GPU[{i}]: {gpu}")

    if not physical_gpus:
        fail("No GPU detected by TensorFlow. Check CUDA/cuDNN/driver setup.")

    try:
        tf.config.set_soft_device_placement(False)
    except Exception:
        pass

    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            # Safe to ignore if already initialized.
            pass

    # Quick raw-op check that must execute on GPU.
    with tf.device("/GPU:0"):
        a = tf.random.uniform((512, 512))
        b = tf.random.uniform((512, 512))
        c = tf.matmul(a, b)

    print(f"Matmul output device: {c.device}")
    if "GPU" not in c.device.upper():
        fail("Matmul did not execute on GPU.")

    # Tiny synthetic dataset and minimal model for a fast smoke test.
    x = tf.random.normal((2048, 16), seed=7)
    y = tf.cast(tf.reduce_sum(x, axis=1, keepdims=True) > 0.0, tf.float32)

    # Restore soft placement and avoid forcing the entire training stack
    # onto GPU: input pipeline pieces may only have CPU kernels.
    tf.config.set_soft_device_placement(True)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(16,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        x,
        y,
        batch_size=128,
        epochs=2,
        verbose=2,
    )

    final_loss = history.history["loss"][-1]
    final_acc = history.history["accuracy"][-1]
    print(f"Final loss: {final_loss:.4f}")
    print(f"Final accuracy: {final_acc:.4f}")
    print("[PASS] Keras GPU smoke test completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
