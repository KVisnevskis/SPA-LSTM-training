"""MLP experiment configuration models and YAML loading utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "RuntimeConfig",
    "ScalingConfig",
    "TrainingConfig",
    "load_experiment_config",
]


def _as_mapping(raw: object, field_name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    return raw


def _as_string_list(raw: object, field_name: str) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a list.")

    values: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{index}] must be a string.")
        value = item.strip()
        if not value:
            raise ValueError(f"{field_name}[{index}] must not be empty.")
        values.append(value)
    return values


def _as_int_list(raw: object, field_name: str) -> list[int]:
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a list.")

    values: list[int] = []
    for index, item in enumerate(raw):
        try:
            value = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}[{index}] must be an integer.") from exc
        values.append(value)
    return values


@dataclass
class ScalingConfig:
    """Scaling policy for MLP inputs and target."""

    mode: Literal["prescaled"] = "prescaled"
    output_min: float = -1.0
    output_max: float = 1.0

    def validate(self) -> None:
        if self.mode != "prescaled":
            raise ValueError(f"Unsupported scaling mode '{self.mode}'. Expected 'prescaled'.")
        if self.output_max <= self.output_min:
            raise ValueError("data.scaling.output_max must be greater than data.scaling.output_min.")


@dataclass
class DataConfig:
    """Data source and split configuration."""

    h5_path: str
    features: list[str]
    target: str
    train_runs: list[str]
    val_runs: list[str]
    eval_runs: list[str]
    scaling: ScalingConfig = field(default_factory=ScalingConfig)

    def validate(self) -> None:
        if not self.h5_path:
            raise ValueError("data.h5_path must not be empty.")
        if not self.features:
            raise ValueError("data.features must not be empty.")
        if not self.target:
            raise ValueError("data.target must not be empty.")
        if not self.train_runs:
            raise ValueError("data.train_runs must not be empty.")
        if not self.val_runs:
            raise ValueError("data.val_runs must not be empty.")
        if not self.eval_runs:
            raise ValueError("data.eval_runs must not be empty.")
        self.scaling.validate()


@dataclass
class ModelConfig:
    """MLP architecture selection and hyperparameters."""

    hidden_layers: list[int] = field(default_factory=lambda: [64])
    activation: str = "relu"
    dropout: float = 0.0
    learning_rate: float = 1e-3

    def validate(self) -> None:
        if not self.hidden_layers:
            raise ValueError("model.hidden_layers must not be empty.")
        if any(units <= 0 for units in self.hidden_layers):
            raise ValueError("model.hidden_layers values must be > 0.")
        if not self.activation:
            raise ValueError("model.activation must not be empty.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in the range [0.0, 1.0).")
        if self.learning_rate <= 0.0:
            raise ValueError("model.learning_rate must be > 0.")


@dataclass
class TrainingConfig:
    """Training process controls."""

    epochs: int = 200
    patience: int = 15
    batch_size: int = 128
    seed: int = 42
    verbose: int = 1

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("training.epochs must be > 0.")
        if self.patience < 0:
            raise ValueError("training.patience must be >= 0.")
        if self.batch_size <= 0:
            raise ValueError("training.batch_size must be > 0.")
        if self.verbose < 0:
            raise ValueError("training.verbose must be >= 0.")


@dataclass
class RuntimeConfig:
    """Output and runtime options."""

    output_dir: str = "outputs/experiments/mlp"
    run_name: str = "mlp_baseline"
    save_best_path: str = "best.keras"
    save_final_path: str = "final.keras"
    bounds_path: str = "scaler_bounds.json"

    def validate(self) -> None:
        if not self.output_dir:
            raise ValueError("runtime.output_dir must not be empty.")
        if not self.run_name:
            raise ValueError("runtime.run_name must not be empty.")
        if not self.save_best_path:
            raise ValueError("runtime.save_best_path must not be empty.")
        if not self.save_final_path:
            raise ValueError("runtime.save_final_path must not be empty.")
        if not self.bounds_path:
            raise ValueError("runtime.bounds_path must not be empty.")


@dataclass
class ExperimentConfig:
    """Top-level MLP experiment configuration."""

    name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("experiment name must not be empty.")
        self.data.validate()
        self.model.validate()
        self.training.validate()
        self.runtime.validate()


def _parse_scaling(raw: object) -> ScalingConfig:
    raw_map = _as_mapping(raw or {}, "data.scaling")
    return ScalingConfig(
        mode=str(raw_map.get("mode", "prescaled")),
        output_min=float(raw_map.get("output_min", -1.0)),
        output_max=float(raw_map.get("output_max", 1.0)),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an MLP experiment config YAML file into typed dataclasses."""

    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw_map = _as_mapping(raw, "Config root")
    data_raw = _as_mapping(raw_map.get("data"), "data")
    model_raw = _as_mapping(raw_map.get("model"), "model")
    training_raw = _as_mapping(raw_map.get("training", {}), "training")
    runtime_raw = _as_mapping(raw_map.get("runtime", {}), "runtime")

    try:
        data_cfg = DataConfig(
            h5_path=str(data_raw["h5_path"]).strip(),
            features=_as_string_list(data_raw["features"], "data.features"),
            target=str(data_raw["target"]).strip(),
            train_runs=_as_string_list(data_raw["train_runs"], "data.train_runs"),
            val_runs=_as_string_list(data_raw["val_runs"], "data.val_runs"),
            eval_runs=_as_string_list(data_raw["eval_runs"], "data.eval_runs"),
            scaling=_parse_scaling(data_raw.get("scaling", {})),
        )
    except KeyError as exc:
        raise ValueError(f"Missing required data field: {exc.args[0]}") from exc

    try:
        model_cfg = ModelConfig(
            hidden_layers=_as_int_list(model_raw.get("hidden_layers", [64]), "model.hidden_layers"),
            activation=str(model_raw.get("activation", "relu")).strip(),
            dropout=float(model_raw.get("dropout", 0.0)),
            learning_rate=float(model_raw.get("learning_rate", 1e-3)),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Invalid model configuration: {exc}") from exc

    training_cfg = TrainingConfig(
        epochs=int(training_raw.get("epochs", 200)),
        patience=int(training_raw.get("patience", 15)),
        batch_size=int(training_raw.get("batch_size", 128)),
        seed=int(training_raw.get("seed", 42)),
        verbose=int(training_raw.get("verbose", 1)),
    )

    runtime_cfg = RuntimeConfig(
        output_dir=str(runtime_raw.get("output_dir", "outputs/experiments/mlp")).strip(),
        run_name=str(runtime_raw.get("run_name", raw_map.get("name", "mlp_baseline"))).strip(),
        save_best_path=str(runtime_raw.get("save_best_path", "best.keras")).strip(),
        save_final_path=str(runtime_raw.get("save_final_path", "final.keras")).strip(),
        bounds_path=str(runtime_raw.get("bounds_path", "scaler_bounds.json")).strip(),
    )

    cfg = ExperimentConfig(
        name=str(raw_map.get("name", "mlp_baseline")).strip(),
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        runtime=runtime_cfg,
    )
    cfg.validate()
    return cfg
