"""Configuration models and YAML loading utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True)
class ColumnBounds:
    """Inclusive min/max bounds for min-max scaling."""

    lo: float
    hi: float

    def validate(self) -> None:
        if self.hi <= self.lo:
            raise ValueError(f"Invalid bounds: hi ({self.hi}) must be > lo ({self.lo}).")


@dataclass
class ScalingConfig:
    """Scaling policy for input features and target."""

    mode: Literal["prescaled"] = "prescaled"
    output_min: float = -1.0
    output_max: float = 1.0

    def validate(self) -> None:
        if self.mode != "prescaled":
            raise ValueError(f"Unsupported scaling mode '{self.mode}'. Expected 'prescaled'.")
        if self.output_max <= self.output_min:
            raise ValueError("output_max must be greater than output_min.")


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


@dataclass
class ModelConfig:
    """Model architecture selection and hyperparameters."""

    variant: str = "slm_lstm"
    learning_rate: float = 1e-3

    def validate(self) -> None:
        valid = {"slm_lstm", "tlm_lstm", "slu_lstm", "tlu_lstm"}
        if self.variant not in valid:
            raise ValueError(f"Unknown model variant '{self.variant}'. Expected one of {sorted(valid)}.")
        if self.learning_rate <= 0.0:
            raise ValueError("model.learning_rate must be > 0.")


@dataclass
class TrainingConfig:
    """Training process controls."""

    epochs: int = 300
    patience: int = 10
    batch_size: int = 1
    stateful: bool = True
    seed: int = 42

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("training.epochs must be > 0.")
        if self.patience < 0:
            raise ValueError("training.patience must be >= 0.")
        if self.batch_size != 1 and self.stateful:
            raise ValueError("Stateful thesis baseline requires batch_size = 1.")


@dataclass
class RuntimeConfig:
    """Output and runtime options."""

    output_dir: str = "outputs/experiments"
    run_name: str = "thesis_baseline"
    save_best_path: str = "best.keras"
    save_final_path: str = "final.keras"
    bounds_path: str = "scaler_bounds.json"


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

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
        self.data.scaling.validate()


def _parse_scaling(raw: dict[str, object] | None) -> ScalingConfig:
    raw = raw or {}
    return ScalingConfig(
        mode=str(raw.get("mode", "prescaled")),
        output_min=float(raw.get("output_min", -1.0)),
        output_max=float(raw.get("output_max", 1.0)),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config YAML file into typed dataclasses."""

    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping.")

    data_raw = raw.get("data", {})
    model_raw = raw.get("model", {})
    training_raw = raw.get("training", {})
    runtime_raw = raw.get("runtime", {})

    data_cfg = DataConfig(
        h5_path=str(data_raw["h5_path"]),
        features=list(data_raw["features"]),
        target=str(data_raw["target"]),
        train_runs=list(data_raw.get("train_runs", [])),
        val_runs=list(data_raw.get("val_runs", [])),
        eval_runs=list(data_raw.get("eval_runs", [])),
        scaling=_parse_scaling(data_raw.get("scaling", {})),
    )

    model_cfg = ModelConfig(
        variant=str(model_raw.get("variant", "slm_lstm")),
        learning_rate=float(model_raw.get("learning_rate", 1e-3)),
    )

    training_cfg = TrainingConfig(
        epochs=int(training_raw.get("epochs", 300)),
        patience=int(training_raw.get("patience", 10)),
        batch_size=int(training_raw.get("batch_size", 1)),
        stateful=bool(training_raw.get("stateful", True)),
        seed=int(training_raw.get("seed", 42)),
    )

    runtime_cfg = RuntimeConfig(
        output_dir=str(runtime_raw.get("output_dir", "outputs/experiments")),
        run_name=str(runtime_raw.get("run_name", raw.get("name", "thesis_baseline"))),
        save_best_path=str(runtime_raw.get("save_best_path", "best.keras")),
        save_final_path=str(runtime_raw.get("save_final_path", "final.keras")),
        bounds_path=str(runtime_raw.get("bounds_path", "scaler_bounds.json")),
    )

    cfg = ExperimentConfig(
        name=str(raw.get("name", "thesis_baseline")),
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        runtime=runtime_cfg,
    )
    cfg.validate()
    return cfg
