from __future__ import annotations

from pathlib import Path

import pytest

from mlp.config import load_experiment_config


def test_load_mlp_experiment_config_parses_valid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "baseline_slm_mlp.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: baseline_slm_mlp",
                "data:",
                "  h5_path: outputs/preprocessed_all_trials.h5",
                "  features: [pressure, acc_x, acc_y, acc_z]",
                "  target: phi",
                "  train_runs: [run_a, run_b]",
                "  val_runs: [run_c, run_d]",
                "  eval_runs: [run_e]",
                "  scaling:",
                "    mode: prescaled",
                "    output_min: -1.0",
                "    output_max: 1.0",
                "model:",
                "  hidden_layers: [64, 32]",
                "  activation: relu",
                "  dropout: 0.1",
                "  learning_rate: 0.001",
                "training:",
                "  epochs: 200",
                "  patience: 15",
                "  batch_size: 128",
                "  seed: 42",
                "  verbose: 1",
                "runtime:",
                "  output_dir: outputs/experiments/mlp",
                "  run_name: baseline_slm_mlp",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_experiment_config(config_path)

    assert cfg.name == "baseline_slm_mlp"
    assert cfg.data.features == ["pressure", "acc_x", "acc_y", "acc_z"]
    assert cfg.data.eval_runs == ["run_e"]
    assert cfg.model.hidden_layers == [64, 32]
    assert cfg.model.activation == "relu"
    assert cfg.model.dropout == 0.1
    assert cfg.training.batch_size == 128
    assert cfg.runtime.run_name == "baseline_slm_mlp"


def test_load_mlp_experiment_config_uses_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "minimal_mlp.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: minimal_mlp",
                "data:",
                "  h5_path: outputs/preprocessed_all_trials.h5",
                "  features: [pressure, acc_x, acc_y, acc_z]",
                "  target: phi",
                "  train_runs: [run_a]",
                "  val_runs: [run_b]",
                "  eval_runs: [run_c]",
                "model: {}",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_experiment_config(config_path)

    assert cfg.model.hidden_layers == [64]
    assert cfg.model.activation == "relu"
    assert cfg.model.dropout == 0.0
    assert cfg.model.learning_rate == 1e-3
    assert cfg.training.epochs == 200
    assert cfg.runtime.output_dir == "outputs/experiments/mlp"
    assert cfg.runtime.run_name == "minimal_mlp"


def test_load_mlp_experiment_config_rejects_empty_hidden_layers(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_hidden_layers.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: bad_hidden_layers",
                "data:",
                "  h5_path: outputs/preprocessed_all_trials.h5",
                "  features: [pressure, acc_x, acc_y, acc_z]",
                "  target: phi",
                "  train_runs: [run_a]",
                "  val_runs: [run_b]",
                "  eval_runs: [run_c]",
                "model:",
                "  hidden_layers: []",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model.hidden_layers must not be empty"):
        load_experiment_config(config_path)


def test_load_mlp_experiment_config_rejects_empty_eval_runs(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_eval_runs.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: bad_eval_runs",
                "data:",
                "  h5_path: outputs/preprocessed_all_trials.h5",
                "  features: [pressure, acc_x, acc_y, acc_z]",
                "  target: phi",
                "  train_runs: [run_a]",
                "  val_runs: [run_b]",
                "  eval_runs: []",
                "model:",
                "  hidden_layers: [64]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="data.eval_runs must not be empty"):
        load_experiment_config(config_path)


def test_load_mlp_experiment_config_rejects_non_prescaled_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_scaling.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: bad_scaling",
                "data:",
                "  h5_path: outputs/preprocessed_all_trials.h5",
                "  features: [pressure, acc_x, acc_y, acc_z]",
                "  target: phi",
                "  train_runs: [run_a]",
                "  val_runs: [run_b]",
                "  eval_runs: [run_c]",
                "  scaling:",
                "    mode: passthrough",
                "model:",
                "  hidden_layers: [64]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported scaling mode"):
        load_experiment_config(config_path)
