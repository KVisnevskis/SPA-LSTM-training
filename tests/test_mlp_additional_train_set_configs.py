from __future__ import annotations

from pathlib import Path

from mlp.config import load_experiment_config


def test_mlp_additional_train_set_configs_load_with_expected_architectures() -> None:
    config_dir = Path("configs/experiments/MLP_additional_train_sets")
    config_paths = sorted(config_dir.glob("*.yaml"))

    assert [path.name for path in config_paths] == [
        "accel_only_mlp_additional_train_set.yaml",
        "pressure_accel_mlp_additional_train_set.yaml",
        "pressure_only_mlp_additional_train_set.yaml",
    ]

    expected = {
        "accel_only_mlp_additional_train_set": {
            "features": ["acc_x", "acc_y", "acc_z"],
            "hidden_layers": [64],
            "learning_rate": 0.001,
        },
        "pressure_accel_mlp_additional_train_set": {
            "features": ["pressure", "acc_x", "acc_y", "acc_z"],
            "hidden_layers": [64, 32],
            "learning_rate": 0.0003,
        },
        "pressure_only_mlp_additional_train_set": {
            "features": ["pressure"],
            "hidden_layers": [128, 64],
            "learning_rate": 0.001,
        },
    }

    for config_path in config_paths:
        cfg = load_experiment_config(config_path)
        cfg_expected = expected[cfg.name]

        assert cfg.data.features == cfg_expected["features"]
        assert cfg.model.hidden_layers == cfg_expected["hidden_layers"]
        assert cfg.model.learning_rate == cfg_expected["learning_rate"]
        assert cfg.data.train_runs[-1] == "run_180roll_45pitch_tt_1"
        assert cfg.data.val_runs[-1] == "run_135roll_45pitch_tt_1"
        assert len(cfg.data.train_runs) == 6
        assert len(cfg.data.val_runs) == 6
        assert len(cfg.data.eval_runs) == 6
        assert cfg.runtime.output_dir == "outputs/experiments/mlp_additional_train_sets"
