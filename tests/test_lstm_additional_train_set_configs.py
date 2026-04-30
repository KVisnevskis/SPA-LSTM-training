from __future__ import annotations

from pathlib import Path

from spa_lstm.config import load_experiment_config


def test_lstm_additional_train_set_configs_load_with_expected_split_and_runtime() -> None:
    config_dir = Path("configs/experiments/lstm_additional_train_sets")
    config_paths = sorted(config_dir.glob("slm_lstm_additional_train_set__seed*.yaml"))

    assert [path.name for path in config_paths] == [
        "slm_lstm_additional_train_set__seed007.yaml",
        "slm_lstm_additional_train_set__seed017.yaml",
        "slm_lstm_additional_train_set__seed027.yaml",
    ]

    expected_seeds = {
        "slm_lstm_additional_train_set__seed007": 7,
        "slm_lstm_additional_train_set__seed017": 17,
        "slm_lstm_additional_train_set__seed027": 27,
    }

    for config_path in config_paths:
        cfg = load_experiment_config(config_path)

        assert cfg.model.variant == "slm_lstm"
        assert cfg.model.learning_rate == 0.001
        assert cfg.data.features == ["pressure", "acc_x", "acc_y", "acc_z"]
        assert cfg.data.train_runs[-1] == "run_180roll_45pitch_tt_1"
        assert cfg.data.val_runs[-1] == "run_135roll_45pitch_tt_1"
        assert len(cfg.data.train_runs) == 6
        assert len(cfg.data.val_runs) == 6
        assert len(cfg.data.eval_runs) == 6
        assert cfg.training.seed == expected_seeds[cfg.name]
        assert cfg.runtime.output_dir == "outputs/experiments/lstm_additional_train_sets"
