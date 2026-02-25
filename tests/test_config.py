from __future__ import annotations

from spa_lstm.config import load_experiment_config


def test_load_experiment_config() -> None:
    cfg = load_experiment_config("configs/experiments/thesis_slm_lstm.yaml")
    assert cfg.name == "thesis_slm_lstm"
    assert cfg.model.variant == "slm_lstm"
    assert cfg.data.target == "phi"

