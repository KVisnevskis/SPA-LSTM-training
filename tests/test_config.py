from __future__ import annotations

from pathlib import Path

import pytest

from spa_lstm.config import load_experiment_config


def test_load_experiment_config() -> None:
    cfg = load_experiment_config("configs/experiments/thesis_slm_lstm.yaml")
    assert cfg.name == "thesis_slm_lstm"
    assert cfg.model.variant == "slm_lstm"
    assert cfg.data.target == "phi"
    assert cfg.data.scaling.mode == "prescaled"


def test_load_experiment_config_rejects_non_prescaled_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_scaling.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: bad_mode",
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
                "  variant: slm_lstm",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported scaling mode"):
        load_experiment_config(config_path)
