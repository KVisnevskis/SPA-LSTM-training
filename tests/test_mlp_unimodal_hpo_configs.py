from __future__ import annotations

from pathlib import Path

from mlp.config import load_experiment_config


def test_all_mlp_unimodal_hpo_configs_load_and_have_unique_run_names() -> None:
    config_paths = sorted(Path("configs/experiments/mlp/hpo_unimodal").glob("*.yaml"))

    assert len(config_paths) == 16

    pressure_run_names: set[str] = set()
    accel_run_names: set[str] = set()
    pressure_shapes: set[tuple[int, ...]] = set()
    accel_shapes: set[tuple[int, ...]] = set()
    pressure_learning_rates: set[float] = set()
    accel_learning_rates: set[float] = set()

    for config_path in config_paths:
        cfg = load_experiment_config(config_path)
        assert cfg.data.target == "phi"
        assert len(cfg.data.train_runs) == 5
        assert len(cfg.data.val_runs) == 5
        assert len(cfg.data.eval_runs) == 6
        assert cfg.runtime.output_dir == "outputs/experiments/mlp/hpo_unimodal"

        if cfg.runtime.run_name.startswith("pressure_mlp__"):
            assert cfg.data.features == ["pressure"]
            pressure_run_names.add(cfg.runtime.run_name)
            pressure_shapes.add(tuple(cfg.model.hidden_layers))
            pressure_learning_rates.add(cfg.model.learning_rate)
        elif cfg.runtime.run_name.startswith("accel_mlp__"):
            assert cfg.data.features == ["acc_x", "acc_y", "acc_z"]
            accel_run_names.add(cfg.runtime.run_name)
            accel_shapes.add(tuple(cfg.model.hidden_layers))
            accel_learning_rates.add(cfg.model.learning_rate)
        else:
            raise AssertionError(f"Unexpected run name: {cfg.runtime.run_name}")

    assert len(pressure_run_names) == 8
    assert len(accel_run_names) == 8
    expected_shapes = {
        (32,),
        (64,),
        (128,),
        (64, 32),
        (64, 64),
        (128, 64),
    }
    assert pressure_shapes == expected_shapes
    assert accel_shapes == expected_shapes
    assert pressure_learning_rates == {0.001, 0.0003}
    assert accel_learning_rates == {0.001, 0.0003}
