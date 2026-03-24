from __future__ import annotations

from pathlib import Path

from mlp.config import load_experiment_config


def test_all_mlp_hpo_configs_load_and_have_unique_run_names() -> None:
    config_paths = sorted(Path("configs/experiments/mlp/hpo").glob("*.yaml"))

    assert len(config_paths) == 8

    run_names: set[str] = set()
    model_shapes: set[tuple[int, ...]] = set()
    learning_rates: set[float] = set()

    for config_path in config_paths:
        cfg = load_experiment_config(config_path)
        assert cfg.data.features == ["pressure", "acc_x", "acc_y", "acc_z"]
        assert cfg.data.target == "phi"
        assert len(cfg.data.train_runs) == 5
        assert len(cfg.data.val_runs) == 5
        assert len(cfg.data.eval_runs) == 6
        assert cfg.runtime.output_dir == "outputs/experiments/mlp/hpo"
        run_names.add(cfg.runtime.run_name)
        model_shapes.add(tuple(cfg.model.hidden_layers))
        learning_rates.add(cfg.model.learning_rate)

    assert len(run_names) == len(config_paths)
    assert model_shapes == {
        (32,),
        (64,),
        (128,),
        (64, 32),
        (64, 64),
        (128, 64),
    }
    assert learning_rates == {0.001, 0.0003}
