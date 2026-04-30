from __future__ import annotations

import importlib


def test_mlp_package_modules_import_cleanly() -> None:
    import mlp

    assert mlp.__version__ == "0.1.0"

    for module_name in ("config", "data", "evaluation", "model", "training"):
        module = importlib.import_module(f"mlp.{module_name}")
        assert module is not None
