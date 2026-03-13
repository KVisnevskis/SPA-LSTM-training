from __future__ import annotations

import pandas as pd
import pytest

from spa_lstm.data.schema import require_columns


def test_require_columns_passes_when_all_required_present() -> None:
    df = pd.DataFrame({"pressure": [1.0], "phi": [0.0]})
    require_columns(df, ["pressure", "phi"], run_key="run_ok")


def test_require_columns_raises_with_missing_columns() -> None:
    df = pd.DataFrame({"pressure": [1.0]})
    with pytest.raises(KeyError, match="run_missing"):
        require_columns(df, ["pressure", "phi"], run_key="run_missing")

