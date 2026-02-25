from __future__ import annotations

import pytest

from spa_lstm.data.splits import assert_disjoint_splits, assert_no_duplicate_runs


def test_split_overlap_is_rejected() -> None:
    with pytest.raises(ValueError):
        assert_disjoint_splits(["run_a"], ["run_a"], ["run_c"])


def test_disjoint_splits_pass() -> None:
    assert_disjoint_splits(["run_a"], ["run_b"], ["run_c"])


def test_duplicate_runs_in_split_are_rejected() -> None:
    with pytest.raises(ValueError):
        assert_no_duplicate_runs(["run_a", "run_a"], ["run_b"], ["run_c"])
