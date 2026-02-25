"""Split validation utilities."""

from __future__ import annotations


def assert_disjoint_splits(train_runs: list[str], val_runs: list[str], eval_runs: list[str]) -> None:
    """Ensure train/val/eval split memberships are disjoint."""

    train = set(train_runs)
    val = set(val_runs)
    eval_ = set(eval_runs)

    overlap_train_val = sorted(train & val)
    overlap_train_eval = sorted(train & eval_)
    overlap_val_eval = sorted(val & eval_)

    if overlap_train_val or overlap_train_eval or overlap_val_eval:
        raise ValueError(
            "Split overlap detected. "
            f"train∩val={overlap_train_val}, "
            f"train∩eval={overlap_train_eval}, "
            f"val∩eval={overlap_val_eval}"
        )

