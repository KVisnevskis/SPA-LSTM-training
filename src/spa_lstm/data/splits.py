"""Split validation utilities."""

from __future__ import annotations


def _find_duplicates(items: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return sorted(duplicates)


def assert_no_duplicate_runs(train_runs: list[str], val_runs: list[str], eval_runs: list[str]) -> None:
    """Ensure no split list contains duplicate run keys."""

    train_dup = _find_duplicates(train_runs)
    val_dup = _find_duplicates(val_runs)
    eval_dup = _find_duplicates(eval_runs)

    if train_dup or val_dup or eval_dup:
        raise ValueError(
            "Duplicate run keys detected within split lists. "
            f"train={train_dup}, val={val_dup}, eval={eval_dup}"
        )


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
