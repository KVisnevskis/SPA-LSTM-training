#!/usr/bin/env python3
"""Compute per-run P95 absolute error (degrees) for baseline models.

Reads:
- outputs/experiments/baseline/*/eval_metrics_all_runs.json
- outputs/experiments/baseline/*/predictions_all_runs/*.csv

Writes a CSV with one row per (model, run_key).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


MODEL_RUNS: tuple[tuple[str, str], ...] = (
    ("SLM-LSTM", "outputs/experiments/baseline/baseline_slm_lstm"),
    ("SLU-LSTM", "outputs/experiments/baseline/baseline_slu_lstm"),
    ("TLM-LSTM", "outputs/experiments/baseline/baseline_tlm_lstm"),
    ("TLU-LSTM", "outputs/experiments/baseline/baseline_tlu_lstm"),
)

VALID_SPLIT_ROLES = {"train", "val", "eval", "unseen"}
SPLIT_ROLE_ORDER = {"train": 0, "val": 1, "eval": 2, "unseen": 3}
RAD_TO_DEG = 180.0 / math.pi


def _safe_float(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    vals = sorted(values)
    pos = (pct / 100.0) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _parse_split_roles(raw: str) -> tuple[str, ...]:
    roles = tuple(role.strip().lower() for role in raw.split(",") if role.strip())
    if not roles:
        raise ValueError("No split roles selected.")
    invalid = sorted(set(roles) - VALID_SPLIT_ROLES)
    if invalid:
        raise ValueError(
            f"Invalid split roles: {invalid}. Allowed: {sorted(VALID_SPLIT_ROLES)}"
        )
    return roles


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return [row for row in data if isinstance(row, dict)]


def _read_abs_errors_deg(csv_path: Path) -> list[float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {csv_path}")

    errors: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"phi_true_deg", "phi_pred_deg"}
        fieldnames = set(reader.fieldnames or [])
        if not required.issubset(fieldnames):
            raise ValueError(f"CSV missing required columns {sorted(required)}: {csv_path}")

        for row in reader:
            yt = _safe_float(row.get("phi_true_deg"))
            yp = _safe_float(row.get("phi_pred_deg"))
            if yt is None or yp is None:
                continue
            # Stored in radians despite legacy *_deg names.
            abs_err_deg = abs((yp - yt) * RAD_TO_DEG)
            errors.append(abs_err_deg)
    return errors


def _compute_rows_for_model(
    repo_root: Path,
    model_id: str,
    run_dir_rel: str,
    split_roles: tuple[str, ...],
    percentile: float,
) -> list[dict[str, Any]]:
    run_dir = repo_root / run_dir_rel
    rows = _load_metrics(run_dir / "eval_metrics_all_runs.json")
    allowed = set(split_roles)

    out_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        run_key = str(row.get("run_key", "")).strip()
        split_role = str(row.get("split_role", "")).strip().lower()
        if not run_key or run_key in seen:
            continue
        if split_role not in allowed:
            continue
        seen.add(run_key)

        csv_path = run_dir / "predictions_all_runs" / f"{run_key}.csv"
        abs_errors = _read_abs_errors_deg(csv_path)
        if not abs_errors:
            continue

        out_rows.append(
            {
                "model_id": model_id,
                "run_dir": run_dir.name,
                "run_key": run_key,
                "split_role": split_role,
                "n_samples": len(abs_errors),
                "p95_abs_error_deg": _percentile(abs_errors, percentile),
            }
        )
    return out_rows


def _parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Compute per-run baseline P95 absolute error (degrees)."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help=f"Repository root path (default: {default_repo_root}).",
    )
    parser.add_argument(
        "--split-roles",
        default="train,val,eval,unseen",
        help="Comma-separated split roles to include (default: train,val,eval,unseen).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile to compute on absolute error (default: 95).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tables/tab_ch5/baseline_p95_per_run_all_models.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not (0.0 < args.percentile < 100.0):
        raise ValueError("--percentile must be in (0, 100).")

    repo_root = args.repo_root.resolve()
    split_roles = _parse_split_roles(args.split_roles)
    output_path = args.output if args.output.is_absolute() else (repo_root / args.output)

    all_rows: list[dict[str, Any]] = []
    for model_id, run_dir_rel in MODEL_RUNS:
        all_rows.extend(
            _compute_rows_for_model(
                repo_root=repo_root,
                model_id=model_id,
                run_dir_rel=run_dir_rel,
                split_roles=split_roles,
                percentile=args.percentile,
            )
        )

    all_rows.sort(
        key=lambda r: (
            SPLIT_ROLE_ORDER.get(str(r["split_role"]), 999),
            str(r["run_key"]).lower(),
            str(r["model_id"]).lower(),
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_id",
                "run_dir",
                "run_key",
                "split_role",
                "n_samples",
                "p95_abs_error_deg",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote CSV: {output_path}")
    print(f"Rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
