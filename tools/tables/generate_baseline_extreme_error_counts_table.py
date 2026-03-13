#!/usr/bin/env python3
"""Generate LaTeX table for extreme error event counts across all datasets.

This script computes, for each baseline model, the number and percentage of
samples where absolute prediction error exceeds a given threshold (default: 360 deg)
using all-runs evaluation artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_RUNS: tuple[tuple[str, str], ...] = (
    ("SLM-LSTM", "outputs/experiments/baseline/baseline_slm_lstm"),
    ("SLU-LSTM", "outputs/experiments/baseline/baseline_slu_lstm"),
    ("TLM-LSTM", "outputs/experiments/baseline/baseline_tlm_lstm"),
    ("TLU-LSTM", "outputs/experiments/baseline/baseline_tlu_lstm"),
)

RAD_TO_DEG = 180.0 / math.pi
VALID_SPLIT_ROLES = {"train", "val", "eval", "unseen"}


@dataclass(frozen=True)
class ExtremeErrorStats:
    model_id: str
    n_runs: int
    n_samples_total: int
    n_exceed: int
    pct_exceed: float


def _latex_escape(text: str) -> str:
    escaped = text
    replacements = (
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    )
    for old, new in replacements:
        escaped = escaped.replace(old, new)
    return escaped


def _safe_float(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _load_metrics_rows(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {metrics_path}")
    return [row for row in data if isinstance(row, dict)]


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


def _run_keys_from_metrics(rows: list[dict[str, Any]], split_roles: tuple[str, ...]) -> list[str]:
    allowed = set(split_roles)
    seen: set[str] = set()
    run_keys: list[str] = []
    for row in rows:
        split_role = str(row.get("split_role", "")).strip().lower()
        if split_role not in allowed:
            continue
        run_key = str(row.get("run_key", "")).strip()
        if not run_key or run_key in seen:
            continue
        seen.add(run_key)
        run_keys.append(run_key)
    return run_keys


def _prediction_csv_path(repo_root: Path, run_dir: Path, run_key: str) -> Path:
    # Standard location for all-runs prediction traces.
    return run_dir / "predictions_all_runs" / f"{run_key}.csv"


def _count_extreme_errors_in_csv(csv_path: Path, threshold_deg: float) -> tuple[int, int]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {csv_path}")

    n_total = 0
    n_exceed = 0
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
            # Values are stored in radians despite legacy *_deg names.
            abs_err_deg = abs((yp - yt) * RAD_TO_DEG)
            n_total += 1
            if abs_err_deg > threshold_deg:
                n_exceed += 1
    return n_total, n_exceed


def _compute_stats_for_model(
    repo_root: Path,
    model_id: str,
    run_dir_rel: str,
    threshold_deg: float,
    split_roles: tuple[str, ...],
) -> ExtremeErrorStats:
    run_dir = repo_root / run_dir_rel
    metrics_rows = _load_metrics_rows(run_dir / "eval_metrics_all_runs.json")
    run_keys = _run_keys_from_metrics(metrics_rows, split_roles=split_roles)
    if not run_keys:
        raise ValueError(f"No run keys found for model: {model_id}")

    total_samples = 0
    exceed_samples = 0
    for run_key in run_keys:
        csv_path = _prediction_csv_path(repo_root=repo_root, run_dir=run_dir, run_key=run_key)
        n_total, n_exceed = _count_extreme_errors_in_csv(csv_path, threshold_deg=threshold_deg)
        total_samples += n_total
        exceed_samples += n_exceed

    pct_exceed = (100.0 * exceed_samples / total_samples) if total_samples > 0 else float("nan")
    return ExtremeErrorStats(
        model_id=model_id,
        n_runs=len(run_keys),
        n_samples_total=total_samples,
        n_exceed=exceed_samples,
        pct_exceed=pct_exceed,
    )


def _render_latex_table(
    stats: list[ExtremeErrorStats],
    threshold_deg: float,
    caption: str,
    label: str,
) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Model & Samples with $|e| > "
        + f"{threshold_deg:.0f}"
        + r"^\circ$ & Total samples & Percentage (\%) \\"
    )
    lines.append(r"\midrule")
    for row in stats:
        lines.append(
            "{} & {} & {} & {:.3f} \\\\".format(
                _latex_escape(row.model_id),
                f"{row.n_exceed:,}",
                f"{row.n_samples_total:,}",
                row.pct_exceed,
            )
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Generate a LaTeX table of extreme error counts "
            "(|prediction error| above threshold) for baseline models."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help=f"Repository root path (default: {default_repo_root}).",
    )
    parser.add_argument(
        "--threshold-deg",
        type=float,
        default=360.0,
        help="Absolute error threshold in degrees (default: 360).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables/ch4/baseline_extreme_error_counts_360deg.tex"),
        help="Output .tex table path.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Counts and percentages of samples with absolute prediction error "
            "exceeding $360^\\circ$ across evaluation and unseen datasets."
        ),
        help="Table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:ch4_extreme_error_counts",
        help="Table label.",
    )
    parser.add_argument(
        "--split-roles",
        default="eval,unseen",
        help="Comma-separated split roles to include (default: eval,unseen).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.threshold_deg <= 0:
        raise ValueError("--threshold-deg must be positive.")

    repo_root = args.repo_root.resolve()
    output_path = args.output if args.output.is_absolute() else (repo_root / args.output)
    split_roles = _parse_split_roles(args.split_roles)

    stats = [
        _compute_stats_for_model(
            repo_root=repo_root,
            model_id=model_id,
            run_dir_rel=run_dir_rel,
            threshold_deg=args.threshold_deg,
            split_roles=split_roles,
        )
        for model_id, run_dir_rel in MODEL_RUNS
    ]

    table_tex = _render_latex_table(
        stats=stats,
        threshold_deg=args.threshold_deg,
        caption=args.caption,
        label=args.label,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_tex, encoding="utf-8")

    print(f"Wrote table: {output_path}")
    for row in stats:
        print(
            f"{row.model_id}: exceed={row.n_exceed:,}, total={row.n_samples_total:,}, "
            f"pct={row.pct_exceed:.6f}%"
        )


if __name__ == "__main__":
    main()
