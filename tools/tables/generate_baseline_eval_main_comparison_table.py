#!/usr/bin/env python3
"""Generate compact baseline comparison table for held-out evaluation runs.

Table columns (rows=models):
- Mean RMSE across held-out evaluation runs
- Median RMSE across held-out evaluation runs
- Mean MAE
- Mean bias
- Mean R2
- P95AE (mean of per-run 95th percentile absolute error)
- Worst-run RMSE
- Best-on-run count

Data source:
- eval_metrics.json (run selection only)
- predictions/*.csv (metric computation)

Angles are reported in degrees; legacy `*_deg` fields are treated as radians and
converted to degrees for this table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
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


@dataclass(frozen=True)
class PerRunStats:
    run_key: str
    rmse_deg: float
    mae_deg: float
    bias_deg: float
    r2: float
    p95ae_deg: float


def _safe_float(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


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


def _load_eval_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON list in {path}")
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("split_role", "")).strip().lower() != "eval":
            continue
        out.append(row)
    if not out:
        raise ValueError(f"No eval rows found in {path}")
    return out


def _read_prediction_arrays(csv_path: Path) -> tuple[list[float], list[float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {csv_path}")

    y_true_deg: list[float] = []
    y_pred_deg: list[float] = []
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
            # Stored in radians despite legacy *_deg naming.
            y_true_deg.append(yt * RAD_TO_DEG)
            y_pred_deg.append(yp * RAD_TO_DEG)
    if not y_true_deg:
        raise ValueError(f"No usable samples in {csv_path}")
    return y_true_deg, y_pred_deg


def _compute_bias_and_r2(y_true: list[float], y_pred: list[float]) -> tuple[float, float]:
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return float("nan"), float("nan")

    yt = y_true[:n]
    yp = y_pred[:n]
    errors = [b - a for a, b in zip(yt, yp)]
    bias = statistics.fmean(errors)

    mean_true = statistics.fmean(yt)
    sse = sum((a - b) ** 2 for a, b in zip(yt, yp))
    sst = sum((a - mean_true) ** 2 for a in yt)
    if sst <= 0:
        r2 = float("nan")
    else:
        r2 = 1.0 - (sse / sst)
    return bias, r2


def _resolve_prediction_csv(repo_root: Path, run_dir: Path, row: dict[str, Any]) -> Path:
    run_key = str(row.get("run_key", "")).strip()
    raw_path = row.get("prediction_csv")
    if isinstance(raw_path, str) and raw_path.strip():
        p = Path(raw_path.strip())
        if not p.is_absolute():
            p = repo_root / p
        if p.exists():
            return p
    fallback = run_dir / "predictions" / f"{run_key}.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing prediction CSV for run '{run_key}'")


def _compute_model_run_stats(
    repo_root: Path,
    model_id: str,
    run_dir_rel: str,
) -> dict[str, PerRunStats]:
    run_dir = repo_root / run_dir_rel
    eval_rows = _load_eval_metrics(run_dir / "eval_metrics.json")

    out: dict[str, PerRunStats] = {}
    for row in eval_rows:
        run_key = str(row.get("run_key", "")).strip()
        if not run_key:
            continue

        pred_csv = _resolve_prediction_csv(repo_root=repo_root, run_dir=run_dir, row=row)
        y_true_deg, y_pred_deg = _read_prediction_arrays(pred_csv)
        n = min(len(y_true_deg), len(y_pred_deg))
        if n == 0:
            continue

        yt = y_true_deg[:n]
        yp = y_pred_deg[:n]
        errors = [pred - true for true, pred in zip(yt, yp)]
        abs_errors = [abs(err) for err in errors]
        rmse_deg = math.sqrt(sum(err * err for err in errors) / n)
        mae_deg = statistics.fmean(abs_errors)
        bias_deg, r2 = _compute_bias_and_r2(y_true_deg, y_pred_deg)

        out[run_key] = PerRunStats(
            run_key=run_key,
            rmse_deg=rmse_deg,
            mae_deg=mae_deg,
            bias_deg=bias_deg,
            r2=r2,
            p95ae_deg=_percentile(abs_errors, 95.0),
        )

    if not out:
        raise ValueError(f"No per-run stats computed for {model_id}")
    return out


def _best_on_run_counts(per_model_stats: dict[str, dict[str, PerRunStats]]) -> dict[str, int]:
    model_ids = list(per_model_stats.keys())
    common_runs = set.intersection(*(set(per_model_stats[m].keys()) for m in model_ids))
    counts = {m: 0 for m in model_ids}
    tol = 1e-12
    for run_key in sorted(common_runs):
        rmse_by_model = {m: per_model_stats[m][run_key].rmse_deg for m in model_ids}
        best = min(rmse_by_model.values())
        for m, val in rmse_by_model.items():
            if abs(val - best) <= tol:
                counts[m] += 1
    return counts


def _aggregate_row(
    model_id: str,
    run_stats: dict[str, PerRunStats],
    best_count: int,
) -> dict[str, Any]:
    rmse_vals = [s.rmse_deg for s in run_stats.values()]
    mae_vals = [s.mae_deg for s in run_stats.values()]
    bias_vals = [s.bias_deg for s in run_stats.values()]
    r2_vals = [s.r2 for s in run_stats.values() if math.isfinite(s.r2)]
    p95ae_vals = [s.p95ae_deg for s in run_stats.values() if math.isfinite(s.p95ae_deg)]

    return {
        "model_id": model_id,
        "mean_rmse_deg": statistics.fmean(rmse_vals),
        "median_rmse_deg": statistics.median(rmse_vals),
        "mean_mae_deg": statistics.fmean(mae_vals),
        "mean_bias_deg": statistics.fmean(bias_vals),
        "mean_r2": statistics.fmean(r2_vals) if r2_vals else float("nan"),
        "p95ae_deg": statistics.fmean(p95ae_vals) if p95ae_vals else float("nan"),
        "worst_run_rmse_deg": max(rmse_vals),
        "best_on_run_count": best_count,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model_id",
        "mean_rmse_deg",
        "median_rmse_deg",
        "mean_mae_deg",
        "mean_bias_deg",
        "mean_r2",
        "p95ae_deg",
        "worst_run_rmse_deg",
        "best_on_run_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(val: Any, decimals: int = 2, signed: bool = False) -> str:
    if isinstance(val, (int, float)):
        if not math.isfinite(float(val)):
            return "n/a"
        if signed:
            return f"{float(val):+.{decimals}f}"
        return f"{float(val):.{decimals}f}"
    return str(val)


def _write_tex(path: Path, rows: list[dict[str, Any]], caption: str, label: str) -> None:
    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{tabular}{lrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Model & Mean RMSE & Median RMSE & Mean MAE & Mean Bias & Mean $R^2$ & "
        r"$P_{95}(|e|)$ & Worst-run RMSE & Best-on-run count \\"
    )
    lines.append(r"\midrule")
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                _latex_escape(str(row["model_id"])),
                _fmt(row["mean_rmse_deg"]),
                _fmt(row["median_rmse_deg"]),
                _fmt(row["mean_mae_deg"]),
                _fmt(row["mean_bias_deg"], signed=True),
                _fmt(row["mean_r2"]),
                _fmt(row["p95ae_deg"]),
                _fmt(row["worst_run_rmse_deg"]),
                int(row["best_on_run_count"]),
            )
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate compact main comparison table for held-out eval runs."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help=f"Repository root path (default: {default_repo_root}).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("tables/tab_ch5/baseline_eval_main_comparison.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=Path("tables/tab_ch5/baseline_eval_main_comparison.tex"),
        help="Output LaTeX table path.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Compact comparison of baseline models on the designated held-out evaluation runs. "
            "Angle-based metrics are reported in degrees. Summary values are aggregated from "
            "per-run metrics (except median RMSE, worst-run RMSE, and best-on-run count)."
        ),
        help="LaTeX caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:baseline-eval-main-comparison",
        help="LaTeX label.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    output_csv = args.output_csv if args.output_csv.is_absolute() else (repo_root / args.output_csv)
    output_tex = args.output_tex if args.output_tex.is_absolute() else (repo_root / args.output_tex)

    per_model_stats: dict[str, dict[str, PerRunStats]] = {}
    for model_id, run_dir_rel in MODEL_RUNS:
        run_stats = _compute_model_run_stats(
            repo_root=repo_root,
            model_id=model_id,
            run_dir_rel=run_dir_rel,
        )
        per_model_stats[model_id] = run_stats

    best_counts = _best_on_run_counts(per_model_stats)

    rows: list[dict[str, Any]] = []
    for model_id, _ in MODEL_RUNS:
        rows.append(
            _aggregate_row(
                model_id=model_id,
                run_stats=per_model_stats[model_id],
                best_count=best_counts.get(model_id, 0),
            )
        )

    _write_csv(output_csv, rows)
    _write_tex(output_tex, rows, caption=args.caption, label=args.label)

    print(f"Wrote CSV: {output_csv}")
    print(f"Wrote TEX: {output_tex}")


if __name__ == "__main__":
    main()
