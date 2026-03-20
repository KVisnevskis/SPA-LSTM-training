#!/usr/bin/env python3
"""Generate a LaTeX table with per-run RMSE for seeded baseline runs.

This helper is aimed at the currently available seeded SLM-LSTM replication runs
under `outputs/experiments/baseline_slm_seeded_replication`, but the run root
can be overridden. By default it also includes the original baseline SLM-LSTM
run from `outputs/experiments/baseline/baseline_slm_lstm` as the `Seed 42`
column. RMSE values are read from `eval_metrics_all_runs.json`, converted to
degrees, and grouped by split role:

train -> val -> eval -> unseen
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SPLIT_ROLE_ORDER: dict[str, int] = {
    "train": 0,
    "val": 1,
    "eval": 2,
    "unseen": 3,
}

SECTION_ROW_LABEL: dict[str, str] = {
    "train": "training datasets",
    "val": "validation datasets",
    "eval": "evaluation datasets",
    "unseen": "unseen datasets",
}

RAD_TO_DEG = 180.0 / math.pi


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


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_seed_run_info(run_dir: Path) -> tuple[int, str, Path] | None:
    config_snapshot = run_dir / "config_snapshot.json"
    metrics_path = run_dir / "eval_metrics_all_runs.json"
    if not config_snapshot.exists() or not metrics_path.exists():
        return None

    snapshot = _load_json(config_snapshot)
    if not isinstance(snapshot, dict):
        return None

    training_cfg = snapshot.get("training")
    if not isinstance(training_cfg, dict):
        return None

    raw_seed = training_cfg.get("seed")
    if not isinstance(raw_seed, (int, float)):
        return None

    seed = int(raw_seed)
    return seed, f"Seed {seed}", run_dir


def _discover_seed_runs(
    seed_run_root: Path,
    base_run_dir: Path | None,
    include_base_run: bool,
) -> list[tuple[int, str, Path]]:
    discovered_by_seed: dict[int, tuple[int, str, Path]] = {}

    candidate_dirs: list[Path] = []
    if include_base_run and base_run_dir is not None:
        candidate_dirs.append(base_run_dir)
    candidate_dirs.extend(sorted(path for path in seed_run_root.iterdir() if path.is_dir()))

    for run_dir in candidate_dirs:
        info = _load_seed_run_info(run_dir)
        if info is None:
            continue
        seed, _, _ = info
        if seed in discovered_by_seed:
            other_dir = discovered_by_seed[seed][2]
            raise ValueError(
                f"Duplicate seeded run detected for seed {seed}: {other_dir} and {run_dir}"
            )
        discovered_by_seed[seed] = info

    if not discovered_by_seed:
        raise ValueError(f"No seeded run directories found in {seed_run_root}")
    return sorted(discovered_by_seed.values(), key=lambda item: item[0])


def _extract_per_run(metrics_rows: list[dict[str, Any]]) -> dict[str, tuple[str, float]]:
    out: dict[str, tuple[str, float]] = {}
    for row in metrics_rows:
        split_role = str(row.get("split_role", "")).strip().lower()
        if split_role not in SPLIT_ROLE_ORDER:
            continue

        run_key = str(row.get("run_key", "")).strip()
        if not run_key:
            continue

        raw_rmse = row.get("rmse", row.get("rmse_deg"))
        if not isinstance(raw_rmse, (int, float)):
            continue

        rmse_deg = float(raw_rmse) * RAD_TO_DEG
        out[run_key] = (split_role, rmse_deg)
    return out


def _build_rows(seed_runs: list[tuple[int, str, Path]]) -> tuple[list[str], list[dict[str, Any]]]:
    per_seed: dict[str, dict[str, tuple[str, float]]] = {}
    seed_labels = [label for _, label, _ in seed_runs]

    for _, seed_label, run_dir in seed_runs:
        metrics = _load_json(run_dir / "eval_metrics_all_runs.json")
        if not isinstance(metrics, list):
            raise ValueError(f"Expected JSON list in {run_dir / 'eval_metrics_all_runs.json'}")
        metric_rows = [row for row in metrics if isinstance(row, dict)]
        per_seed[seed_label] = _extract_per_run(metric_rows)

    run_names = sorted({run_name for seed_map in per_seed.values() for run_name in seed_map})
    merged_rows: list[dict[str, Any]] = []

    for run_name in run_names:
        roles = {
            seed_map[run_name][0]
            for seed_map in per_seed.values()
            if run_name in seed_map
        }
        if not roles:
            continue
        if len(roles) > 1:
            raise ValueError(f"Inconsistent split_role across seeds for run '{run_name}': {roles}")
        split_role = next(iter(roles))

        row: dict[str, Any] = {
            "run_name": run_name,
            "split_role": split_role,
        }
        for seed_label in seed_labels:
            row[seed_label] = per_seed[seed_label].get(run_name, (split_role, float("nan")))[1]
        merged_rows.append(row)

    merged_rows.sort(key=lambda row: (SPLIT_ROLE_ORDER[str(row["split_role"])], str(row["run_name"]).lower()))
    return seed_labels, merged_rows


def _render_longtable(
    seed_labels: list[str],
    rows: list[dict[str, Any]],
    caption: str,
    label: str,
    decimals: int,
    colorize: bool,
) -> str:
    col_spec = "l" + ("r" * len(seed_labels))
    top_header = rf"Run name & \multicolumn{{{len(seed_labels)}}}{{c}}{{RMSE [deg]}} \\"
    bottom_header = "& " + " & ".join(_latex_escape(seed_label) for seed_label in seed_labels) + r" \\"
    cmidrule = rf"\cmidrule(lr){{2-{len(seed_labels) + 1}}}"

    non_outlier_values = [
        float(row[seed_label])
        for row in rows
        for seed_label in seed_labels
        if isinstance(row.get(seed_label), (int, float))
        and math.isfinite(float(row[seed_label]))
        and float(row[seed_label]) <= 100.0
    ]
    if not non_outlier_values:
        raise ValueError("No non-outlier RMSE values available for color scaling.")

    min_non_outlier = min(non_outlier_values)
    max_non_outlier = max(non_outlier_values)

    def _format_cell(value: float) -> str:
        value_text = f"{value:.{decimals}f}"
        if not colorize or not math.isfinite(value):
            return value_text
        if value > 100.0:
            return rf"\cellcolor{{RMSEAnomaly}}{value_text}"
        if max_non_outlier <= min_non_outlier:
            norm = 0.0
        else:
            norm = (value - min_non_outlier) / (max_non_outlier - min_non_outlier)
        norm = max(0.0, min(1.0, norm))
        color_pct = int(round(norm * 100))
        return rf"\cellcolor{{RMSELow!{color_pct}!RMSEHigh}}{value_text}"

    lines: list[str] = []
    if colorize:
        lines.append(r"% Requires: \usepackage[table]{xcolor}")
        lines.append(
            r"% Colorblind-friendly, softer blue->orange scale. "
            r"Outliers (>100 deg) highlighted in contrasting yellow."
        )
        lines.append(r"% Non-outlier gradient bounds are min..max over values <= 100 deg.")
        lines.append(r"\definecolor{RMSELow}{HTML}{E8F1F2}")
        lines.append(r"\definecolor{RMSEHigh}{HTML}{E38B5B}")
        lines.append(r"\definecolor{RMSEAnomaly}{HTML}{FFF176}")

    lines.append(r"\begin{longtable}{" + col_spec + "}")
    lines.append(r"\caption{" + _latex_escape(caption) + r"}\label{" + _latex_escape(label) + r"} \\")
    lines.append(r"\toprule")
    lines.append(top_header)
    lines.append(cmidrule)
    lines.append(bottom_header)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(top_header)
    lines.append(cmidrule)
    lines.append(bottom_header)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")

    last_split_role: str | None = None
    for row in rows:
        split_role = str(row["split_role"])
        if split_role != last_split_role:
            last_split_role = split_role
            section_text = _latex_escape(SECTION_ROW_LABEL.get(split_role, f"{split_role} datasets"))
            lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{{len(seed_labels) + 1}}}{{l}}{{\textbf{{{section_text}}}}} \\")
            lines.append(r"\midrule")

        rendered_seed_cells = " & ".join(_format_cell(float(row[seed_label])) for seed_label in seed_labels)
        lines.append(f"{_latex_escape(str(row['run_name']))} & {rendered_seed_cells} \\\\")

    lines.append(r"\end{longtable}")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate a grouped LaTeX table comparing RMSE across seeded baseline runs."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help=f"Repository root path (default: {default_repo_root})",
    )
    parser.add_argument(
        "--seed-run-root",
        type=Path,
        default=Path("outputs/experiments/baseline_slm_seeded_replication"),
        help="Directory containing seeded run subdirectories.",
    )
    parser.add_argument(
        "--base-run-dir",
        type=Path,
        default=Path("outputs/experiments/baseline/baseline_slm_lstm"),
        help="Original baseline SLM-LSTM run directory to include as Seed 42.",
    )
    parser.add_argument(
        "--include-base-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the original baseline SLM-LSTM run as an additional seed column (default: true).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables/ch4/baseline_slm_seeded_rmse_per_run_grouped.tex"),
        help="Output .tex path.",
    )
    parser.add_argument(
        "--caption",
        default="Per-run RMSE [deg] across seeded SLM-LSTM baseline runs grouped by run type.",
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:baseline-slm-seeded-rmse-per-run-grouped",
        help="LaTeX table label.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places for displayed RMSE values (default: 2).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable cell background gradient coloring.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    seed_run_root = args.seed_run_root if args.seed_run_root.is_absolute() else (repo_root / args.seed_run_root)
    base_run_dir = args.base_run_dir if args.base_run_dir.is_absolute() else (repo_root / args.base_run_dir)
    output_path = args.output if args.output.is_absolute() else (repo_root / args.output)

    seed_runs = _discover_seed_runs(
        seed_run_root=seed_run_root,
        base_run_dir=base_run_dir,
        include_base_run=bool(args.include_base_run),
    )
    seed_labels, rows = _build_rows(seed_runs)
    table_tex = _render_longtable(
        seed_labels=seed_labels,
        rows=rows,
        caption=args.caption,
        label=args.label,
        decimals=args.decimals,
        colorize=(not args.no_color),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_tex, encoding="utf-8")
    print(f"Wrote table: {output_path}")
    print(f"Seed columns: {', '.join(seed_labels)}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
