#!/usr/bin/env python3
"""Generate a grouped per-run RMSE table for the additional-train-set SLM runs.

The table compares the three additional-train-set SLM-LSTM runs against the
current best width-ablation comparison run (`u016`, seed 7). Metrics are read
from `eval_metrics_all_runs.json`, converted to degrees, and grouped by split
role:

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

        out[run_key] = (split_role, float(raw_rmse) * RAD_TO_DEG)
    return out


def _build_column_specs(
    repo_root: Path,
    additional_run_root: Path,
    comparison_run_dir: Path,
) -> list[tuple[str, Path]]:
    additional_specs: list[tuple[str, Path]] = []
    for run_dir in sorted(path for path in additional_run_root.iterdir() if path.is_dir()):
        snapshot_path = run_dir / "config_snapshot.json"
        metrics_path = run_dir / "eval_metrics_all_runs.json"
        if not snapshot_path.exists() or not metrics_path.exists():
            continue

        snapshot = _load_json(snapshot_path)
        if not isinstance(snapshot, dict):
            continue
        training_cfg = snapshot.get("training")
        if not isinstance(training_cfg, dict):
            continue
        raw_seed = training_cfg.get("seed")
        if not isinstance(raw_seed, (int, float)):
            continue
        seed = int(raw_seed)
        additional_specs.append((f"Add. set S{seed}", run_dir))

    if not additional_specs:
        raise ValueError(f"No additional-train-set runs with metrics found in {additional_run_root}")

    comparison_dir = comparison_run_dir if comparison_run_dir.is_absolute() else (repo_root / comparison_run_dir)
    if not (comparison_dir / "eval_metrics_all_runs.json").exists():
        raise FileNotFoundError(
            f"Comparison run is missing eval_metrics_all_runs.json: {comparison_dir / 'eval_metrics_all_runs.json'}"
        )

    return additional_specs + [("Width 16", comparison_dir)]


def _build_rows(
    primary_column_labels: set[str],
    column_specs: list[tuple[str, Path]],
) -> tuple[list[str], list[dict[str, Any]]]:
    per_column: dict[str, dict[str, tuple[str, float]]] = {}
    column_labels = [label for label, _ in column_specs]

    for column_label, run_dir in column_specs:
        metrics = _load_json(run_dir / "eval_metrics_all_runs.json")
        if not isinstance(metrics, list):
            raise ValueError(f"Expected JSON list in {run_dir / 'eval_metrics_all_runs.json'}")
        metric_rows = [row for row in metrics if isinstance(row, dict)]
        per_column[column_label] = _extract_per_run(metric_rows)

    run_names = sorted({run_name for column_map in per_column.values() for run_name in column_map})
    merged_rows: list[dict[str, Any]] = []

    for run_name in run_names:
        primary_roles = {
            per_column[column_label][run_name][0]
            for column_label in column_labels
            if column_label in primary_column_labels and run_name in per_column[column_label]
        }
        all_roles = {
            per_column[column_label][run_name][0]
            for column_label in column_labels
            if run_name in per_column[column_label]
        }
        if not all_roles:
            continue
        if primary_roles:
            if len(primary_roles) > 1:
                raise ValueError(
                    f"Inconsistent split_role across primary runs for '{run_name}': {primary_roles}"
                )
            split_role = next(iter(primary_roles))
        elif len(all_roles) == 1:
            split_role = next(iter(all_roles))
        else:
            raise ValueError(f"Inconsistent split_role across compared runs for '{run_name}': {all_roles}")

        row: dict[str, Any] = {
            "run_name": run_name,
            "split_role": split_role,
        }
        for column_label in column_labels:
            row[column_label] = per_column[column_label].get(run_name, (split_role, float("nan")))[1]
        merged_rows.append(row)

    merged_rows.sort(key=lambda row: (SPLIT_ROLE_ORDER[str(row["split_role"])], str(row["run_name"]).lower()))
    return column_labels, merged_rows


def _render_longtable(
    column_labels: list[str],
    rows: list[dict[str, Any]],
    caption: str,
    label: str,
    decimals: int,
    colorize: bool,
) -> str:
    col_spec = "l" + ("r" * len(column_labels))
    top_header = rf"Run name & \multicolumn{{{len(column_labels)}}}{{c}}{{RMSE [deg]}} \\"
    bottom_header = "& " + " & ".join(_latex_escape(col) for col in column_labels) + r" \\"
    cmidrule = rf"\cmidrule(lr){{2-{len(column_labels) + 1}}}"

    non_outlier_values = [
        float(row[column_label])
        for row in rows
        for column_label in column_labels
        if isinstance(row.get(column_label), (int, float))
        and math.isfinite(float(row[column_label]))
        and float(row[column_label]) <= 100.0
    ]
    if not non_outlier_values:
        raise ValueError("No non-outlier RMSE values available for color scaling.")

    min_non_outlier = min(non_outlier_values)
    max_non_outlier = max(non_outlier_values)

    def _format_cell(value: float) -> str:
        if not math.isfinite(value):
            return "n/a"
        value_text = f"{value:.{decimals}f}"
        if not colorize:
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
            lines.append(rf"\multicolumn{{{len(column_labels) + 1}}}{{l}}{{\textbf{{{section_text}}}}} \\")
            lines.append(r"\midrule")

        rendered_cells = " & ".join(_format_cell(float(row[column_label])) for column_label in column_labels)
        lines.append(f"{_latex_escape(str(row['run_name']))} & {rendered_cells} \\\\")

    lines.append(r"\end{longtable}")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Generate a grouped LaTeX table comparing RMSE across the additional-train-set "
            "SLM-LSTM runs and the width-16 comparison run."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help=f"Repository root path (default: {default_repo_root})",
    )
    parser.add_argument(
        "--additional-run-root",
        type=Path,
        default=Path("outputs/experiments/lstm_additional_train_sets"),
        help="Directory containing the additional-train-set SLM run subdirectories.",
    )
    parser.add_argument(
        "--comparison-run-dir",
        type=Path,
        default=Path("outputs/experiments/hpo/width_ablation/slm_width_ablation__baseline__u016__seed007"),
        help="Run directory for the comparison LSTM column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables/ch4/lstm_additional_train_sets_rmse_per_run_grouped.tex"),
        help="Output .tex path.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Per-run RMSE [deg] across the additional-train-set SLM-LSTM runs, "
            "with the 16-node width-ablation LSTM included as a comparison. "
            "Rows are grouped using the split definition from the additional-train-set configs."
        ),
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:lstm-additional-train-sets-rmse-per-run-grouped",
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
    additional_run_root = (
        args.additional_run_root
        if args.additional_run_root.is_absolute()
        else (repo_root / args.additional_run_root)
    )
    comparison_run_dir = (
        args.comparison_run_dir
        if args.comparison_run_dir.is_absolute()
        else (repo_root / args.comparison_run_dir)
    )
    output_path = args.output if args.output.is_absolute() else (repo_root / args.output)

    column_specs = _build_column_specs(
        repo_root=repo_root,
        additional_run_root=additional_run_root,
        comparison_run_dir=comparison_run_dir,
    )
    primary_column_labels = {label for label, _ in column_specs[:-1]}
    column_labels, rows = _build_rows(primary_column_labels, column_specs)
    table_tex = _render_longtable(
        column_labels=column_labels,
        rows=rows,
        caption=args.caption,
        label=args.label,
        decimals=args.decimals,
        colorize=(not args.no_color),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_tex, encoding="utf-8")
    print(f"Wrote table: {output_path}")
    print(f"Columns: {', '.join(column_labels)}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
