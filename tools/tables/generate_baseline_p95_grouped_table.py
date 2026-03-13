#!/usr/bin/env python3
"""Generate a grouped LaTeX table for per-run baseline P95 absolute error (deg).

This script consumes CSV produced by:
`tools/tables/compute_baseline_p95_per_run.py`
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


MODEL_IDS: tuple[str, ...] = ("SLM-LSTM", "SLU-LSTM", "TLM-LSTM", "TLU-LSTM")

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
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _load_p95_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"model_id", "run_key", "split_role", "p95_abs_error_deg"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Input CSV missing required columns {sorted(required)}")
        return [row for row in reader if isinstance(row, dict)]


def _build_rows(input_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    per_model: dict[str, dict[str, tuple[str, float]]] = {model: {} for model in MODEL_IDS}

    for row in input_rows:
        model_id = str(row.get("model_id", "")).strip()
        run_key = str(row.get("run_key", "")).strip()
        split_role = str(row.get("split_role", "")).strip().lower()
        p95 = _safe_float(row.get("p95_abs_error_deg"))
        if model_id not in per_model or not run_key or p95 is None:
            continue
        if split_role not in SPLIT_ROLE_ORDER:
            continue
        per_model[model_id][run_key] = (split_role, p95)

    run_names = sorted({run for m in per_model.values() for run in m})
    merged_rows: list[dict[str, Any]] = []
    for run_name in run_names:
        roles = {
            model_map[run_name][0]
            for model_map in per_model.values()
            if run_name in model_map
        }
        if not roles:
            continue
        if len(roles) > 1:
            raise ValueError(f"Inconsistent split_role across models for run '{run_name}': {roles}")
        split_role = next(iter(roles))

        row: dict[str, Any] = {"run_name": run_name, "split_role": split_role}
        for model in MODEL_IDS:
            row[model] = per_model[model].get(run_name, (split_role, float("nan")))[1]
        merged_rows.append(row)

    merged_rows.sort(key=lambda r: (SPLIT_ROLE_ORDER[str(r["split_role"])], str(r["run_name"]).lower()))
    return merged_rows


def _render_longtable(
    rows: list[dict[str, Any]],
    caption: str,
    label: str,
    decimals: int,
    colorize: bool,
) -> str:
    col_spec = "lrrrr"
    header_top = r"Run name & \multicolumn{4}{c}{$P_{95}(|e|)$ [deg]} \\"
    header_bottom = r"& SLM-LSTM & SLU-LSTM & TLM-LSTM & TLU-LSTM \\"
    model_cols = list(MODEL_IDS)

    non_outlier_values = [
        float(row[col])
        for row in rows
        for col in model_cols
        if isinstance(row.get(col), (int, float))
        and math.isfinite(float(row[col]))
        and float(row[col]) <= 100.0
    ]
    if not non_outlier_values:
        raise ValueError("No non-outlier P95 values available for color scaling.")
    min_non_outlier = min(non_outlier_values)
    max_non_outlier = max(non_outlier_values)

    def _format_cell(value: float) -> str:
        value_text = f"{value:.{decimals}f}"
        if not colorize or not math.isfinite(value):
            return value_text
        if value > 100.0:
            return rf"\cellcolor{{P95Anomaly}}{value_text}"
        if max_non_outlier <= min_non_outlier:
            norm = 0.0
        else:
            norm = (value - min_non_outlier) / (max_non_outlier - min_non_outlier)
        norm = max(0.0, min(1.0, norm))
        color_pct = int(round(norm * 100))
        return rf"\cellcolor{{P95Low!{color_pct}!P95High}}{value_text}"

    lines: list[str] = []
    if colorize:
        lines.append(r"% Requires: \usepackage[table]{xcolor}")
        lines.append(
            r"% Colorblind-friendly, softer blue->orange scale. "
            r"Outliers (>100 deg) highlighted in contrasting yellow."
        )
        lines.append(r"% Non-outlier gradient bounds are min..max over values <= 100 deg.")
        lines.append(r"\definecolor{P95Low}{HTML}{E8F1F2}")
        lines.append(r"\definecolor{P95High}{HTML}{E38B5B}")
        lines.append(r"\definecolor{P95Anomaly}{HTML}{FFF176}")

    lines.append(r"\begin{longtable}{" + col_spec + "}")
    lines.append(r"\caption{" + _latex_escape(caption) + r"}\label{" + _latex_escape(label) + r"} \\")
    lines.append(r"\toprule")
    lines.append(header_top)
    lines.append(r"\cmidrule(lr){2-5}")
    lines.append(header_bottom)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(header_top)
    lines.append(r"\cmidrule(lr){2-5}")
    lines.append(header_bottom)
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
            lines.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{section_text}}}}} \\")
            lines.append(r"\midrule")

        lines.append(
            "{} & {} & {} & {} & {} \\\\".format(
                _latex_escape(str(row["run_name"])),
                _format_cell(float(row["SLM-LSTM"])),
                _format_cell(float(row["SLU-LSTM"])),
                _format_cell(float(row["TLM-LSTM"])),
                _format_cell(float(row["TLU-LSTM"])),
            )
        )

    lines.append(r"\end{longtable}")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate grouped LaTeX table for per-run baseline P95 absolute error."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help=f"Repository root path (default: {default_repo_root}).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tables/tab_ch5/baseline_p95_per_run_all_models.csv"),
        help="Input CSV from compute_baseline_p95_per_run.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tables/tab_ch5/baseline_p95_per_run_grouped.tex"),
        help="Output .tex path.",
    )
    parser.add_argument(
        "--caption",
        default="Per-run P95 absolute error [deg] for baseline models grouped by run type.",
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:baseline-p95-per-run-grouped",
        help="LaTeX table label.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places for displayed P95 values (default: 2).",
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
    input_path = args.input if args.input.is_absolute() else (repo_root / args.input)
    output_path = args.output if args.output.is_absolute() else (repo_root / args.output)

    input_rows = _load_p95_rows(input_path)
    rows = _build_rows(input_rows)
    table_tex = _render_longtable(
        rows=rows,
        caption=args.caption,
        label=args.label,
        decimals=args.decimals,
        colorize=(not args.no_color),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_tex, encoding="utf-8")
    print(f"Wrote table: {output_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
