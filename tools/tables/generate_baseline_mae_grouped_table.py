#!/usr/bin/env python3
"""Generate a LaTeX table with per-run MAE for baseline models.

Table layout:
1) Run name
2) SLM-LSTM MAE
3) SLU-LSTM MAE
4) TLM-LSTM MAE
5) TLU-LSTM MAE

Rows are grouped in this order:
training -> validation -> eval -> unseen
"""

from __future__ import annotations

import argparse
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

SPLIT_ROLE_LABEL: dict[str, str] = {
    "train": "training",
    "val": "validation",
    "eval": "eval",
    "unseen": "unseen",
}

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


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    rows = [row for row in data if isinstance(row, dict)]
    return rows


def _extract_per_run(rows: list[dict[str, Any]]) -> dict[str, tuple[str, float]]:
    out: dict[str, tuple[str, float]] = {}
    for row in rows:
        split_role = str(row.get("split_role", "")).strip().lower()
        if split_role not in SPLIT_ROLE_LABEL:
            continue

        run_key = str(row.get("run_key", "")).strip()
        if not run_key:
            continue

        # Repository convention: legacy *_deg field names may still store radians.
        raw_mae = row.get("mae", row.get("mae_deg"))
        if not isinstance(raw_mae, (int, float)):
            continue

        mae_deg = float(raw_mae) * RAD_TO_DEG
        out[run_key] = (split_role, mae_deg)
    return out


def _build_rows(repo_root: Path) -> list[dict[str, Any]]:
    per_model: dict[str, dict[str, tuple[str, float]]] = {}
    for model_label, run_dir_rel in MODEL_RUNS:
        metrics_path = repo_root / run_dir_rel / "eval_metrics_all_runs.json"
        rows = _load_metrics(metrics_path)
        per_model[model_label] = _extract_per_run(rows)

    run_names = sorted({run for model_map in per_model.values() for run in model_map})

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

        row: dict[str, Any] = {
            "run_name": run_name,
            "split_role": split_role,
        }
        for model_label, _ in MODEL_RUNS:
            model_map = per_model[model_label]
            row[model_label] = model_map.get(run_name, (split_role, float("nan")))[1]
        merged_rows.append(row)

    merged_rows.sort(key=lambda r: (SPLIT_ROLE_ORDER[r["split_role"]], r["run_name"].lower()))
    return merged_rows


def _render_longtable(
    rows: list[dict[str, Any]],
    caption: str,
    label: str,
    decimals: int,
    colorize: bool,
) -> str:
    col_spec = "lrrrr"
    header_top = r"Run name & \multicolumn{4}{c}{MAE [deg]} \\"
    header_bottom = r"& SLM-LSTM & SLU-LSTM & TLM-LSTM & TLU-LSTM \\"
    model_cols = [model_label for model_label, _ in MODEL_RUNS]
    non_outlier_values = [
        float(row[col])
        for row in rows
        for col in model_cols
        if isinstance(row.get(col), (int, float))
        and math.isfinite(float(row[col]))
        and float(row[col]) <= 100.0
    ]
    if not non_outlier_values:
        raise ValueError("No non-outlier MAE values available for color scaling.")
    min_non_outlier = min(non_outlier_values)
    max_non_outlier = max(non_outlier_values)

    def _format_mae_cell(value: float) -> str:
        value_text = f"{value:.{decimals}f}"
        if not colorize or not math.isfinite(value):
            return value_text
        # Outlier threshold: >100 deg.
        if value > 100.0:
            return rf"\cellcolor{{MAEAnomaly}}{value_text}"
        if max_non_outlier <= min_non_outlier:
            norm = 0.0
        else:
            norm = (value - min_non_outlier) / (max_non_outlier - min_non_outlier)
        norm = max(0.0, min(1.0, norm))
        color_pct = int(round(norm * 100))
        return rf"\cellcolor{{MAELow!{color_pct}!MAEHigh}}{value_text}"

    lines: list[str] = []
    if colorize:
        lines.append(r"% Requires: \usepackage[table]{xcolor}")
        lines.append(
            r"% Colorblind-friendly, softer blue->orange scale. "
            r"Outliers (>100 deg) highlighted in contrasting yellow."
        )
        lines.append(r"% Non-outlier gradient bounds are min..max over values <= 100 deg.")
        lines.append(r"\definecolor{MAELow}{HTML}{E8F1F2}")
        lines.append(r"\definecolor{MAEHigh}{HTML}{E38B5B}")
        lines.append(r"\definecolor{MAEAnomaly}{HTML}{FFF176}")
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
                _format_mae_cell(float(row["SLM-LSTM"])),
                _format_mae_cell(float(row["SLU-LSTM"])),
                _format_mae_cell(float(row["TLM-LSTM"])),
                _format_mae_cell(float(row["TLU-LSTM"])),
            )
        )

    lines.append(r"\end{longtable}")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Generate a LaTeX table with per-run baseline MAE values, grouped by "
            "training/validation/eval/unseen."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help=f"Repository root path (default: {default_repo_root})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables/ch4/baseline_mae_per_run_grouped.tex"),
        help="Output .tex path (default: outputs/tables/ch4/baseline_mae_per_run_grouped.tex).",
    )
    parser.add_argument(
        "--caption",
        default="Per-run MAE [deg] for baseline models grouped by run type.",
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:baseline-mae-per-run-grouped",
        help="LaTeX table label.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places for MAE values (default: 2).",
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
    output_path = args.output if args.output.is_absolute() else (repo_root / args.output)

    rows = _build_rows(repo_root=repo_root)
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
