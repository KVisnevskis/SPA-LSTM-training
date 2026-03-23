#!/usr/bin/env python3
"""Generate a grouped per-run RMSE table for the SLM width-ablation study.

The study definition is taken from the experiment configs under
`configs/experiments/hpo/width_ablation`. For each matching config, the helper
reads `eval_metrics_all_runs.json` from the corresponding run directory under
`outputs/experiments/hpo/width_ablation`, converts RMSE to degrees, and writes
a grouped LaTeX longtable.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


CONFIG_NAME_RE = re.compile(r"^slm_width_ablation__baseline__u(?P<width>\d+)__seed(?P<seed>\d+)$")

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


def _discover_width_runs(config_dir: Path, run_root: Path) -> list[tuple[int, str, Path]]:
    discovered: list[tuple[int, str, Path]] = []
    for config_path in sorted(config_dir.glob("slm_width_ablation__baseline__u*__seed*.yaml")):
        match = CONFIG_NAME_RE.match(config_path.stem)
        if match is None:
            continue
        width = int(match.group("width"))
        run_name = config_path.stem
        run_dir = run_root / run_name
        if not run_dir.exists():
            raise FileNotFoundError(
                f"Expected run directory for config {config_path} at {run_dir}, but it does not exist."
            )
        discovered.append((width, run_name, run_dir))

    if not discovered:
        raise ValueError(f"No matching width-ablation configs found in {config_dir}")
    return sorted(discovered, key=lambda item: item[0])


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


def _build_rows(config_dir: Path, run_root: Path) -> tuple[list[str], list[dict[str, Any]]]:
    per_width: dict[str, dict[str, tuple[str, float]]] = {}
    width_labels: list[str] = []

    for width, _, run_dir in _discover_width_runs(config_dir=config_dir, run_root=run_root):
        width_label = str(width)
        width_labels.append(width_label)
        metrics = _load_json(run_dir / "eval_metrics_all_runs.json")
        if not isinstance(metrics, list):
            raise ValueError(f"Expected JSON list in {run_dir / 'eval_metrics_all_runs.json'}")
        metric_rows = [row for row in metrics if isinstance(row, dict)]
        per_width[width_label] = _extract_per_run(metric_rows)

    run_names = sorted({run_name for width_map in per_width.values() for run_name in width_map})
    merged_rows: list[dict[str, Any]] = []

    for run_name in run_names:
        roles = {
            width_map[run_name][0]
            for width_map in per_width.values()
            if run_name in width_map
        }
        if not roles:
            continue
        if len(roles) > 1:
            raise ValueError(f"Inconsistent split_role across widths for run '{run_name}': {roles}")

        split_role = next(iter(roles))
        row: dict[str, Any] = {
            "run_name": run_name,
            "split_role": split_role,
        }
        for width_label in width_labels:
            row[width_label] = per_width[width_label].get(run_name, (split_role, float("nan")))[1]
        merged_rows.append(row)

    merged_rows.sort(key=lambda row: (SPLIT_ROLE_ORDER[str(row["split_role"])], str(row["run_name"]).lower()))
    return width_labels, merged_rows


def _render_longtable(
    width_labels: list[str],
    rows: list[dict[str, Any]],
    caption: str,
    label: str,
    decimals: int,
    colorize: bool,
) -> str:
    col_spec = "l" + ("r" * len(width_labels))
    top_header = rf"Run name & \multicolumn{{{len(width_labels)}}}{{c}}{{RMSE [deg]}} \\"
    bottom_header = "& " + " & ".join(_latex_escape(width_label) for width_label in width_labels) + r" \\"
    cmidrule = rf"\cmidrule(lr){{2-{len(width_labels) + 1}}}"

    non_outlier_values = [
        float(row[width_label])
        for row in rows
        for width_label in width_labels
        if isinstance(row.get(width_label), (int, float))
        and math.isfinite(float(row[width_label]))
        and float(row[width_label]) <= 100.0
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
            lines.append(rf"\multicolumn{{{len(width_labels) + 1}}}{{l}}{{\textbf{{{section_text}}}}} \\")
            lines.append(r"\midrule")

        rendered_cells = " & ".join(_format_cell(float(row[width_label])) for width_label in width_labels)
        lines.append(f"{_latex_escape(str(row['run_name']))} & {rendered_cells} \\\\")

    lines.append(r"\end{longtable}")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate a grouped LaTeX table comparing RMSE across SLM width-ablation runs."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help=f"Repository root path (default: {default_repo_root})",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/experiments/hpo/width_ablation"),
        help="Directory containing the width-ablation experiment configs.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("outputs/experiments/hpo/width_ablation"),
        help="Directory containing the width-ablation run directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables/ch4/width_ablation_rmse_per_run_grouped.tex"),
        help="Output .tex path.",
    )
    parser.add_argument(
        "--caption",
        default="Per-run RMSE [deg] across the SLM-LSTM width-ablation runs grouped by run type.",
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:width-ablation-rmse-per-run-grouped",
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
    config_dir = args.config_dir if args.config_dir.is_absolute() else (repo_root / args.config_dir)
    run_root = args.run_root if args.run_root.is_absolute() else (repo_root / args.run_root)
    output_path = args.output if args.output.is_absolute() else (repo_root / args.output)

    width_labels, rows = _build_rows(config_dir=config_dir, run_root=run_root)
    table_tex = _render_longtable(
        width_labels=width_labels,
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
    print(f"Widths: {', '.join(width_labels)}")


if __name__ == "__main__":
    main()
