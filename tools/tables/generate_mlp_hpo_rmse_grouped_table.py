#!/usr/bin/env python3
"""Generate a grouped per-run RMSE table for the MLP HPO study.

The study definition is taken from the experiment configs under
`configs/experiments/mlp/hpo`. For each matching config, the helper reads
`eval_metrics_all_runs.json` from the corresponding run directory under
`outputs/experiments/mlp/hpo`, converts RMSE to degrees, and writes a grouped
LaTeX longtable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


CONFIG_NAME_RE = re.compile(r"^baseline_mlp__(?P<arch>u\d+(?:_u\d+)*)__lr(?P<lr>\d+e\d+)$")

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
LR_ORDER = {"1e3": 0, "3e4": 1}


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


def _arch_label(arch_token: str) -> str:
    parts = [str(int(part[1:])) for part in arch_token.split("_")]
    return "-".join(parts)


def _lr_label(lr_token: str) -> str:
    if lr_token == "1e3":
        return "1e-3"
    if lr_token == "3e4":
        return "3e-4"
    raise ValueError(f"Unsupported learning-rate token: {lr_token}")


def _discover_model_runs(config_dir: Path, run_root: Path) -> list[tuple[tuple[int, tuple[int, ...]], str, Path]]:
    discovered: list[tuple[tuple[int, tuple[int, ...]], str, Path]] = []
    for config_path in sorted(config_dir.glob("baseline_mlp__u*__lr*.yaml")):
        match = CONFIG_NAME_RE.match(config_path.stem)
        if match is None:
            continue

        arch_token = match.group("arch")
        lr_token = match.group("lr")
        arch_parts = tuple(int(part[1:]) for part in arch_token.split("_"))
        lr_sort = LR_ORDER.get(lr_token)
        if lr_sort is None:
            raise ValueError(f"Unsupported learning-rate token in config name: {config_path.stem}")

        model_label = f"{_arch_label(arch_token)}@{_lr_label(lr_token)}"
        run_dir = run_root / config_path.stem
        if not run_dir.exists():
            raise FileNotFoundError(
                f"Expected run directory for config {config_path} at {run_dir}, but it does not exist."
            )
        discovered.append(((lr_sort, arch_parts), model_label, run_dir))

    if not discovered:
        raise ValueError(f"No matching MLP HPO configs found in {config_dir}")
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
    per_model: dict[str, dict[str, tuple[str, float]]] = {}
    model_labels: list[str] = []

    for _, model_label, run_dir in _discover_model_runs(config_dir=config_dir, run_root=run_root):
        model_labels.append(model_label)
        metrics = _load_json(run_dir / "eval_metrics_all_runs.json")
        if not isinstance(metrics, list):
            raise ValueError(f"Expected JSON list in {run_dir / 'eval_metrics_all_runs.json'}")
        metric_rows = [row for row in metrics if isinstance(row, dict)]
        per_model[model_label] = _extract_per_run(metric_rows)

    run_names = sorted({run_name for model_map in per_model.values() for run_name in model_map})
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
        for model_label in model_labels:
            row[model_label] = per_model[model_label].get(run_name, (split_role, float("nan")))[1]
        merged_rows.append(row)

    merged_rows.sort(key=lambda row: (SPLIT_ROLE_ORDER[str(row["split_role"])], str(row["run_name"]).lower()))
    return model_labels, merged_rows


def _display_label(model_label: str) -> str:
    arch, lr = model_label.split("@", maxsplit=1)
    arch = arch.replace("-", "/")
    if lr == "1e-3":
        return arch
    if lr == "3e-4":
        return f"{arch}*"
    raise ValueError(f"Unsupported model label: {model_label}")


def _render_longtable(
    model_labels: list[str],
    rows: list[dict[str, Any]],
    caption: str,
    label: str,
    decimals: int,
    colorize: bool,
) -> str:
    col_spec = "l" + ("r" * len(model_labels))
    display_labels = [_display_label(model_label) for model_label in model_labels]
    top_header = rf"Run name & \multicolumn{{{len(model_labels)}}}{{c}}{{RMSE [deg]}} \\"
    bottom_header = "& " + " & ".join(_latex_escape(label) for label in display_labels) + r" \\"
    cmidrule = rf"\cmidrule(lr){{2-{len(model_labels) + 1}}}"

    non_outlier_values = [
        float(row[model_label])
        for row in rows
        for model_label in model_labels
        if isinstance(row.get(model_label), (int, float))
        and math.isfinite(float(row[model_label]))
        and float(row[model_label]) <= 100.0
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
        lines.append(r"% Compact model labels use `*` for lr=3e-4; unstarred labels use lr=1e-3.")
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
            lines.append(rf"\multicolumn{{{len(model_labels) + 1}}}{{l}}{{\textbf{{{section_text}}}}} \\")
            lines.append(r"\midrule")

        rendered_cells = " & ".join(_format_cell(float(row[model_label])) for model_label in model_labels)
        lines.append(f"{_latex_escape(str(row['run_name']))} & {rendered_cells} \\\\")

    lines.append(r"\end{longtable}")
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, model_labels: list[str], rows: list[dict[str, Any]]) -> None:
    fieldnames = ["run_name", "split_role", *model_labels]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row: dict[str, Any] = {
                "run_name": row["run_name"],
                "split_role": row["split_role"],
            }
            for model_label in model_labels:
                value = row.get(model_label, float("nan"))
                out_row[model_label] = (
                    f"{float(value):.6f}" if isinstance(value, (int, float)) and math.isfinite(float(value)) else ""
                )
            writer.writerow(out_row)


def _parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate a grouped LaTeX table comparing RMSE across MLP HPO runs."
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
        default=Path("configs/experiments/mlp/hpo"),
        help="Directory containing the MLP HPO experiment configs.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("outputs/experiments/mlp/hpo"),
        help="Directory containing the MLP HPO run directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables/ch4/mlp_hpo_rmse_per_run_grouped.tex"),
        help="Output .tex path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/tables/ch4/mlp_hpo_rmse_per_run_grouped.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--caption",
        default="Per-run RMSE [deg] across the MLP HPO runs grouped by run type.",
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:mlp-hpo-rmse-per-run-grouped",
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
    output_csv = args.output_csv if args.output_csv.is_absolute() else (repo_root / args.output_csv)

    model_labels, rows = _build_rows(config_dir=config_dir, run_root=run_root)
    table_tex = _render_longtable(
        model_labels=model_labels,
        rows=rows,
        caption=args.caption,
        label=args.label,
        decimals=args.decimals,
        colorize=(not args.no_color),
    )

    _write_csv(output_csv, model_labels=model_labels, rows=rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_tex, encoding="utf-8")
    print(f"Wrote CSV: {output_csv}")
    print(f"Wrote table: {output_path}")
    print(f"Rows: {len(rows)}")
    print(f"Models: {', '.join(model_labels)}")


if __name__ == "__main__":
    main()
