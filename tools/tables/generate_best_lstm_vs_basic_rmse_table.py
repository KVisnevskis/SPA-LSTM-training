#!/usr/bin/env python3
"""Compare selected LSTM, MLP, and basic-estimator baselines.

Selection rule:
- Seeded LSTM: fixed to the seed-7 SLM-LSTM run from the seeded variability study.
- Best width-ablation LSTM: lowest mean RMSE over held-out `eval` runs from the
  baseline SLM-LSTM width-ablation study.
- Best MLP: lowest mean RMSE over held-out `eval` runs from the MLP HPO runs.
- Basic estimators: fixed to the pressure+accelerometer linear and quadratic models
  parsed from `outputs/tables/ch4/basic_estimator_RMSE_grouped.tex`.

The output is a grouped longtable with per-run RMSE [deg] across all runs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


WIDTH_CONFIG_RE = re.compile(r"^slm_width_ablation__baseline__u(?P<width>\d+)__seed(?P<seed>\d+)$")
MLP_RUN_RE = re.compile(r"^baseline_mlp__(?P<layers>u\d+(?:_u\d+)*)__lr(?P<lr>\d+e\d+)$")
SECTION_TITLE_TO_ROLE = {
    "training datasets": "train",
    "validation datasets": "val",
    "evaluation datasets": "eval",
    "unseen datasets": "unseen",
}
SPLIT_ROLE_ORDER = {
    "train": 0,
    "val": 1,
    "eval": 2,
    "unseen": 3,
}
SECTION_ROW_LABEL = {
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


def _latex_unescape(text: str) -> str:
    unescaped = text.strip()
    replacements = (
        (r"\textbackslash{}", "\\"),
        (r"\&", "&"),
        (r"\%", "%"),
        (r"\$", "$"),
        (r"\#", "#"),
        (r"\_", "_"),
        (r"\{", "{"),
        (r"\}", "}"),
        (r"\textasciitilde{}", "~"),
        (r"\textasciicircum{}", "^"),
    )
    for old, new in replacements:
        unescaped = unescaped.replace(old, new)
    return unescaped


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_per_run_from_metrics(metrics_rows: list[dict[str, Any]]) -> dict[str, tuple[str, float]]:
    per_run: dict[str, tuple[str, float]] = {}
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
        per_run[run_key] = (split_role, float(raw_rmse) * RAD_TO_DEG)
    return per_run


def _load_per_run_from_metrics(path: Path) -> dict[str, tuple[str, float]]:
    metrics = _load_json(path)
    if not isinstance(metrics, list):
        raise ValueError(f"Expected JSON list in {path}")
    return _extract_per_run_from_metrics([row for row in metrics if isinstance(row, dict)])


def _mean_eval_rmse(per_run: dict[str, tuple[str, float]]) -> float:
    eval_values = [value for split_role, value in per_run.values() if split_role == "eval"]
    if not eval_values:
        raise ValueError("Expected at least one eval RMSE value.")
    return sum(eval_values) / len(eval_values)


def _load_seeded_baseline(run_dir: Path) -> tuple[str, dict[str, tuple[str, float]]]:
    return "SLM-LSTM (seed 7)", _load_per_run_from_metrics(run_dir / "eval_metrics_all_runs.json")


def _discover_width_runs(config_dir: Path, run_root: Path) -> list[tuple[int, str, Path]]:
    discovered: list[tuple[int, str, Path]] = []
    for config_path in sorted(config_dir.glob("slm_width_ablation__baseline__u*__seed*.yaml")):
        match = WIDTH_CONFIG_RE.match(config_path.stem)
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
        raise ValueError(f"No width-ablation runs discovered from {config_dir}")
    return sorted(discovered, key=lambda item: item[0])


def _select_best_width_ablation(
    config_dir: Path,
    run_root: Path,
) -> tuple[str, dict[str, tuple[str, float]]]:
    best_label: str | None = None
    best_per_run: dict[str, tuple[str, float]] | None = None
    best_mean = float("inf")

    for width, _, run_dir in _discover_width_runs(config_dir=config_dir, run_root=run_root):
        per_run = _load_per_run_from_metrics(run_dir / "eval_metrics_all_runs.json")
        mean_eval = _mean_eval_rmse(per_run)
        if mean_eval < best_mean:
            best_mean = mean_eval
            best_label = f"SLM-LSTM (u{width})"
            best_per_run = per_run

    if best_label is None or best_per_run is None:
        raise ValueError("Could not determine the best width-ablation LSTM model.")
    return best_label, best_per_run


def _discover_mlp_runs(run_root: Path) -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []
    for run_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        if MLP_RUN_RE.match(run_dir.name) is None:
            continue
        metrics_path = run_dir / "eval_metrics_all_runs.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Expected MLP metrics at {metrics_path}, but it does not exist.")
        discovered.append((run_dir.name, run_dir))
    if not discovered:
        raise ValueError(f"No MLP runs discovered from {run_root}")
    return discovered


def _format_mlp_label(run_name: str) -> str:
    match = MLP_RUN_RE.match(run_name)
    if match is None:
        raise ValueError(f"Unexpected MLP run name format: {run_name}")

    layer_tokens = match.group("layers").split("_")
    layer_label = "/".join(str(int(token.removeprefix("u"))) for token in layer_tokens)
    lr_token = match.group("lr")
    lr_label = lr_token.replace("e", "e-")
    return f"MLP ({layer_label}, {lr_label})"


def _select_best_mlp(run_root: Path) -> tuple[str, dict[str, tuple[str, float]]]:
    best_label: str | None = None
    best_per_run: dict[str, tuple[str, float]] | None = None
    best_mean = float("inf")

    for run_name, run_dir in _discover_mlp_runs(run_root):
        per_run = _load_per_run_from_metrics(run_dir / "eval_metrics_all_runs.json")
        mean_eval = _mean_eval_rmse(per_run)
        if mean_eval < best_mean:
            best_mean = mean_eval
            best_label = _format_mlp_label(run_name)
            best_per_run = per_run

    if best_label is None or best_per_run is None:
        raise ValueError("Could not determine the best MLP model.")
    return best_label, best_per_run


def _parse_numeric_latex_cell(cell: str) -> float:
    cleaned = cell.strip()
    cleaned = re.sub(r"\\cellcolor\{[^}]+\}", "", cleaned)
    cleaned = cleaned.replace(r"\,", "")
    cleaned = cleaned.strip()
    if cleaned.lower() == "n/a" or not cleaned:
        return float("nan")
    return float(cleaned)


def _parse_basic_estimator_table(
    path: Path,
) -> tuple[list[str], dict[str, dict[str, tuple[str, float]]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing basic-estimator table: {path}")
    raw_text = path.read_text(encoding="utf-8")
    if not raw_text.strip():
        raise ValueError(
            f"Basic-estimator table is empty on disk: {path}. Save/populate it before generating the comparison table."
        )

    estimator_headers: list[str] | None = None
    current_split_role: str | None = None
    per_estimator: dict[str, dict[str, tuple[str, float]]] = {}

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue

        section_match = re.search(
            r"\\multicolumn\{\d+\}\{l\}\{\\textbf\{([^}]*)\}\}",
            line,
        )
        if section_match is not None:
            section_title = _latex_unescape(section_match.group(1)).strip().lower()
            current_split_role = SECTION_TITLE_TO_ROLE.get(section_title)
            continue

        if estimator_headers is None:
            header_match = re.match(r"^&\s*(.*?)\\\\\s*$", line)
            if header_match is not None:
                cells = [c.strip() for c in header_match.group(1).split("&")]
                if cells:
                    estimator_headers = [_latex_unescape(cell) for cell in cells]
                    for estimator_name in estimator_headers:
                        per_estimator.setdefault(estimator_name, {})
                    continue

        if current_split_role is None:
            continue
        if line.startswith("\\") or "&" not in line or not line.endswith(r"\\"):
            continue

        content = line[:-2].strip()
        cells = [cell.strip() for cell in content.split("&")]
        if estimator_headers is None or len(cells) != len(estimator_headers) + 1:
            continue

        run_name = _latex_unescape(cells[0])
        for estimator_name, raw_value in zip(estimator_headers, cells[1:]):
            value = _parse_numeric_latex_cell(raw_value)
            per_estimator[estimator_name][run_name] = (current_split_role, value)

    if estimator_headers is None:
        raise ValueError(f"Could not parse estimator headers from {path}")
    if not any(per_estimator.values()):
        raise ValueError(f"No data rows were parsed from {path}")
    return estimator_headers, per_estimator


def _select_basic_estimators(
    basic_table_path: Path,
    estimator_names: tuple[str, ...],
) -> list[tuple[str, dict[str, tuple[str, float]]]]:
    estimator_headers, per_estimator = _parse_basic_estimator_table(basic_table_path)
    selected: list[tuple[str, dict[str, tuple[str, float]]]] = []
    for estimator_name in estimator_names:
        if estimator_name not in estimator_headers:
            raise ValueError(
                f"Basic estimator '{estimator_name}' not found in {basic_table_path}. "
                f"Available columns: {estimator_headers}"
            )
        selected.append((estimator_name, per_estimator[estimator_name]))
    return selected


def _build_rows(
    column_specs: list[tuple[str, dict[str, tuple[str, float]]]],
) -> list[dict[str, Any]]:
    run_names = sorted({run_name for _, per_run in column_specs for run_name in per_run})
    rows: list[dict[str, Any]] = []
    for run_name in run_names:
        role_candidates = {per_run[run_name][0] for _, per_run in column_specs if run_name in per_run}
        if not role_candidates:
            continue
        if len(role_candidates) > 1:
            raise ValueError(f"Inconsistent split role for run '{run_name}': {sorted(role_candidates)}")
        split_role = next(iter(role_candidates))
        row: dict[str, Any] = {"run_name": run_name, "split_role": split_role}
        for label, per_run in column_specs:
            row[label] = per_run.get(run_name, (split_role, float("nan")))[1]
        rows.append(row)
    rows.sort(key=lambda row: (SPLIT_ROLE_ORDER[str(row["split_role"])], str(row["run_name"]).lower()))
    return rows


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
        float(row[col])
        for row in rows
        for col in column_labels
        if isinstance(row.get(col), (int, float))
        and math.isfinite(float(row[col]))
        and float(row[col]) <= 100.0
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

        rendered_cells = " & ".join(_format_cell(float(row[col])) for col in column_labels)
        lines.append(f"{_latex_escape(str(row['run_name']))} & {rendered_cells} \\\\")

    lines.append(r"\end{longtable}")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Generate a grouped RMSE table comparing the seed-7 SLM-LSTM, "
            "the best width-ablation LSTM, the best MLP, and selected basic estimators."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help=f"Repository root path (default: {default_repo_root}).",
    )
    parser.add_argument(
        "--width-config-dir",
        type=Path,
        default=Path("configs/experiments/hpo/width_ablation"),
        help="Directory containing the width-ablation configs.",
    )
    parser.add_argument(
        "--width-run-root",
        type=Path,
        default=Path("outputs/experiments/hpo/width_ablation"),
        help="Directory containing the width-ablation run directories.",
    )
    parser.add_argument(
        "--seeded-run-dir",
        type=Path,
        default=Path("outputs/experiments/baseline_slm_seeded_replication/baseline_slm_lstm_seed_7"),
        help="Directory containing the fixed seed-7 seeded-replication run.",
    )
    parser.add_argument(
        "--mlp-run-root",
        type=Path,
        default=Path("outputs/experiments/mlp/hpo"),
        help="Directory containing the MLP HPO run directories.",
    )
    parser.add_argument(
        "--basic-source",
        type=Path,
        default=Path("outputs/tables/ch4/basic_estimator_RMSE_grouped.tex"),
        help="Path to the grouped basic-estimator RMSE LaTeX table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables/ch4/best_lstm_vs_basic_rmse_per_run_grouped.tex"),
        help="Output .tex path.",
    )
    parser.add_argument(
        "--label",
        default="tab:best-lstm-vs-basic-rmse-per-run-grouped",
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
    width_config_dir = (
        args.width_config_dir if args.width_config_dir.is_absolute() else (repo_root / args.width_config_dir)
    )
    width_run_root = (
        args.width_run_root if args.width_run_root.is_absolute() else (repo_root / args.width_run_root)
    )
    seeded_run_dir = (
        args.seeded_run_dir if args.seeded_run_dir.is_absolute() else (repo_root / args.seeded_run_dir)
    )
    mlp_run_root = args.mlp_run_root if args.mlp_run_root.is_absolute() else (repo_root / args.mlp_run_root)
    basic_source = args.basic_source if args.basic_source.is_absolute() else (repo_root / args.basic_source)
    output_path = args.output if args.output.is_absolute() else (repo_root / args.output)

    baseline_label, baseline_per_run = _load_seeded_baseline(seeded_run_dir)
    width_label, width_per_run = _select_best_width_ablation(width_config_dir, width_run_root)
    mlp_label, mlp_per_run = _select_best_mlp(mlp_run_root)
    basic_specs = _select_basic_estimators(basic_source, estimator_names=("LPA-QR", "PA-QR"))

    column_specs = [
        (baseline_label, baseline_per_run),
        (width_label, width_per_run),
        (mlp_label, mlp_per_run),
        *basic_specs,
    ]
    rows = _build_rows(column_specs=column_specs)
    caption = (
        "Per-run RMSE [deg] comparing the fixed seeded SLM-LSTM run "
        f"({_latex_escape(baseline_label)}), the best width-ablation SLM-LSTM run "
        f"({_latex_escape(width_label)}), the best MLP run "
        f"({_latex_escape(mlp_label)}), and the LPA-QR / PA-QR basic estimators. "
        "The width-ablation LSTM and MLP models are selected by lowest mean RMSE over the "
        "held-out evaluation runs."
    )
    table_tex = _render_longtable(
        column_labels=[label for label, _ in column_specs],
        rows=rows,
        caption=caption,
        label=args.label,
        decimals=args.decimals,
        colorize=(not args.no_color),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_tex, encoding="utf-8")
    print(f"Wrote table: {output_path}")
    print(f"Seeded LSTM: {baseline_label}")
    print(f"Width-ablation best: {width_label}")
    print(f"MLP best: {mlp_label}")
    print(f"Basic estimators: {', '.join(label for label, _ in basic_specs)}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
