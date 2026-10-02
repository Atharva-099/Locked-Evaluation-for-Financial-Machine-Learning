"""Build compact publication tables and a locked-result comparison figure."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


TASKS = {
    "volatility": {
        "title": "Volatility: QLIKE difference vs HAR",
        "metric": "qlike",
        "baseline": "har",
        "lower_better": True,
    },
    "returns": {
        "title": "Returns: rank IC difference vs reversal",
        "metric": "rank_ic",
        "baseline": "reversal",
        "lower_better": False,
    },
    "delisting": {
        "title": "Delisting: log-loss difference vs base rate",
        "metric": "log_loss",
        "baseline": "base_rate",
        "lower_better": True,
    },
}

MODEL_LABELS = {
    "base_rate": "Base rate",
    "catboost": "CatBoost",
    "extratrees": "ExtraTrees",
    "har": "HAR",
    "lightgbm": "LightGBM",
    "logit": "Logit",
    "mlp": "MLP",
    "persistence": "Persistence",
    "reversal": "Reversal",
    "ridge": "Ridge",
    "xgboost": "XGBoost",
    "zero": "Zero",
}


def _locked_report(root: Path, task: str) -> Path:
    reports = sorted((root / task).glob("*-locked/report"))
    if len(reports) != 1:
        raise RuntimeError(f"expected one locked {task} report under {root}, found {len(reports)}")
    return reports[0]


def _read_json(path: Path) -> list[dict]:
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise TypeError(f"expected a list in {path}")
    return value


def model_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for task in TASKS:
        report = _locked_report(root, task)
        for item in _read_json(report / "model_summary.json"):
            common = {
                "task": task,
                "model": item["model"],
                "n_months": item["n_months"],
                "n_examples": item.get("n_examples", ""),
            }
            if task == "volatility":
                values = {"metric": "QLIKE", "estimate": item["qlike"], "ci_lo": "", "ci_hi": ""}
            elif task == "returns":
                values = {
                    "metric": "Mean monthly rank IC",
                    "estimate": item["mean_rank_ic"],
                    "ci_lo": item["rank_ic_ci_lo"],
                    "ci_hi": item["rank_ic_ci_hi"],
                }
            else:
                values = {
                    "metric": "Mean monthly log loss",
                    "estimate": item["mean_monthly_log_loss"],
                    "ci_lo": item["log_loss_ci_lo"],
                    "ci_hi": item["log_loss_ci_hi"],
                }
            rows.append(common | values)
    return rows


def comparison_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for task, spec in TASKS.items():
        report = _locked_report(root, task)
        for item in _read_json(report / "comparisons.json"):
            if item["metric"] != spec["metric"] or item["baseline"] != spec["baseline"]:
                continue
            if task == "volatility":
                estimate = item["mean_estimate"]
                ci_lo = item["mean_ci_lo"]
                ci_hi = item["mean_ci_hi"]
                p_value = item["mean_p_value"]
                p_adjusted = item.get("mean_p_holm", "")
                n_months = item["mean_n_months"]
            else:
                estimate = item["estimate"]
                ci_lo = item["ci_lo"]
                ci_hi = item["ci_hi"]
                p_value = item["p_value"]
                p_adjusted = ""
                n_months = item["n_months"]
            rows.append({
                "task": task,
                "model": item["model"],
                "baseline": item["baseline"],
                "metric": item["metric"],
                "estimate": estimate,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "p_value": p_value,
                "p_value_holm": p_adjusted,
                "n_months": n_months,
                "lower_is_better": spec["lower_better"],
            })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _scale(value: float, low: float, high: float, left: float, width: float) -> float:
    return left + (value - low) / (high - low) * width


def write_figure(path: Path, rows: list[dict]) -> None:
    panel_width, panel_height = 440, 390
    width, height = panel_width * len(TASKS), panel_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.title{font-size:16px;font-weight:700}.label{font-size:13px}.axis{font-size:11px;fill:#4b5563}.note{font-size:11px;fill:#374151}</style>',
    ]
    colors = {"volatility": "#0072B2", "returns": "#E69F00", "delisting": "#009E73"}
    for panel, (task, spec) in enumerate(TASKS.items()):
        task_rows = [row for row in rows if row["task"] == task]
        task_rows.sort(key=lambda row: float(row["estimate"]))
        all_values = [0.0] + [float(row[key]) for row in task_rows for key in ("ci_lo", "ci_hi")]
        low, high = min(all_values), max(all_values)
        pad = max((high - low) * 0.12, 0.01)
        low, high = low - pad, high + pad
        x0 = panel * panel_width
        plot_left, plot_width = x0 + 118, 292
        top, row_gap = 72, 38
        zero_x = _scale(0.0, low, high, plot_left, plot_width)
        title = html.escape(spec["title"])
        parts.append(f'<text class="title" x="{x0 + 18}" y="28">{title}</text>')
        parts.append(f'<line x1="{zero_x:.1f}" y1="52" x2="{zero_x:.1f}" y2="{top + row_gap * len(task_rows)}" stroke="#9ca3af" stroke-dasharray="4 4"/>')
        for index, row in enumerate(task_rows):
            y = top + index * row_gap
            estimate = float(row["estimate"])
            ci_lo, ci_hi = float(row["ci_lo"]), float(row["ci_hi"])
            lo_x = _scale(ci_lo, low, high, plot_left, plot_width)
            hi_x = _scale(ci_hi, low, high, plot_left, plot_width)
            est_x = _scale(estimate, low, high, plot_left, plot_width)
            label = html.escape(MODEL_LABELS.get(row["model"], row["model"]))
            parts.extend([
                f'<text class="label" x="{x0 + 18}" y="{y + 4}">{label}</text>',
                f'<line x1="{lo_x:.1f}" y1="{y}" x2="{hi_x:.1f}" y2="{y}" stroke="{colors[task]}" stroke-width="2"/>',
                f'<line x1="{lo_x:.1f}" y1="{y - 5}" x2="{lo_x:.1f}" y2="{y + 5}" stroke="{colors[task]}"/>',
                f'<line x1="{hi_x:.1f}" y1="{y - 5}" x2="{hi_x:.1f}" y2="{y + 5}" stroke="{colors[task]}"/>',
                f'<circle cx="{est_x:.1f}" cy="{y}" r="4.5" fill="{colors[task]}"/>',
            ])
        axis_y = top + row_gap * len(task_rows) + 12
        parts.extend([
            f'<line x1="{plot_left}" y1="{axis_y}" x2="{plot_left + plot_width}" y2="{axis_y}" stroke="#4b5563"/>',
            f'<text class="axis" x="{plot_left}" y="{axis_y + 18}" text-anchor="start">{low:.3f}</text>',
            f'<text class="axis" x="{plot_left + plot_width}" y="{axis_y + 18}" text-anchor="end">{high:.3f}</text>',
            f'<text class="note" x="{x0 + 18}" y="{height - 20}">Point estimate and 95% moving-block interval; vertical line = no difference.</text>',
        ])
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard-data", type=Path, default=Path("dashboard_data"))
    parser.add_argument("--output", type=Path, default=Path("paper_assets"))
    args = parser.parse_args()

    models = model_rows(args.dashboard_data)
    comparisons = comparison_rows(args.dashboard_data)
    write_csv(args.output / "tables" / "locked_model_results.csv", models)
    write_csv(args.output / "tables" / "locked_primary_comparisons.csv", comparisons)
    write_figure(args.output / "figures" / "locked_primary_comparisons.svg", comparisons)
    print(f"Wrote {len(models)} model rows and {len(comparisons)} comparisons to {args.output}")


if __name__ == "__main__":
    main()
