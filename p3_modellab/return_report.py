"""Evaluation and plain-language report for saved Stage 8 return runs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .manifest import code_hash
from .uncertainty import block_bootstrap


def _monthly(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, month), group in predictions.groupby(["model", "yyyymm"]):
        y = group["target_ret"].to_numpy(dtype="float64")
        forecast = group["forecast"].to_numpy(dtype="float64")
        ic = np.nan
        spread = np.nan
        if len(group) >= 3 and np.unique(y).size > 1 and np.unique(forecast).size > 1:
            ic = float(spearmanr(forecast, y).statistic)
        if len(group) >= 20 and np.unique(forecast).size > 1:
            rank = pd.Series(forecast).rank(method="first", pct=True).to_numpy()
            spread = float(y[rank > 0.9].mean() - y[rank <= 0.1].mean())
        rows.append({
            "model": model, "yyyymm": int(month), "rank_ic": ic,
            "mse": float(np.mean((y - forecast) ** 2)),
            "mae": float(np.mean(np.abs(y - forecast))),
            "gross_top_bottom_decile_return": spread,
            "n": len(group),
        })
    return pd.DataFrame(rows)


def build_report(run_dir: Path, seed: int = 0) -> Path:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("task") != "returns":
        raise ValueError(f"expected a returns run, got {manifest.get('task')!r}")
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    monthly = _monthly(predictions)
    summaries = []
    for model, group in monthly.groupby("model"):
        interval = block_bootstrap(group["rank_ic"].to_numpy(), block=12, seed=seed)
        summaries.append({
            "model": model, "mean_rank_ic": interval.estimate, "rank_ic_ci_lo": interval.lo,
            "rank_ic_ci_hi": interval.hi, "rank_ic_p_value": interval.p_value,
            "mean_mse": float(group["mse"].mean()), "mean_mae": float(group["mae"].mean()),
            "mean_gross_top_bottom_decile_return": float(group["gross_top_bottom_decile_return"].mean()),
            "n_months": int(group["yyyymm"].nunique()),
        })
    summary = pd.DataFrame(summaries)

    comparisons = []
    wide_ic = monthly.pivot(index="yyyymm", columns="model", values="rank_ic")
    wide_mse = monthly.pivot(index="yyyymm", columns="model", values="mse")
    if "reversal" in wide_ic:
        for model in wide_ic.columns:
            if model == "reversal":
                continue
            diff = (wide_ic[model] - wide_ic["reversal"]).dropna()
            if diff.empty:
                continue
            interval = block_bootstrap(diff.to_numpy(), block=12, seed=seed)
            common = wide_mse[[model, "reversal"]].dropna()
            comparisons.append({
                "model": model, "baseline": "reversal", "metric": "rank_ic",
                **interval.as_dict(),
                "mse_difference": float((common[model] - common["reversal"]).mean()),
            })
    comparison = pd.DataFrame(comparisons)

    out = run_dir / "report"
    out.mkdir(exist_ok=True)
    monthly.to_parquet(out / "monthly_metrics.parquet", index=False)
    summary.to_json(out / "model_summary.json", orient="records", indent=2)
    comparison.to_json(out / "comparisons.json", orient="records", indent=2)
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": manifest["run_id"], "run_type": manifest["run_type"], "evidence": manifest["evidence"],
        "test_years_inspected": manifest["test_years"], "report_code_hash": code_hash(), "seed": seed,
        "primary_metric": "mean monthly Spearman rank information coefficient",
        "uncertainty": "95 percent moving-block bootstrap over months, 12-month blocks",
        "portfolio_note": "top-minus-bottom decile return is gross and ignores costs, turnover, capacity, and shorting constraints",
    }
    (out / "report_info.json").write_text(json.dumps(info, indent=2))
    (out / "REPORT.md").write_text(summary_markdown(info, summary, comparison))
    return out


def _number(value: float, digits: int = 4) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def summary_markdown(info: dict, summary: pd.DataFrame, comparison: pd.DataFrame) -> str:
    lines = [
        f"# Return forecast report for `{info['run_id']}`", "",
        f"- Run type: **{info['run_type']}**{'' if info['evidence'] else ' (quick functional check, not evidence)'}",
        f"- Years inspected: {info['test_years_inspected'][0]} to {info['test_years_inspected'][-1]}",
        f"- Primary metric: {info['primary_metric']}",
        f"- Uncertainty: {info['uncertainty']}", "",
        "## Overall", "",
        "| Model | Mean monthly rank IC | 95% interval | MSE | Gross top-bottom decile return |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values("mean_rank_ic", ascending=False, na_position="last").itertuples():
        lines.append(
            f"| {row.model} | {_number(row.mean_rank_ic)} | "
            f"[{_number(row.rank_ic_ci_lo)}, {_number(row.rank_ic_ci_hi)}] | "
            f"{_number(row.mean_mse)} | {_number(row.mean_gross_top_bottom_decile_return)} |"
        )
    lines += ["", "## Paired comparison with one-month reversal", ""]
    if comparison.empty:
        lines.append("No non-constant paired rank comparison was available.")
    else:
        for row in comparison.itertuples():
            lines.append(
                f"- {row.model}: rank-IC difference {_number(row.estimate)} "
                f"[{_number(row.ci_lo)}, {_number(row.ci_hi)}]; monthly MSE difference {_number(row.mse_difference)}."
            )
    lines += [
        "", "## Interpretation guardrail", "",
        "The decile spread is a gross diagnostic. It is not a tradable strategy result because transaction costs, "
        "turnover, capacity, borrow availability, and short-sale constraints are not modelled.", "",
    ]
    return "\n".join(lines)
