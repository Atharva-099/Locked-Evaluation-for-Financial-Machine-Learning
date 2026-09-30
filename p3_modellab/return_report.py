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


COST_BPS = (10, 25, 50)


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


def _drift(weights: pd.Series, realised_returns: pd.Series) -> pd.Series | None:
    """Move one sleeve's weights through its holding-period returns."""
    grown = weights * (1 + realised_returns.reindex(weights.index).fillna(0.0))
    total = float(grown.sum())
    if not np.isfinite(total) or total <= 0:
        return None
    return grown / total


def _turnover(current: pd.Series, previous_after_returns: pd.Series | None) -> float:
    if previous_after_returns is None:
        return np.nan
    names = current.index.union(previous_after_returns.index)
    return float(0.5 * (current.reindex(names, fill_value=0.0)
                        - previous_after_returns.reindex(names, fill_value=0.0)).abs().sum())


def _portfolio_monthly(predictions: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight forecast-decile spread, turnover, and simple cost scenarios.

    Each long and short sleeve carries one dollar of notional. Turnover is the
    one-way fraction traded in both sleeves after drifting last month's weights
    through realised returns, so it ranges from zero to two in ordinary cases.
    """
    rows = []
    for model, model_data in predictions.groupby("model", sort=True):
        previous_long = previous_short = previous_returns = None
        previous_month = None
        for month, group in model_data.groupby("yyyymm", sort=True):
            target = group.set_index("permno")["target_ret"].astype("float64")
            forecast = group.set_index("permno")["forecast"].astype("float64")
            rankable = len(group) >= 20 and forecast.nunique() > 1
            if not rankable:
                rows.append({"model": model, "yyyymm": int(month), "gross_return": np.nan,
                             "long_return": np.nan, "short_return": np.nan,
                             "long_turnover": np.nan, "short_turnover": np.nan,
                             "turnover": np.nan, "n_long": 0, "n_short": 0,
                             **{f"net_return_{cost}bps": np.nan for cost in COST_BPS}})
                previous_long = previous_short = previous_returns = previous_month = None
                continue

            ranks = forecast.rank(method="first", pct=True)
            long_names = ranks[ranks > 0.9].index
            short_names = ranks[ranks <= 0.1].index
            long_weights = pd.Series(1 / len(long_names), index=long_names, dtype="float64")
            short_weights = pd.Series(1 / len(short_names), index=short_names, dtype="float64")
            long_return = float(target.reindex(long_names).mean())
            short_return = float(target.reindex(short_names).mean())
            gross_return = long_return - short_return

            month_number = int(month)
            period = pd.Period(year=month_number // 100, month=month_number % 100, freq="M")
            adjacent = previous_month is not None and period == previous_month + 1
            drifted_long = _drift(previous_long, previous_returns) if adjacent else None
            drifted_short = _drift(previous_short, previous_returns) if adjacent else None
            long_turnover = _turnover(long_weights, drifted_long)
            short_turnover = _turnover(short_weights, drifted_short)
            turnover = long_turnover + short_turnover
            row = {
                "model": model, "yyyymm": int(month), "gross_return": gross_return,
                "long_return": long_return, "short_return": short_return,
                "long_turnover": long_turnover, "short_turnover": short_turnover,
                "turnover": turnover, "n_long": len(long_names), "n_short": len(short_names),
            }
            row.update({f"net_return_{cost}bps": gross_return - turnover * cost / 10_000
                        for cost in COST_BPS})
            rows.append(row)
            previous_long, previous_short = long_weights, short_weights
            previous_returns, previous_month = target, period
    return pd.DataFrame(rows)


def _portfolio_summary(monthly: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for model, group in monthly.groupby("model", sort=True):
        comparable = group.dropna(subset=["gross_return", "turnover"])
        gross = comparable["gross_return"].to_numpy(dtype="float64")
        gross_interval = block_bootstrap(gross, block=12, seed=seed)
        turnover = float(comparable["turnover"].mean())
        row = {
            "model": model,
            "mean_gross_return": gross_interval.estimate,
            "gross_return_ci_lo": gross_interval.lo,
            "gross_return_ci_hi": gross_interval.hi,
            "mean_turnover": turnover,
            "break_even_cost_bps": (gross_interval.estimate / turnover * 10_000
                                     if np.isfinite(turnover) and turnover > 0 and gross_interval.estimate > 0 else np.nan),
            "average_long_names": float(group["n_long"].replace(0, np.nan).mean()),
            "average_short_names": float(group["n_short"].replace(0, np.nan).mean()),
            "portfolio_months": int(len(comparable)),
        }
        for cost in COST_BPS:
            values = group[f"net_return_{cost}bps"].dropna().to_numpy(dtype="float64")
            interval = block_bootstrap(values, block=12, seed=seed)
            row[f"mean_net_return_{cost}bps"] = interval.estimate
            row[f"net_return_{cost}bps_ci_lo"] = interval.lo
            row[f"net_return_{cost}bps_ci_hi"] = interval.hi
        rows.append(row)
    return pd.DataFrame(rows)


def build_report(run_dir: Path, seed: int = 0) -> Path:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("task") != "returns":
        raise ValueError(f"expected a returns run, got {manifest.get('task')!r}")
    predictions = pd.read_parquet(
        run_dir / "predictions.parquet",
        columns=["model", "yyyymm", "permno", "forecast", "target_ret"],
    )
    monthly = _monthly(predictions)
    monthly_portfolios = _portfolio_monthly(predictions)
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
    summary = pd.DataFrame(summaries).merge(_portfolio_summary(monthly_portfolios, seed), on="model", how="left")

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
    monthly_portfolios.to_parquet(out / "monthly_portfolios.parquet", index=False)
    summary.to_json(out / "model_summary.json", orient="records", indent=2)
    comparison.to_json(out / "comparisons.json", orient="records", indent=2)
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": manifest["run_id"], "run_type": manifest["run_type"], "evidence": manifest["evidence"],
        "test_years_inspected": manifest["test_years"], "report_code_hash": code_hash(), "seed": seed,
        "primary_metric": "mean monthly Spearman rank information coefficient",
        "uncertainty": "95 percent moving-block bootstrap over months, 12-month blocks",
        "portfolio_method": (
            "equal-weight top and bottom forecast deciles; each sleeve has one dollar notional; one-way turnover "
            "is measured after drifting prior weights through realised returns; net scenarios charge 10, 25, or 50 "
            "basis points per dollar traded"
        ),
        "portfolio_note": "cost scenarios exclude market impact, capacity, borrow availability, and short-sale fees",
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
        "", "## Turnover and transaction-cost diagnostic", "",
        "| Model | Gross return | One-way turnover | Net at 10 bps | Net at 25 bps | Net at 50 bps | Break-even cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values("mean_gross_return", ascending=False, na_position="last").itertuples():
        lines.append(
            f"| {row.model} | {_number(row.mean_gross_return)} | {_number(row.mean_turnover)} | "
            f"{_number(row.mean_net_return_10bps)} | {_number(row.mean_net_return_25bps)} | "
            f"{_number(row.mean_net_return_50bps)} | {_number(row.break_even_cost_bps, 1)} bps |"
        )
    lines += [
        "", "## Interpretation guardrail", "",
        "The cost scenarios apply a fixed charge to measured turnover. They remain diagnostics rather than an "
        "investable-strategy claim because market impact, capacity, borrow availability, and short-sale fees are not modelled.", "",
    ]
    return "\n".join(lines)
