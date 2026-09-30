"""Evaluation and report artifacts for Stage 9 adverse-delisting runs."""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from .delisting_models import PROBABILITY_FLOOR
from .manifest import code_hash
from .uncertainty import block_bootstrap


PRIMARY_BLOCK = 12
BUDGETS = (0.005, 0.01, 0.05)


def _clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype="float64"), PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)


def _monthly(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, month), group in predictions.groupby(["model", "yyyymm"], sort=True):
        y = group["target_adverse_delisting"].to_numpy(dtype="int8")
        p = _clip(group["forecast"])
        pr_auc = float(average_precision_score(y, p)) if np.unique(y).size == 2 else np.nan
        rows.append({
            "model": model, "yyyymm": int(month),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "pr_auc": pr_auc, "event_rate": float(y.mean()),
            "mean_forecast": float(p.mean()), "n": int(len(group)), "positive_rows": int(y.sum()),
        })
    return pd.DataFrame(rows)


def _calibration(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model", sort=True):
        ranked = group["forecast"].rank(method="first", pct=True)
        bucket = np.minimum((ranked * bins).astype(int), bins - 1)
        work = group.assign(calibration_bin=bucket)
        for b, cell in work.groupby("calibration_bin", sort=True):
            rows.append({
                "model": model, "bin": int(b), "n": int(len(cell)),
                "mean_forecast": float(cell["forecast"].mean()),
                "observed_rate": float(cell["target_adverse_delisting"].mean()),
            })
    return pd.DataFrame(rows)


def _month_ordinal(yyyymm: pd.Series) -> np.ndarray:
    values = yyyymm.to_numpy(dtype="int64")
    return (values // 100) * 12 + (values % 100)


def _budget_metrics(predictions: pd.DataFrame, budgets: tuple[float, ...] = BUDGETS) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model", sort=True):
        positives = group[group["target_adverse_delisting"] == 1]
        event_cols = ["permno", "next_adverse_date"]
        events = positives[event_cols].drop_duplicates()
        total_events = len(events)
        ranking_available = bool((group.groupby("yyyymm")["forecast"].nunique() > 1).any())
        for budget in budgets:
            if not ranking_available:
                rows.append({
                    "model": model, "budget": budget, "ranking_available": False,
                    "alerted_rows": 0, "positive_alert_rows": 0, "row_precision": np.nan,
                    "unique_events_hit": 0, "total_unique_events": int(total_events),
                    "event_recall": np.nan, "events_per_1000_alerts": np.nan,
                    "median_first_alert_lead_months": np.nan,
                })
                continue
            ordered = group.sort_values(["yyyymm", "forecast", "row_id"], ascending=[True, False, True]).copy()
            ordered["rank"] = ordered.groupby("yyyymm").cumcount() + 1
            ordered["review_slots"] = np.ceil(ordered.groupby("yyyymm")["row_id"].transform("size") * budget).astype(int)
            alerts = ordered[ordered["rank"] <= ordered["review_slots"]]
            useful = alerts[alerts["target_adverse_delisting"] == 1].copy()
            hit_events = useful[event_cols].drop_duplicates()

            lead = np.array([], dtype="float64")
            if len(useful):
                useful["prediction_ordinal"] = _month_ordinal(useful["yyyymm"])
                useful["event_ordinal"] = (useful["next_adverse_date"].dt.year * 12
                                           + useful["next_adverse_date"].dt.month).to_numpy()
                first = useful.groupby(event_cols, dropna=False)["prediction_ordinal"].min().reset_index()
                first["event_ordinal"] = first["next_adverse_date"].dt.year * 12 + first["next_adverse_date"].dt.month
                lead = (first["event_ordinal"] - first["prediction_ordinal"]).to_numpy(dtype="float64")
            rows.append({
                "model": model, "budget": budget, "ranking_available": True, "alerted_rows": int(len(alerts)),
                "positive_alert_rows": int(len(useful)), "row_precision": float(len(useful) / len(alerts)),
                "unique_events_hit": int(len(hit_events)), "total_unique_events": int(total_events),
                "event_recall": float(len(hit_events) / total_events) if total_events else np.nan,
                "events_per_1000_alerts": float(1000 * len(hit_events) / len(alerts)),
                "median_first_alert_lead_months": float(np.median(lead)) if lead.size else np.nan,
            })
    return pd.DataFrame(rows)


def _pairwise(monthly: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    models = sorted(monthly["model"].unique())
    wide_log = monthly.pivot(index="yyyymm", columns="model", values="log_loss")
    wide_brier = monthly.pivot(index="yyyymm", columns="model", values="brier")
    rows = []
    for model, baseline in itertools.combinations(models, 2):
        common = wide_log[[model, baseline]].dropna()
        diff = common[model] - common[baseline]
        interval = block_bootstrap(diff.to_numpy(), block=PRIMARY_BLOCK, seed=seed)
        brier = wide_brier[[model, baseline]].dropna()
        rows.append({
            "model": model, "baseline": baseline, "metric": "log_loss",
            **interval.as_dict(), "months_model_better": float((diff < 0).mean()),
            "brier_difference": float((brier[model] - brier[baseline]).mean()),
        })
    return pd.DataFrame(rows)


def _summary(predictions: pd.DataFrame, monthly: pd.DataFrame, calibration: pd.DataFrame,
             budgets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model", sort=True):
        y = group["target_adverse_delisting"].to_numpy(dtype="int8")
        p = _clip(group["forecast"])
        months = monthly[monthly["model"] == model]
        log_interval = block_bootstrap(months["log_loss"].to_numpy(), block=PRIMARY_BLOCK)
        cal = calibration[calibration["model"] == model]
        ece = float(np.average(np.abs(cal["mean_forecast"] - cal["observed_rate"]), weights=cal["n"]))
        row = {
            "model": model, "mean_monthly_log_loss": log_interval.estimate,
            "log_loss_ci_lo": log_interval.lo, "log_loss_ci_hi": log_interval.hi,
            "overall_log_loss": float(log_loss(y, p, labels=[0, 1])),
            "mean_monthly_brier": float(months["brier"].mean()),
            "overall_pr_auc": float(average_precision_score(y, p)),
            "mean_monthly_pr_auc": float(months["pr_auc"].mean()),
            "event_rate": float(y.mean()), "mean_forecast": float(p.mean()),
            "expected_calibration_error": ece, "n_examples": int(len(group)),
            "n_months": int(group["yyyymm"].nunique()),
        }
        for budget in BUDGETS:
            metric = budgets[(budgets["model"] == model) & np.isclose(budgets["budget"], budget)].iloc[0]
            tag = f"{budget:.3f}".rstrip("0").rstrip(".").replace(".", "p")
            row[f"event_recall_at_{tag}"] = float(metric["event_recall"])
            row[f"row_precision_at_{tag}"] = float(metric["row_precision"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_report(run_dir: Path, seed: int = 0) -> Path:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("task") != "delisting":
        raise ValueError(f"expected a delisting run, got {manifest.get('task')!r}")
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    monthly = _monthly(predictions)
    calibration = _calibration(predictions)
    budgets = _budget_metrics(predictions)
    summary = _summary(predictions, monthly, calibration, budgets)
    pairwise = _pairwise(monthly, seed=seed)
    base_rows = []
    for row in pairwise[(pairwise["model"] == "base_rate") | (pairwise["baseline"] == "base_rate")].to_dict("records"):
        if row["baseline"] == "base_rate":
            base_rows.append(row)
            continue
        flipped = dict(row)
        flipped["model"], flipped["baseline"] = row["baseline"], "base_rate"
        flipped["estimate"], flipped["ci_lo"], flipped["ci_hi"] = -row["estimate"], -row["ci_hi"], -row["ci_lo"]
        flipped["months_model_better"] = 1.0 - row["months_model_better"]
        flipped["brier_difference"] = -row["brier_difference"]
        base_rows.append(flipped)
    comparisons = pd.DataFrame(base_rows)

    out = run_dir / "report"
    out.mkdir(exist_ok=True)
    monthly.to_parquet(out / "monthly_metrics.parquet", index=False)
    calibration.to_parquet(out / "calibration.parquet", index=False)
    budgets.to_parquet(out / "budget_metrics.parquet", index=False)
    summary.to_json(out / "model_summary.json", orient="records", indent=2)
    comparisons.to_json(out / "comparisons.json", orient="records", indent=2)
    pairwise.to_json(out / "pairwise_comparisons.json", orient="records", indent=2)
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": manifest["run_id"], "run_type": manifest["run_type"], "evidence": manifest["evidence"],
        "test_years_inspected": manifest["test_years"], "embargo_years": manifest.get("embargo_years", []),
        "report_code_hash": code_hash(), "seed": seed,
        "primary_metric": "mean monthly log loss (lower is better)",
        "secondary_metrics": ["Brier score", "PR-AUC", "calibration error"],
        "operational_metrics": "unique-event recall and row precision at fixed monthly review budgets",
        "uncertainty": f"95 percent moving-block bootstrap over months, {PRIMARY_BLOCK}-month blocks",
    }
    (out / "report_info.json").write_text(json.dumps(info, indent=2))
    (out / "REPORT.md").write_text(summary_markdown(info, summary, comparisons, budgets))
    return out


def _number(value: float, digits: int = 4) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def _percent(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.1%}"


def summary_markdown(info: dict, summary: pd.DataFrame, comparison: pd.DataFrame,
                     budgets: pd.DataFrame) -> str:
    lines = [
        f"# Adverse-delisting report for `{info['run_id']}`", "",
        f"- Run type: **{info['run_type']}**{'' if info['evidence'] else ' (quick functional check, not evidence)'}",
        f"- Years inspected: {info['test_years_inspected'][0]} to {info['test_years_inspected'][-1]}",
        f"- Primary metric: {info['primary_metric']}",
        f"- Uncertainty: {info['uncertainty']}",
        f"- Embargo years omitted because their outcome horizon overlaps sealed data: {info['embargo_years']}", "",
        "## Probability quality", "",
        "| Model | Mean monthly log loss | 95% interval | PR-AUC | Brier | Calibration error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values("mean_monthly_log_loss").itertuples():
        lines.append(
            f"| {row.model} | {_number(row.mean_monthly_log_loss)} | "
            f"[{_number(row.log_loss_ci_lo)}, {_number(row.log_loss_ci_hi)}] | "
            f"{_number(row.overall_pr_auc)} | {_number(row.mean_monthly_brier)} | "
            f"{_number(row.expected_calibration_error)} |"
        )
    lines += ["", "## Paired log-loss differences against the historical base rate", ""]
    for row in comparison.itertuples():
        lines.append(f"- {row.model}: {_number(row.estimate)} [{_number(row.ci_lo)}, {_number(row.ci_hi)}]; "
                     f"negative favours {row.model}.")
    lines += ["", "## Fixed monthly review budgets", "",
              "| Model | Review budget | Event recall | Alert-row precision | Events per 1,000 alerts | Median lead (months) |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in budgets.itertuples():
        lines.append(
            f"| {row.model} | {row.budget:.1%} | {_percent(row.event_recall)} | {_percent(row.row_precision)} | "
            f"{_number(row.events_per_1000_alerts, 1)} | {_number(row.median_first_alert_lead_months, 1)} |"
        )
    lines += [
        "", "## Interpretation", "",
        "A stock can contribute several positive monthly rows for one later delisting. Probability metrics therefore "
        "describe twelve-month firm-month risk, while event recall counts each adverse delisting once. Review-budget "
        "metrics simulate inspecting only the highest-risk stocks each month.", "",
    ]
    return "\n".join(lines)
