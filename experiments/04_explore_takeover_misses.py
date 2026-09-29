"""Exploratory: are LightGBM's worst misses against HAR concentrated in takeover targets?

For each test year, take the month where LightGBM lost most to HAR (QLIKE),
take the 50 stocks with the largest loss that month, and compare how often they
left the exchange through a merger (CRSP codes 200-299) within the next 12 months
versus all other stocks that month.

RETROSPECTIVE: uses what happened after the forecast, so it explains misses but
cannot be used to forecast. Writes results/explore/takeover_misses.csv.

Usage: python experiments/04_explore_takeover_misses.py [run_folder]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from p3_modellab import config  # noqa: E402
from p3_modellab.data import raw  # noqa: E402
from p3_modellab.losses import qlike  # noqa: E402

TOP = 50

if __name__ == "__main__":
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else max(config.RUNS_DIR.glob("*-volatility-exploratory"), key=lambda p: p.stat().st_mtime)
    p = pd.read_parquet(run / "predictions.parquet", columns=["row_id", "permno", "yyyymm", "model", "forecast", "target_rv"])
    p = p[p["model"].isin(["lightgbm", "har"])]
    p["q"] = qlike(p["target_rv"].to_numpy(), p["forecast"].to_numpy())
    w = p.pivot(index="row_id", columns="model", values="q").join(p.drop_duplicates("row_id").set_index("row_id")[["permno", "yyyymm"]])
    w["gap"] = w["lightgbm"] - w["har"]
    dl = raw.load_msedelist()
    dl = dl[dl["dlstcd"] // 100 == 2]
    rows = []
    for year, g in w.groupby(w["yyyymm"] // 100):
        monthly = g.groupby("yyyymm")["gap"].agg(["mean", "median"])
        worst_month = int(monthly["mean"].idxmax())
        m = g[g["yyyymm"] == worst_month]
        start = pd.Period(year=worst_month // 100, month=worst_month % 100, freq="M").end_time.normalize() + pd.Timedelta(days=1)
        merged = set(dl[(dl["dlstdt"] >= start) & (dl["dlstdt"] < start + pd.DateOffset(months=12))]["permno"])
        top = m.nlargest(TOP, "gap")
        rest = m.drop(top.index)
        rows.append({
            "year": int(year), "worst_month": worst_month,
            "month_mean_gap": round(float(monthly.loc[worst_month, "mean"]), 4),
            "month_median_gap": round(float(monthly.loc[worst_month, "median"]), 4),
            # only meaningful when LightGBM lost to HAR overall that month
            "top5_share_of_month_gap": round(float(m.nlargest(5, "gap")["gap"].sum() / m["gap"].sum()), 3) if m["gap"].sum() > 0 else None,
            f"top{TOP}_merged_within_12m": round(float(top["permno"].isin(merged).mean()), 3),
            "others_merged_within_12m": round(float(rest["permno"].isin(merged).mean()), 4),
        })
    out = pd.DataFrame(rows)
    out["ratio"] = (out[f"top{TOP}_merged_within_12m"] / out["others_merged_within_12m"]).round(1)
    dest = config.RESULTS_DIR / "explore"
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / "takeover_misses.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nyears where LightGBM's top-{TOP} misses were at least 3x more likely to be takeover targets: "
          f"{int((out['ratio'] >= 3).sum())} of {len(out)}")
