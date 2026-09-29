"""The plain-language REPORT.md for a saved report folder."""
import json
from pathlib import Path

from . import plots

NICE = plots.NICE


def fmt(x: float) -> str:
    return f"{x:+.4f}"


def summary_markdown(report_dir: Path) -> str:
    r = plots.load(report_dir)
    info = json.loads((report_dir / "report_info.json").read_text())
    comp, summ, sl, yrs = r["comparisons"], r["summary"], r["slices"], r["by_year"]
    lines = [f"# Report for run `{info['run_id']}`", "",
             f"- Run type: **{info['run_type']}**{'' if info['evidence'] else ' (QUICK: functional check, not evidence)'}",
             f"- Test years scored (now inspected): {info['test_years_inspected'][0]} to {info['test_years_inspected'][-1]}",
             f"- Main score: {info['primary_metric']}",
             f"- Weighting: {info['weighting']}",
             f"- Error bars: {info['interval']}",
             f"- Industry grouping checked against the official file: {info['industry_mapping_verified']}", "",
             "## Overall", "",
             "| Model | QLIKE (lower better) | Ranking skill (rank correlation) |", "|---|---|---|"]
    for _, s in summ.sort_values("qlike").iterrows():
        lines.append(f"| {NICE[s['model']]} | {s['qlike']:.4f} | {s['rank_corr']:.3f} |")
    lines += ["", "## Paired comparisons (model minus baseline; negative = model better)", "",
              "| Comparison | Average difference [95% interval] | Median-month view | Without top 0.1% answers (retrospective) | Months model better | Holm p |",
              "|---|---|---|---|---|---|"]
    for _, c in comp.iterrows():
        lines.append(f"| {NICE[c['model']]} vs {NICE[c['baseline']]} | {fmt(c['mean_estimate'])} [{fmt(c['mean_ci_lo'])}, {fmt(c['mean_ci_hi'])}] | "
                     f"{fmt(c['median_estimate'])} [{fmt(c['median_ci_lo'])}, {fmt(c['median_ci_hi'])}] | "
                     f"{fmt(c['trimmed_estimate'])} [{fmt(c['trimmed_ci_lo'])}, {fmt(c['trimmed_ci_hi'])}] | {c['months_model_better']:.0%} | {c['mean_p_holm']:.3f} |")
    lines += ["", "Error-bar sensitivity to block length (average difference):", ""]
    for _, c in comp.iterrows():
        ci = "; ".join(f"{b}m [{fmt(v[0])}, {fmt(v[1])}]" for b, v in ((k.replace('ci_block', ''), c[k]) for k in comp.columns if k.startswith("ci_block")))
        lines.append(f"- {NICE[c['model']]} vs {NICE[c['baseline']]}: {ci}")
    lines += ["", "## Year by year", ""]
    for (m, b), g in yrs.groupby(["model", "baseline"]):
        if b != "har" and not (m == "har" and b == "persistence"):
            continue
        better = int((g["ci_hi"] < 0).sum())
        worse = int((g["ci_lo"] > 0).sum())
        lines.append(f"- {NICE[m]} vs {NICE[b]}: clearly better in {better} of {len(g)} years, clearly worse in {worse}, unclear in {len(g) - better - worse}.")
    lines += ["", "## Where models win or lose (slices, exploratory)", "",
              "Many groups are checked at once, so roughly 1 in 20 'better' or 'worse' verdicts can appear by chance. Retrospective slices use the outcome and are diagnostics only.", ""]
    for (m, b), g in sl.groupby(["model", "baseline"]):
        counts = g[~g["retrospective"]]["verdict"].value_counts().to_dict()
        lines.append(f"### {NICE[m]} vs {NICE[b]}")
        lines.append(f"Verdicts across {int((~g['retrospective']).sum())} groups: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
        worst = g[(g["verdict"] == "worse") & ~g["retrospective"]].sort_values("diff", ascending=False).head(5)
        best = g[(g["verdict"] == "better") & ~g["retrospective"]].sort_values("diff").head(5)
        if len(best):
            lines.append("- Largest clear gains: " + "; ".join(f"{r.slice} = {r.value} ({fmt(r.diff)})" for r in best.itertuples()))
        if len(worst):
            lines.append("- Clear losses: " + "; ".join(f"{r.slice} = {r.value} ({fmt(r.diff)})" for r in worst.itertuples()))
        lines.append("")
    imp = r["importance"]
    lines += ["## What each model relies on (input groups)", ""]
    for m, g in imp[imp["kind"] == "group"].groupby("model"):
        lines.append(f"- {NICE[m]}: " + ", ".join(f"{x.input} {x.mean:+.4f}" for x in g.sort_values("mean", ascending=False).itertuples()))
    lines += ["", "Charts: `figures/` (open the .html files in a browser)."]
    return "\n".join(lines) + "\n"
