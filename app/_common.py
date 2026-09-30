"""Shared dashboard helpers. The dashboard only reads saved results and starts the same
commands you could type yourself (python -m p3_modellab ...)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from p3_modellab import config, explore, plots  # noqa: E402
from p3_modellab.tasks import volatility  # noqa: E402

NICE = plots.NICE
TASK_NAMES = {"volatility": "Volatility", "returns": "Returns", "delisting": "Adverse delisting"}

MODEL_CARDS = {
    "persistence": {"one": "Next month will look like the last month.",
                    "how": "Copies the last 22 trading days' volatility. Nothing is learned.",
                    "uses": "1 input", "good": "Simple and honest benchmark", "bad": "Blind to calm stocks that suddenly jump"},
    "har": {"one": "A fixed recipe: a bit of yesterday, last week and last month.",
            "how": "Learns 4 numbers (3 weights and a constant) from past data.",
            "uses": "3 inputs", "good": "The textbook volatility benchmark", "bad": "Ignores size, liquidity and market mood"},
    "ridge": {"one": "One weight per input, all added up.",
              "how": "A straight-line model on 13 inputs, held back from overreacting.",
              "uses": "13 inputs", "good": "Stable; copes with never-seen extremes", "bad": "Cannot learn if-then patterns"},
    "lightgbm": {"one": "Hundreds of small yes/no decision trees.",
                 "how": "Each new tree fixes the mistakes of the trees before it.",
                 "uses": "13 inputs", "good": "Finds if-then patterns; ranks stocks best", "bad": "Over-trusts patterns that break in a crisis"},
    "xgboost": {"one": "Boosted decision trees, grown level by level.",
                "how": "Same idea as LightGBM (each tree fixes earlier mistakes), different way of growing trees.",
                "uses": "13 inputs", "good": "A widely used, well-tested booster", "bad": "Slower; similar blind spots to LightGBM"},
    "catboost": {"one": "Boosted decision trees with balanced shapes.",
                 "how": "Every tree splits on the same question at each level, which tends to resist overfitting.",
                 "uses": "13 inputs", "good": "Stable with little tuning", "bad": "Slowest of the tree models"},
    "extratrees": {"one": "Many independent random trees, averaged (a random forest).",
                   "how": "Each tree is built with random splits; averaging smooths out their mistakes.",
                   "uses": "13 inputs", "good": "Robust, hard to overfit", "bad": "Cannot forecast beyond values it has seen"},
    "mlp": {"one": "A small neural network.",
            "how": "Two layers of simple units learn smooth patterns; training stops when the tuning score stops improving.",
            "uses": "13 inputs", "good": "Learns smooth curved patterns", "bad": "Sensitive to settings; needs lots of data"},
}

ESTIMATES = {
    "run_quick": ("about 30 seconds", False),
    "run_full": ("10 to 15 minutes", True),
    "report": ("2 to 3 minutes", True),
    "sweep_demo": ("about 10 seconds", False),
    "sweep_noise": ("about 1 minute", False),
    "sweep_data": ("10 to 15 minutes", True),
    "sweep_settings": ("45 to 60 minutes", True),
    "sweep_greedy": ("45 to 90 minutes", True),
}

_PROCS: dict[str, subprocess.Popen] = {}


def _read(p: Path) -> dict:
    return json.loads(p.read_text())


def runs() -> list[dict]:
    out = []
    if not config.RUNS_DIR.exists():
        return out
    for d in sorted(config.RUNS_DIR.iterdir(), reverse=True):
        if not (d / "manifest.json").exists():
            continue
        m = _read(d / "manifest.json")
        out.append({"path": d, "run_id": m["run_id"], "type": m["run_type"], "quick": m["quick"], "evidence": m["evidence"],
                    "test_years": f"{m['test_years'][0]} to {m['test_years'][-1]}", "forecasts": m["n_predictions"],
                    "has_report": (d / "report" / "report_info.json").exists(), "created": m["created_utc"]})
    return out


def report_catalog() -> list[dict]:
    """Find compact reports anywhere under results, including downloaded cloud runs."""
    out, seen = [], set()
    roots = [root for root in (config.RESULTS_DIR, config.DASHBOARD_DATA_DIR) if root.exists()]
    for manifest_path in (path for root in roots for path in root.rglob("manifest.json")):
        run_dir = manifest_path.parent
        report_dir = run_dir / "report"
        if not (report_dir / "model_summary.json").exists():
            continue
        try:
            manifest = _read(manifest_path)
        except (OSError, json.JSONDecodeError):
            continue
        task = manifest.get("task")
        if task not in TASK_NAMES or "run_id" not in manifest or manifest["run_id"] in seen:
            continue
        years = manifest.get("test_years") or []
        out.append({
            "path": run_dir, "report": report_dir, "run_id": manifest["run_id"], "task": task,
            "type": manifest.get("run_type", "unknown"), "quick": bool(manifest.get("quick")),
            "evidence": bool(manifest.get("evidence")), "created": manifest.get("created_utc", ""),
            "test_years": years, "forecasts": int(manifest.get("n_predictions", 0)),
            "models": [model.get("name") for model in manifest.get("spec", {}).get("models", [])],
        })
        seen.add(manifest["run_id"])
    return sorted(out, key=lambda row: (row["evidence"], len(row["models"]), row["created"]), reverse=True)


def catalog_label(run: dict) -> str:
    years = run["test_years"]
    span = f"{years[0]}-{years[-1]}" if years else "unknown years"
    status = "quick check" if run["quick"] else run["type"]
    return f"{TASK_NAMES[run['task']]} · {len(run['models'])} models · {span} · {status}"


def sweeps() -> list[dict]:
    root = config.RESULTS_DIR / "sweeps"
    out = []
    if not root.exists():
        return out
    for d in sorted(root.glob("2*"), reverse=True):
        if (d / "manifest.json").exists():
            m = _read(d / "manifest.json")
            out.append({"path": d, "kind": m["kind"], "size": m.get("size", "standard"), "trials": m["n_trials"], "created": m["created_utc"][:16]})
    return out


def mtime(p: Path) -> float:
    return max(f.stat().st_mtime for f in Path(p).rglob("*") if f.is_file())


def run_label(r: dict) -> str:
    kind = "QUICK" if r["quick"] else r["type"]
    return f"{r['created'][:16].replace('T', ' ')}  ({kind}, {r['test_years']})"


def current_run() -> dict | None:
    """The run chosen in the sidebar ('Results from'), if any run has a report."""
    reported = [r for r in runs() if r["has_report"]]
    if not reported:
        return None
    chosen = st.session_state.get("run_id")
    return next((r for r in reported if r["run_id"] == chosen), next((r for r in reported if r["evidence"]), reported[0]))


def need_run() -> dict:
    r = current_run()
    if r is None:
        st.info("No results yet. Start a run from the sidebar, then build its report.")
        st.stop()
    return r


def badge(r: dict) -> None:
    if not r["evidence"]:
        st.error("Quick run: a functional check on 20% of firms. Not evidence.", icon=":material/warning:")
    elif r["type"] == "exploratory":
        years = r["test_years"]
        span = f"{years[0]} to {years[-1]}" if isinstance(years, list) and years else years
        st.caption(f"Exploratory results, test years {span}. Compare with the locked study for the final holdout.")
    elif r["type"] == "locked":
        years = r["test_years"]
        span = f"{years[0]} to {years[-1]}" if isinstance(years, list) and years else years
        st.caption(f"Locked holdout results, test years {span}. The specification was frozen before these outcomes were inspected.")


@st.cache_data(show_spinner=False)
def report(path: str, stamp: float) -> dict:
    return plots.load(path)


def report_for(r: dict) -> dict:
    rep = r["path"] / "report"
    return report(str(rep), mtime(rep))


@st.cache_resource(show_spinner="Loading forecasts (first time only)...")
def _explore(run_path: str, stamp: float, panel_stamp: float) -> pd.DataFrame:
    from p3_modellab.report import first_listed_dates
    panel = pd.read_parquet(volatility.panel_paths()[0], columns=explore.PANEL_COLS)
    return explore.explore_table(Path(run_path), panel, first_listed_dates())


def explore_table(r: dict) -> pd.DataFrame:
    return _explore(str(r["path"]), (r["path"] / "predictions.parquet").stat().st_mtime, volatility.panel_paths()[0].stat().st_mtime)


@st.cache_resource(show_spinner=False)
def names() -> pd.DataFrame:
    from p3_modellab.data import raw
    return explore.names(raw.load_msenames())


def stats(items: list[tuple], accent: str | None = None) -> None:
    """Headline cards that wrap on narrow screens. items: (label, value[, small note[, tooltip]])."""
    cards = []
    for it in items:
        label, value = it[0], it[1]
        note = it[2] if len(it) > 2 and it[2] else ""
        tip = (it[3] if len(it) > 3 and it[3] else "").replace('"', "'")
        cards.append(
            f"<div class='p3card' title=\"{tip}\"><div class='p3lab'>{label}</div><div class='p3val'>{value}</div>"
            + (f"<div class='p3note'>{note}</div>" if note else "") + "</div>")
    color = accent or "#0072B2"
    st.markdown(
        "<style>.p3wrap{display:flex;flex-wrap:wrap;gap:12px;margin:4px 0 12px 0}"
        ".p3card{flex:1 1 170px;min-width:150px;padding:12px 16px;border-radius:12px;"
        f"border:1px solid rgba(128,128,128,0.25);border-left:4px solid {color};background:rgba(128,128,128,0.06)}}"
        ".p3lab{font-size:0.82em;opacity:0.75}.p3val{font-size:1.55em;font-weight:700;line-height:1.3}"
        ".p3note{font-size:0.8em;opacity:0.7}</style>"
        f"<div class='p3wrap'>{''.join(cards)}</div>", unsafe_allow_html=True)


def chart(fig, key: str | None = None) -> None:
    """Plotly chart, full width; scrolling the page must not zoom the chart.
    Each chart gets its own id in page order, so the same figure can appear twice on a page."""
    if key is None:
        n = st.session_state.get("_chart_n", 0) + 1
        st.session_state["_chart_n"] = n
        key = f"chart_{n}"
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": False, "displaylogo": False}, key=key)


# ---- background jobs -------------------------------------------------------------

def jobs_dir() -> Path:
    d = config.RESULTS_DIR / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def start_job(label: str, args: list[str]) -> dict:
    job_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    logs = config.RESULTS_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"job_{job_id}.log"
    with open(log, "w") as fh:
        p = subprocess.Popen([sys.executable, "-u", "-m", "p3_modellab", *args], cwd=PROJECT, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
    _PROCS[job_id] = p
    job = {"id": job_id, "label": label, "command": "python -m p3_modellab " + " ".join(args), "pid": p.pid, "log": str(log),
           "started": datetime.now().isoformat(timespec="seconds")}
    (jobs_dir() / f"{job_id}.json").write_text(json.dumps(job, indent=2))
    return job


def _alive(job: dict) -> bool:
    p = _PROCS.get(job["id"])
    if p is not None:
        return p.poll() is None
    try:
        os.kill(job["pid"], 0)
    except OSError:
        return False
    stat = subprocess.run(["ps", "-o", "stat=", "-p", str(job["pid"])], capture_output=True, text=True).stdout.strip()
    return bool(stat) and not stat.startswith("Z")


def job_status(job: dict) -> str:
    tail = Path(job["log"]).read_text()[-4000:] if Path(job["log"]).exists() else ""
    if _alive(job):
        return "running"
    if "Traceback" in tail:
        return "failed"
    if "saved to" in tail or "written to" in tail:
        return "done"
    return "stopped"


def recent_jobs(n: int = 4) -> list[dict]:
    return [_read(f) for f in sorted(jobs_dir().glob("*.json"), reverse=True)[:n]]


def confirm(key: str, where=st) -> bool:
    text, heavy = ESTIMATES[key]
    if heavy:
        where.caption(f"Takes {text} and uses 3 cores.")
        return where.checkbox(f"Yes, start ({text})", key=f"confirm_{key}")
    where.caption(f"Takes {text}.")
    return True


def theme_toggle() -> None:
    """Light / dark switch. Uses Streamlit's config at runtime (not an official API); falls back to a hint if that fails."""
    ctx_theme = getattr(getattr(st, "context", None), "theme", None)
    start_dark = getattr(ctx_theme, "type", "light") == "dark"
    dark = st.toggle("Dark mode", value=st.session_state.get("dark_mode", start_dark), key="dark_mode")
    want = "dark" if dark else "light"
    if st.session_state.get("_theme_applied") != want:
        st.session_state["_theme_applied"] = want
        try:
            from streamlit import config as st_config
            st_config.set_option("theme.base", want)
            st.rerun()
        except Exception:
            st.caption("Switch themes from the menu at the top right: Settings, Theme.")


def sidebar_controls() -> None:
    """Shown on every page: which results to view, and buttons to run models or build a report."""
    with st.sidebar:
        theme_toggle()
        reported = [r for r in runs() if r["has_report"]]
        if reported:
            cur = current_run()
            st.selectbox("Results from", [r["run_id"] for r in reported], index=[r["run_id"] for r in reported].index(cur["run_id"]),
                         format_func=lambda rid: run_label(next(r for r in reported if r["run_id"] == rid)), key="run_id",
                         help="Which saved run the pages show.")
        st.markdown("#### Run models")
        kind = st.segmented_control("Kind", ["Quick", "Full"], default="Quick", key="run_kind", label_visibility="collapsed",
                                    help="Quick: 20% of firms, retrain every 3 years, a functional check only. Full: all firms, every year 2000 to 2021.")
        full = kind == "Full"
        if confirm("run_full" if full else "run_quick") and st.button("Start run", width="stretch"):
            job = start_job("full run" if full else "quick run", ["run"] if full else ["run", "--quick"])
            st.toast(f"Started: {job['command']}")
        todo = [r for r in runs() if not r["has_report"]]
        if todo:
            st.markdown("#### Build a report")
            pick = st.selectbox("Run", [r["run_id"] for r in todo], format_func=lambda rid: run_label(next(r for r in todo if r["run_id"] == rid)),
                                label_visibility="collapsed")
            if confirm("report") and st.button("Build report", width="stretch"):
                job = start_job("report", ["report", "--run", str(config.RUNS_DIR / pick)])
                st.toast(f"Started: {job['command']}")
        jobs = recent_jobs()
        if jobs:
            st.markdown("#### Jobs")
            for j in jobs:
                s = job_status(j)
                icon = {"running": ":material/progress_activity:", "done": ":material/check_circle:", "failed": ":material/error:",
                        "stopped": ":material/stop_circle:"}[s]
                st.markdown(f"{icon} {j['label']} &middot; {s}", help=f"{j['command']}\nstarted {j['started']}")
            if st.button("Refresh", width="stretch"):
                st.rerun()
