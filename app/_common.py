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
    "persistence": {"one": "The market's copycat: next month repeats the last one.",
                    "how": "Copies the last 22 trading days' volatility. Nothing is learned.",
                    "uses": "1 input", "good": "Simple and honest benchmark", "bad": "Blind to calm stocks that suddenly jump"},
    "har": {"one": "The time-horizon blender: yesterday, last week and last month in one forecast.",
            "how": "Learns 4 numbers (3 weights and a constant) from past data.",
            "uses": "3 inputs", "good": "The textbook volatility benchmark", "bad": "Ignores size, liquidity and market mood"},
    "ridge": {"one": "The disciplined scorecard: one restrained weight for every signal.",
              "how": "A straight-line model on 13 inputs, held back from overreacting.",
              "uses": "13 inputs", "good": "Stable; copes with never-seen extremes", "bad": "Cannot learn if-then patterns"},
    "lightgbm": {"one": "The speedy tree relay: each tiny tree fixes the last one's mistakes.",
                 "how": "Each new tree fixes the mistakes of the trees before it.",
                 "uses": "13 inputs", "good": "Finds if-then patterns; ranks stocks best", "bad": "Over-trusts patterns that break in a crisis"},
    "xgboost": {"one": "The careful tree team: every round studies what the earlier rounds missed.",
                "how": "Same idea as LightGBM (each tree fixes earlier mistakes), different way of growing trees.",
                "uses": "13 inputs", "good": "A widely used, well-tested booster", "bad": "Slower; similar blind spots to LightGBM"},
    "catboost": {"one": "The balanced-tree specialist: steady shapes with little fussing over settings.",
                 "how": "Every tree splits on the same question at each level, which tends to resist overfitting.",
                 "uses": "13 inputs", "good": "Stable with little tuning", "bad": "Slowest of the tree models"},
    "extratrees": {"one": "The crowd vote: many randomized trees average away one another's noise.",
                   "how": "Each tree is built with random splits; averaging smooths out their mistakes.",
                   "uses": "13 inputs", "good": "Robust, hard to overfit", "bad": "Cannot forecast beyond values it has seen"},
    "mlp": {"one": "The compact pattern hunter: two neural layers chase smooth nonlinear clues.",
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


def show_development_runs() -> bool:
    """Quick runs and demo sweeps stay local unless explicitly requested."""
    return os.environ.get("P3_SHOW_DEVELOPMENT_RUNS", "").strip().lower() in {"1", "true", "yes"}


def has_training_panel() -> bool:
    """Whether this checkout can fit models instead of only reading saved reports."""
    panel, _, meta = volatility.panel_paths()
    return panel.exists() and meta.exists()


def has_detailed_pages() -> bool:
    """Detailed diagnostics need row-level forecasts and the local feature panel."""
    return has_training_panel() and bool(detailed_volatility_runs())


def has_sweep_pages() -> bool:
    """Expose the local robustness workspace when any sweep artifact exists."""
    root = config.RESULTS_DIR / "sweeps"
    return root.exists() and any(root.glob("*/manifest.json"))


def _read(p: Path) -> dict:
    return json.loads(p.read_text())


def runs(task: str | None = None) -> list[dict]:
    """Return local runs, optionally limited to one task.

    The detailed Results, Misses, and Time Slider pages consume volatility-only
    report artifacts. Cross-task reports belong in ``report_catalog()`` and the
    Model comparison page.
    """
    out = []
    if not config.RUNS_DIR.exists():
        return out
    for d in sorted(config.RUNS_DIR.iterdir(), reverse=True):
        if not (d / "manifest.json").exists():
            continue
        m = _read(d / "manifest.json")
        run_task = m.get("task", "volatility")
        if task is not None and run_task != task:
            continue
        out.append({"path": d, "run_id": m["run_id"], "type": m["run_type"], "quick": m["quick"], "evidence": m["evidence"],
                    "test_years": f"{m['test_years'][0]} to {m['test_years'][-1]}", "forecasts": m["n_predictions"],
                    "has_report": (d / "report" / "report_info.json").exists(), "created": m["created_utc"],
                    "task": run_task})
    return out


_VOLATILITY_DETAIL_FILES = {
    "report_info.json", "model_summary.json", "comparisons.json", "by_year.parquet",
    "slices.parquet", "worst_cases.parquet", "calibration.parquet", "importance.parquet",
    "monthly_losses.parquet",
}


def detailed_volatility_runs() -> list[dict]:
    """Find complete local volatility reports, including downloaded Kaggle runs."""
    out, seen = [], set()
    if not config.RESULTS_DIR.exists():
        return out
    for manifest_path in config.RESULTS_DIR.rglob("manifest.json"):
        run_dir = manifest_path.parent
        report_dir = run_dir / "report"
        if not (run_dir / "predictions.parquet").exists():
            continue
        if not all((report_dir / name).exists() for name in _VOLATILITY_DETAIL_FILES):
            continue
        try:
            manifest = _read(manifest_path)
            summary = _read(report_dir / "model_summary.json")
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("task", "volatility") != "volatility" or manifest.get("run_id") in seen:
            continue
        row = {
            "path": run_dir, "run_id": manifest["run_id"], "type": manifest["run_type"],
            "quick": manifest["quick"], "evidence": manifest["evidence"],
            "test_years": f"{manifest['test_years'][0]} to {manifest['test_years'][-1]}",
            "forecasts": manifest["n_predictions"], "has_report": True,
            "created": manifest["created_utc"], "task": "volatility", "model_count": len(summary),
        }
        out.append(row)
        seen.add(manifest["run_id"])
    if not show_development_runs():
        out = [row for row in out if row["evidence"] and not row["quick"]]
        if out:
            complete_count = max(row["model_count"] for row in out)
            out = [row for row in out if row["model_count"] == complete_count]
    return sorted(out, key=lambda row: (row["evidence"], row["model_count"], row["created"]), reverse=True)


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
            "label_definition": manifest.get("spec", {}).get("label_definition", "primary"),
        })
        seen.add(manifest["run_id"])
    if not show_development_runs():
        out = [row for row in out if row["evidence"] and not row["quick"]]
        maximums: dict[tuple, int] = {}
        for row in out:
            if row["task"] != "volatility":
                continue
            key = (row["type"], tuple(row["test_years"]), row["label_definition"])
            maximums[key] = max(maximums.get(key, 0), len(row["models"]))
        out = [
            row for row in out
            if row["task"] != "volatility"
            or len(row["models"]) == maximums[(row["type"], tuple(row["test_years"]), row["label_definition"])]
        ]
    return sorted(out, key=lambda row: (row["evidence"], len(row["models"]), row["created"]), reverse=True)


def catalog_label(run: dict) -> str:
    years = run["test_years"]
    span = f"{years[0]}-{years[-1]}" if years else "unknown years"
    status = "quick check" if run["quick"] else run["type"]
    label = run.get("label_definition", "primary").replace("_", " ")
    label_text = f" · {label} label" if run["task"] == "delisting" and label != "primary" else ""
    return f"{TASK_NAMES[run['task']]} · {len(run['models'])} models · {span} · {status}{label_text}"


def sweeps() -> list[dict]:
    root = config.RESULTS_DIR / "sweeps"
    out = []
    if not root.exists():
        return out
    for d in sorted(root.glob("2*"), reverse=True):
        if (d / "manifest.json").exists():
            m = _read(d / "manifest.json")
            size = m.get("size", "standard") or "standard"
            if size == "demo" and not show_development_runs():
                continue
            out.append({"path": d, "kind": m["kind"], "size": size, "trials": m["n_trials"], "created": m["created_utc"][:16]})
    return out


def mtime(p: Path) -> float:
    return max(f.stat().st_mtime for f in Path(p).rglob("*") if f.is_file())


def run_label(r: dict) -> str:
    kind = "QUICK" if r["quick"] else r["type"]
    count = f", {r['model_count']} models" if r.get("model_count") else ""
    return f"{r['created'][:16].replace('T', ' ')}  ({kind}, {r['test_years']}{count})"


def current_run() -> dict | None:
    """The volatility run chosen for the detailed pages, if one has a report."""
    reported = detailed_volatility_runs()
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
    """Render the light/dark control at the top-right of the main page."""
    ctx_theme = getattr(getattr(st, "context", None), "theme", None)
    start_dark = getattr(ctx_theme, "type", "light") == "dark"
    _, control = st.columns([9, 1])
    with control:
        dark = st.toggle("Dark theme", value=st.session_state.get("dark_mode", start_dark), key="dark_mode")
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
    """Shown on every page: detailed volatility run controls and background jobs."""
    with st.sidebar:
        reported = detailed_volatility_runs()
        local_panel = has_training_panel()
        if reported and local_panel:
            cur = current_run()
            run_ids = [r["run_id"] for r in reported]
            if st.session_state.get("run_id") not in run_ids:
                st.session_state["run_id"] = cur["run_id"]
            st.selectbox("Volatility study", run_ids, index=run_ids.index(cur["run_id"]),
                         format_func=lambda rid: run_label(next(r for r in reported if r["run_id"] == rid)), key="run_id",
                         help="Saved eight-model volatility study used by Volatility Analysis, Error Analysis, and Performance Over Time.")
        if not local_panel:
            st.caption(
                "Public view: charts use the saved aggregate results. "
                "Model fitting and detailed forecast analysis run from the repository workflow."
            )
            return
        st.markdown("#### Run models")
        kind = st.segmented_control("Kind", ["Quick", "Full"], default="Quick", key="run_kind", label_visibility="collapsed",
                                    help="Quick: 20% of firms, retrain every 3 years, a functional check only. Full: all firms, every year 2000 to 2021.")
        full = kind == "Full"
        if confirm("run_full" if full else "run_quick") and st.button("Start run", width="stretch"):
            job = start_job("full run" if full else "quick run", ["run"] if full else ["run", "--quick"])
            st.toast(f"Started: {job['command']}")
        todo = [r for r in runs("volatility") if not r["has_report"]]
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
