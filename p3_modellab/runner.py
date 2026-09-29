"""Walk-forward runs: train every model for each test period, save forecasts and a full record.

Run folder results/runs/<run_id>/:
- manifest.json: data fingerprints, code hash, settings, model specs, fold membership
  hashes, every tuning trial, timings, library versions, run type;
- predictions.parquet: one row per (example, model) with the four timestamps, answer and forecast;
- models/: fitted models (joblib), needed later for importance and re-scoring.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict
from functools import partial
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .contract import TIME_COLS, validate_panel
from .data.audit import load_audit
from .manifest import code_hash, environment
from .models import MODEL_NAMES, make_model, spec
from .splits import walk_forward
from .tasks import volatility

say = partial(print, flush=True)

QUICK_FIRM_SHARE = 20  # percent of firms kept in quick mode
QUICK_REFIT_EVERY = 3
VALIDATION_MONTHS = 24


class RunError(RuntimeError):
    pass


def _canonical(obj) -> object:
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(_canonical(obj), sort_keys=True).encode()).hexdigest()[:16]


def reserved_start(audit: dict) -> pd.Timestamp:
    """The sealed period is the last three calendar years that have forecasts."""
    last = pd.Period(audit["cutoffs"]["one_month_ahead_last_prediction_month"], freq="M")
    return pd.Timestamp(year=last.year - 2, month=1, day=1)


def default_test_years(audit: dict, run_type: str) -> list[int]:
    rs = reserved_start(audit).year
    last = pd.Period(audit["cutoffs"]["one_month_ahead_last_prediction_month"], freq="M").year
    return list(range(2000, rs)) if run_type == "exploratory" else list(range(rs, last + 1))


def quick_sample(permno: pd.Series) -> pd.Series:
    """Deterministic firm sample: the same firms every time."""
    return (permno.astype("int64") * 2654435761) % 100 < QUICK_FIRM_SHARE


def run_spec(models: list[str], seed: int, quick: bool, cfg: volatility.VolConfig) -> dict:
    return _canonical({
        "task": volatility.TASK,
        "panel_config": asdict(cfg),
        "features": volatility.FEATURES,
        "models": [spec(m, volatility.FEATURES, seed, quick) for m in models],
        "seed": seed,
        "validation_months": VALIDATION_MONTHS,
        "refit_every": QUICK_REFIT_EVERY if quick else 1,
        "firm_sample_percent": QUICK_FIRM_SHARE if quick else 100,
    })


def lock_path(task: str) -> Path:
    return config.LOCKS_DIR / f"{task}.json"


def create_lock(models: list[str], seed: int = 0, cfg: volatility.VolConfig = volatility.VolConfig(), note: str = "") -> Path:
    """Freeze the full specification before the sealed period is ever evaluated."""
    p = lock_path(volatility.TASK)
    if p.exists():
        raise RunError(f"{p} already exists; a lock is written once. Move it aside deliberately if the design has changed.")
    s = run_spec(models, seed, False, cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": note,
                             "spec": s, "spec_hash": _hash(s), "code_hash": code_hash()}, indent=2))
    return p


def _check_lock(s: dict, allow_rerun: bool) -> dict:
    p = lock_path(volatility.TASK)
    if not p.exists():
        raise RunError("a locked run needs a lock file; create one first (python -m p3_modellab lock)")
    lock = json.loads(p.read_text())
    if lock["spec_hash"] != _hash(s):
        raise RunError("the requested run does not match the locked specification")
    previous = [d for d in config.RUNS_DIR.glob("*-locked") if (d / "manifest.json").exists()
                and json.loads((d / "manifest.json").read_text()).get("lock", {}).get("spec_hash") == lock["spec_hash"]]
    if previous and not allow_rerun:
        raise RunError(f"the locked evaluation already ran ({previous[0].name}); rerunning needs allow_rerun and is recorded")
    return {"spec_hash": lock["spec_hash"], "lock_created_utc": lock["created_utc"], "lock_code_hash": lock["code_hash"],
            "code_changed_since_lock": lock["code_hash"] != code_hash(), "previous_locked_runs": [d.name for d in previous]}


def run(
    models: list[str] = list(MODEL_NAMES),
    run_type: str = "exploratory",
    test_years: list[int] | None = None,
    quick: bool = False,
    seed: int = 0,
    allow_rerun: bool = False,
    panel: pd.DataFrame | None = None,
    audit: dict | None = None,
    out_root: Path | None = None,
    progress=say,
) -> Path:
    started_code_hash = code_hash()
    if run_type == "locked" and quick:
        raise RunError("quick mode is for functional checks and cannot be a locked evaluation")
    unknown = [m for m in models if m not in MODEL_NAMES]
    if unknown:
        raise RunError(f"unknown models {unknown}")
    cfg = volatility.VolConfig()
    audit = audit or load_audit()
    if panel is None:
        panel, panel_meta = volatility.load_or_build(cfg, progress=progress)
        panel_info = {"source": "results/panels", "meta_created_utc": panel_meta["created_utc"],
                      "code_hash": panel_meta["code_hash"], "summary": panel_meta["summary"]}
    else:
        panel_info = {"source": "supplied by caller", "rows": int(len(panel))}
    validate_panel(panel)

    s = run_spec(models, seed, quick, cfg)
    lock_info = _check_lock(s, allow_rerun) if run_type == "locked" else None
    years = test_years or default_test_years(audit, run_type)
    rs = reserved_start(audit)
    work = panel[quick_sample(panel["permno"])].reset_index(drop=True) if quick else panel
    folds = walk_forward(work, years, rs, run_type, validation_months=VALIDATION_MONTHS,
                         refit_every=QUICK_REFIT_EVERY if quick else 1)

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}-{volatility.TASK}-{run_type}{'-quick' if quick else ''}"
    out = Path(out_root or config.RUNS_DIR) / run_id
    (out / "models").mkdir(parents=True, exist_ok=False)
    by_id = work.set_index("row_id")
    preds, fold_records = [], []
    t_run = time.time()
    for f in folds:
        train, val, test = by_id.loc[f.train_ids].reset_index(), by_id.loc[f.val_ids].reset_index(), by_id.loc[f.test_ids].reset_index()
        rec = {**f.summary(), "models": {}}
        for name in models:
            t0 = time.time()
            model = make_model(name, volatility.FEATURES, seed, quick)
            tuning = model.fit(train, val)
            h = model.predict(test)
            if not np.all(np.isfinite(h)) or np.any(h <= 0):
                raise RunError(f"{name} produced non-positive or non-finite forecasts in fold {f.fold}")
            joblib.dump(model, out / "models" / f"{name}_fold{f.fold:02d}.joblib")
            rec["models"][name] = {**tuning, "fit_predict_seconds": round(time.time() - t0, 2)}
            preds.append(pd.DataFrame({"row_id": test["row_id"].to_numpy(), "fold": f.fold, "model": name, "forecast": h}))
            progress(f"fold {f.fold} {f.test_years}: {name} done in {time.time() - t0:.1f}s")
        fold_records.append(rec)

    keep = ["row_id", "permno", "yyyymm", *TIME_COLS, "target_rv"]
    predictions = pd.concat(preds, ignore_index=True).merge(work[keep], on="row_id", how="left", validate="many_to_one")
    predictions.to_parquet(out / "predictions.parquet", index=False)
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_type": run_type,
        "quick": quick,
        "evidence": not quick,
        "task": volatility.TASK,
        "test_years": years,
        "reserved_start": str(rs.date()),
        "spec": s,
        "spec_hash": _hash(s),
        "lock": lock_info,
        "code_hash": started_code_hash,
        "code_changed_during_run": code_hash() != started_code_hash,
        "audit": {"created_utc": audit.get("created_utc"), "coverage": audit.get("coverage"), "cutoffs": audit.get("cutoffs"),
                  "data_sha256": {k: v["fingerprint"].get("sha256") for k, v in audit.get("sources", {}).items()}},
        "panel": panel_info,
        "feature_descriptions": volatility.FEATURE_DESCRIPTIONS,
        "folds": fold_records,
        "n_predictions": int(len(predictions)),
        "total_seconds": round(time.time() - t_run, 1),
        "environment": environment(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out
