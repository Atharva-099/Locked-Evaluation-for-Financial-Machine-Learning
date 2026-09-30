"""Walk-forward runner for Stage 8 return forecasts."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .contract import TIME_COLS, validate_panel
from .data.audit import load_audit
from .manifest import code_hash, environment
from .return_models import RETURN_MODEL_NAMES, make_return_model, return_model_spec
from .runner import QUICK_FIRM_SHARE, QUICK_REFIT_EVERY, VALIDATION_MONTHS, quick_sample
from .splits import walk_forward
from .tasks import returns


class ReturnRunError(RuntimeError):
    pass


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(_canonical(value), sort_keys=True).encode()).hexdigest()[:16]


def reserved_start(audit: dict) -> pd.Timestamp:
    last = pd.Period(audit["cutoffs"]["one_month_ahead_last_prediction_month"], freq="M")
    return pd.Timestamp(year=last.year - 2, month=1, day=1)


def default_test_years(audit: dict, run_type: str) -> list[int]:
    first_locked = reserved_start(audit).year
    last = pd.Period(audit["cutoffs"]["one_month_ahead_last_prediction_month"], freq="M").year
    return list(range(2000, first_locked)) if run_type == "exploratory" else list(range(first_locked, last + 1))


def run_spec(models: list[str], seed: int, quick: bool, cfg: returns.ReturnConfig) -> dict:
    return _canonical({
        "task": returns.TASK,
        "panel_config": asdict(cfg),
        "features": returns.FEATURES,
        "models": [return_model_spec(m, returns.FEATURES, seed, quick) for m in models],
        "seed": seed,
        "validation_months": VALIDATION_MONTHS,
        "refit_every": QUICK_REFIT_EVERY if quick else 1,
        "firm_sample_percent": QUICK_FIRM_SHARE if quick else 100,
        "target_training_clip": "0.1 and 99.9 percentiles learned from each training fold",
        "selection_metric": "mean monthly Spearman rank IC on validation rows; MSE breaks ties",
    })


def lock_path() -> Path:
    return config.LOCKS_DIR / f"{returns.TASK}.json"


def create_lock(models: list[str], seed: int = 0, cfg: returns.ReturnConfig = returns.ReturnConfig(), note: str = "") -> Path:
    path = lock_path()
    if path.exists():
        raise ReturnRunError(f"{path} already exists")
    spec = run_spec(models, seed, False, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note, "spec": spec, "spec_hash": _hash(spec), "code_hash": code_hash(),
    }, indent=2))
    return path


def _check_lock(spec: dict, allow_rerun: bool) -> dict:
    path = lock_path()
    if not path.exists():
        raise ReturnRunError("a locked returns run needs a returns lock file")
    lock = json.loads(path.read_text())
    if lock["spec_hash"] != _hash(spec):
        raise ReturnRunError("the returns run does not match the locked specification")
    prior = []
    for folder in config.RUNS_DIR.glob("*-returns-locked"):
        manifest = folder / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("lock", {}).get("spec_hash") == lock["spec_hash"]:
            prior.append(folder.name)
    if prior and not allow_rerun:
        raise ReturnRunError(f"the locked returns evaluation already ran ({prior[0]})")
    return {"spec_hash": lock["spec_hash"], "lock_created_utc": lock["created_utc"],
            "lock_code_hash": lock["code_hash"], "code_changed_since_lock": lock["code_hash"] != code_hash(),
            "previous_locked_runs": prior}


def run(
    models: list[str] = list(RETURN_MODEL_NAMES),
    run_type: str = "exploratory",
    test_years: list[int] | None = None,
    quick: bool = False,
    seed: int = 0,
    allow_rerun: bool = False,
    panel: pd.DataFrame | None = None,
    audit: dict | None = None,
    out_root: Path | None = None,
    progress=print,
) -> Path:
    started_hash = code_hash()
    if run_type == "locked" and quick:
        raise ReturnRunError("quick mode cannot be a locked evaluation")
    unknown = [m for m in models if m not in RETURN_MODEL_NAMES]
    if unknown:
        raise ReturnRunError(f"unknown return models {unknown}")
    audit = audit or load_audit()
    cfg = returns.ReturnConfig()
    if panel is None:
        panel, panel_meta = returns.load_or_build(cfg, progress=progress)
        panel_info = {"source": "results/panels", "meta_created_utc": panel_meta["created_utc"],
                      "code_hash": panel_meta["code_hash"], "summary": panel_meta["summary"]}
    else:
        panel_info = {"source": "supplied by caller", "rows": int(len(panel))}
    validate_panel(panel)

    spec = run_spec(models, seed, quick, cfg)
    lock = _check_lock(spec, allow_rerun) if run_type == "locked" else None
    years = test_years or default_test_years(audit, run_type)
    reserve = reserved_start(audit)
    work = panel[quick_sample(panel["permno"])].reset_index(drop=True) if quick else panel
    folds = walk_forward(work, years, reserve, run_type, validation_months=VALIDATION_MONTHS,
                         refit_every=QUICK_REFIT_EVERY if quick else 1)

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}-{returns.TASK}-{run_type}{'-quick' if quick else ''}"
    out = Path(out_root or config.RUNS_DIR) / run_id
    (out / "models").mkdir(parents=True, exist_ok=False)
    by_id = work.set_index("row_id")
    predictions, records = [], []
    started = time.time()
    for fold in folds:
        train = by_id.loc[fold.train_ids].reset_index()
        val = by_id.loc[fold.val_ids].reset_index()
        test = by_id.loc[fold.test_ids].reset_index()
        record = {**fold.summary(), "models": {}}
        for name in models:
            t0 = time.time()
            model = make_return_model(name, returns.FEATURES, seed, quick)
            tuning = model.fit(train, val)
            forecast = model.predict(test)
            if not np.all(np.isfinite(forecast)):
                raise ReturnRunError(f"{name} produced non-finite return forecasts")
            joblib.dump(model, out / "models" / f"{name}_fold{fold.fold:02d}.joblib")
            record["models"][name] = {**tuning, "fit_predict_seconds": round(time.time() - t0, 2)}
            predictions.append(pd.DataFrame({"row_id": test["row_id"], "fold": fold.fold,
                                             "model": name, "forecast": forecast}))
            progress(f"returns fold {fold.fold} {fold.test_years}: {name} done in {time.time() - t0:.1f}s")
        records.append(record)

    keep = ["row_id", "permno", "yyyymm", *TIME_COLS, "target_ret", "target_includes_dlret"]
    pred = pd.concat(predictions, ignore_index=True).merge(work[keep], on="row_id", how="left", validate="many_to_one")
    pred.to_parquet(out / "predictions.parquet", index=False)
    source_hashes = {
        name: source.get("fingerprint", {}).get("sha256")
        for name, source in audit.get("sources", {}).items()
    }
    manifest = {
        "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_type": run_type, "quick": quick, "evidence": not quick, "task": returns.TASK,
        "test_years": years, "reserved_start": str(reserve.date()), "spec": spec, "spec_hash": _hash(spec),
        "lock": lock, "code_hash": started_hash, "code_changed_during_run": code_hash() != started_hash,
        "audit": {"created_utc": audit.get("created_utc"), "coverage": audit.get("coverage"),
                  "cutoffs": audit.get("cutoffs"), "data_sha256": source_hashes},
        "panel": panel_info, "feature_descriptions": returns.FEATURE_DESCRIPTIONS,
        "folds": records, "n_predictions": int(len(pred)), "total_seconds": round(time.time() - started, 1),
        "environment": environment(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out
