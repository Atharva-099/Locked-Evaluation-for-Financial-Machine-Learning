"""Walk-forward runner for Stage 9 adverse-delisting probabilities."""
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
from .delisting_models import DELISTING_MODEL_NAMES, delisting_model_spec, make_delisting_model
from .manifest import code_hash, environment
from .runner import QUICK_FIRM_SHARE, QUICK_REFIT_EVERY, VALIDATION_MONTHS, quick_sample
from .splits import walk_forward
from .tasks import delisting


class DelistingRunError(RuntimeError):
    pass


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(_canonical(value), sort_keys=True).encode()).hexdigest()[:16]


def reserved_start(audit: dict) -> pd.Timestamp:
    """The project reserves outcomes from 2022 onward for one final evaluation."""
    last = pd.Period(audit["cutoffs"]["twelve_month_ahead_last_prediction_month"], freq="M")
    return pd.Timestamp(year=last.year - 1, month=1, day=1)


def default_test_years(audit: dict, run_type: str) -> list[int]:
    first_locked = reserved_start(audit).year
    last = pd.Period(audit["cutoffs"]["twelve_month_ahead_last_prediction_month"], freq="M").year
    if run_type == "locked":
        return list(range(first_locked, last + 1))
    # A 2021 forecast uses outcomes from sealed 2022, so it is an embargo year.
    return list(range(2000, first_locked - 1))


def run_spec(models: list[str], seed: int, quick: bool, cfg: delisting.DelistingConfig,
             label_definition: str = "primary",
             adverse_code_reasons: dict[int, str] | None = None) -> dict:
    adverse_code_reasons = adverse_code_reasons or delisting.ADVERSE_CODE_REASONS
    return _canonical({
        "task": delisting.TASK,
        "panel_config": asdict(cfg),
        "features": delisting.FEATURES,
        "models": [delisting_model_spec(m, delisting.FEATURES, seed, quick) for m in models],
        "seed": seed,
        "validation_months": VALIDATION_MONTHS,
        "refit_every": QUICK_REFIT_EVERY if quick else 1,
        "firm_sample_percent": QUICK_FIRM_SHARE if quick else 100,
        "selection_metric": "validation log loss",
        "probability_target": "adverse delisting in months t+1 through t+12",
        "label_definition": label_definition,
        "adverse_codes": sorted(adverse_code_reasons),
        "sealed_outcome_start": "2022-01-01",
        "embargo_year": 2021,
    })


def lock_path() -> Path:
    return config.LOCKS_DIR / f"{delisting.TASK}.json"


def create_lock(models: list[str], seed: int = 0, cfg: delisting.DelistingConfig = delisting.DelistingConfig(),
                note: str = "") -> Path:
    path = lock_path()
    if path.exists():
        raise DelistingRunError(f"{path} already exists")
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
        raise DelistingRunError("a locked delisting run needs a delisting lock file")
    lock = json.loads(path.read_text())
    if lock["spec_hash"] != _hash(spec):
        raise DelistingRunError("the delisting run does not match the locked specification")
    prior = []
    for folder in config.RUNS_DIR.glob("*-delisting-locked"):
        manifest = folder / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("lock", {}).get("spec_hash") == lock["spec_hash"]:
            prior.append(folder.name)
    if prior and not allow_rerun:
        raise DelistingRunError(f"the locked delisting evaluation already ran ({prior[0]})")
    return {"spec_hash": lock["spec_hash"], "lock_created_utc": lock["created_utc"],
            "lock_code_hash": lock["code_hash"], "code_changed_since_lock": lock["code_hash"] != code_hash(),
            "previous_locked_runs": prior}


def run(
    models: list[str] = list(DELISTING_MODEL_NAMES),
    run_type: str = "exploratory",
    test_years: list[int] | None = None,
    quick: bool = False,
    seed: int = 0,
    allow_rerun: bool = False,
    panel: pd.DataFrame | None = None,
    audit: dict | None = None,
    out_root: Path | None = None,
    label_definition: str = "primary",
    adverse_code_reasons: dict[int, str] | None = None,
    progress=print,
) -> Path:
    started_hash = code_hash()
    if run_type == "locked" and quick:
        raise DelistingRunError("quick mode cannot be a locked evaluation")
    unknown = [m for m in models if m not in DELISTING_MODEL_NAMES]
    if unknown:
        raise DelistingRunError(f"unknown delisting models {unknown}")
    audit = audit or load_audit()
    cfg = delisting.DelistingConfig()
    if panel is None:
        panel, panel_meta = delisting.load_or_build(cfg, progress=progress)
        panel_info = {"source": "results/panels", "meta_created_utc": panel_meta["created_utc"],
                      "code_hash": panel_meta["code_hash"], "summary": panel_meta["summary"]}
    else:
        panel_info = {"source": "supplied by caller", "rows": int(len(panel))}
    validate_panel(panel)

    adverse_code_reasons = adverse_code_reasons or delisting.ADVERSE_CODE_REASONS
    spec = run_spec(models, seed, quick, cfg, label_definition, adverse_code_reasons)
    lock = _check_lock(spec, allow_rerun) if run_type == "locked" else None
    years = test_years or default_test_years(audit, run_type)
    reserve = reserved_start(audit)
    work = panel[quick_sample(panel["permno"])].reset_index(drop=True) if quick else panel
    folds = walk_forward(work, years, reserve, run_type, validation_months=VALIDATION_MONTHS,
                         refit_every=QUICK_REFIT_EVERY if quick else 1)

    label_suffix = "" if label_definition == "primary" else f"-{label_definition}"
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}-{delisting.TASK}{label_suffix}-{run_type}{'-quick' if quick else ''}"
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
            model = make_delisting_model(name, delisting.FEATURES, seed, quick)
            tuning = model.fit(train, val)
            forecast = model.predict(test)
            if not np.all(np.isfinite(forecast)) or np.any((forecast <= 0) | (forecast >= 1)):
                raise DelistingRunError(f"{name} produced invalid probabilities")
            joblib.dump(model, out / "models" / f"{name}_fold{fold.fold:02d}.joblib")
            record["models"][name] = {**tuning, "fit_predict_seconds": round(time.time() - t0, 2)}
            predictions.append(pd.DataFrame({"row_id": test["row_id"], "fold": fold.fold,
                                             "model": name, "forecast": forecast}))
            progress(f"delisting fold {fold.fold} {fold.test_years}: {name} done in {time.time() - t0:.1f}s")
        records.append(record)

    keep = ["row_id", "permno", "yyyymm", *TIME_COLS, "target_adverse_delisting",
            "next_adverse_date", "next_adverse_code"]
    pred = pd.concat(predictions, ignore_index=True).merge(
        work[keep], on="row_id", how="left", validate="many_to_one"
    )
    pred.to_parquet(out / "predictions.parquet", index=False)
    source_hashes = {
        name: source.get("fingerprint", {}).get("sha256")
        for name, source in audit.get("sources", {}).items()
    }
    manifest = {
        "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_type": run_type, "quick": quick, "evidence": not quick, "task": delisting.TASK,
        "test_years": years, "reserved_start": str(reserve.date()), "embargo_years": [2021],
        "spec": spec, "spec_hash": _hash(spec), "lock": lock, "code_hash": started_hash,
        "code_changed_during_run": code_hash() != started_hash,
        "audit": {"created_utc": audit.get("created_utc"), "coverage": audit.get("coverage"),
                  "cutoffs": audit.get("cutoffs"), "data_sha256": source_hashes},
        "panel": panel_info, "feature_descriptions": delisting.FEATURE_DESCRIPTIONS,
        "label_definition": label_definition,
        "adverse_codes": {str(k): v for k, v in adverse_code_reasons.items()},
        "folds": records, "n_predictions": int(len(pred)), "total_seconds": round(time.time() - started, 1),
        "environment": environment(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out
