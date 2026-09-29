"""Step 2: check a saved run can be trusted and re-created.

- every saved model reproduces its stored forecasts exactly;
- every forecast was made at or after its model's fit time;
- exploratory runs contain no answer from the sealed period;
- each fold's test rows match the recorded fingerprint;
- the run's data fingerprints match the current audit.

Usage: python experiments/02_verify_run.py [run_folder]   (default: newest run)
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3_modellab import config  # noqa: E402
from p3_modellab.data.audit import load_audit  # noqa: E402
from p3_modellab.tasks import volatility  # noqa: E402


def ids_hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.sort(a).astype("int64").tobytes()).hexdigest()[:16]


def main(run_dir: Path) -> int:
    man = json.loads((run_dir / "manifest.json").read_text())
    pred = pd.read_parquet(run_dir / "predictions.parquet")
    panel = pd.read_parquet(volatility.panel_paths()[0]).set_index("row_id")
    problems = []

    audit = load_audit()
    now = {k: v["fingerprint"].get("sha256") for k, v in audit["sources"].items()}
    if man["audit"]["data_sha256"] != now:
        problems.append("data fingerprints differ from the current audit")

    for f in man["folds"]:
        rows = pred[pred["fold"] == f["fold"]]
        if (rows["prediction_time"] < pd.Timestamp(f["fit_cutoff"])).any():
            problems.append(f"fold {f['fold']}: forecasts before fit time")
        for name in f["models"]:
            r = rows[rows["model"] == name]
            if ids_hash(r["row_id"].to_numpy()) != f["test_ids_sha256"]:
                problems.append(f"fold {f['fold']} {name}: test rows differ from the recorded fingerprint")
            model = joblib.load(run_dir / "models" / f"{name}_fold{f['fold']:02d}.joblib")
            again = model.predict(panel.loc[r["row_id"]].reset_index())
            if not np.array_equal(again, r["forecast"].to_numpy()):
                worst = float(np.max(np.abs(again - r["forecast"].to_numpy()) / r["forecast"].to_numpy()))
                problems.append(f"fold {f['fold']} {name}: saved model does not reproduce stored forecasts (max relative diff {worst:.2e})")
        print(f"fold {f['fold']} {f['test_years']}: checked {len(rows):,} forecasts")

    if man["run_type"] == "exploratory" and (pred["label_available_time"] >= pd.Timestamp(man["reserved_start"])).any():
        problems.append("exploratory run contains answers from the sealed period")
    per_model = pred.groupby("model").size()
    if per_model.nunique() != 1:
        problems.append(f"models have different numbers of forecasts: {per_model.to_dict()}")

    print(f"\nrun {man['run_id']}: {len(pred):,} forecasts, {len(man['folds'])} folds, models {sorted(per_model.index)}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(" -", p)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else max(config.RUNS_DIR.glob("*"), key=lambda p: p.stat().st_mtime)
    sys.exit(main(target))
