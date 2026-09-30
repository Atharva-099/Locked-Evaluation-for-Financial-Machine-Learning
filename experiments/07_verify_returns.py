"""Reproduce every saved Stage 8 forecast from its fitted model."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3_modellab import config
from p3_modellab.contract import validate_panel
from p3_modellab.tasks import returns


def verify(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("task") != "returns":
        raise ValueError(f"expected a returns run, got {manifest.get('task')!r}")
    if manifest.get("code_changed_during_run"):
        raise ValueError("code changed during the saved run")
    panel, _ = returns.load_or_build()
    validate_panel(panel)
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    expected_models = {spec["name"] for spec in manifest["spec"]["models"]}
    if set(predictions["model"]) != expected_models:
        raise ValueError("saved prediction models do not match the manifest")
    if predictions.groupby("model")["row_id"].apply(frozenset).nunique() != 1:
        raise ValueError("models were not evaluated on identical rows")

    by_id = panel.set_index("row_id")
    for fold in manifest["folds"]:
        fold_number = fold["fold"]
        for model_name in expected_models:
            saved = predictions[(predictions["fold"] == fold_number) & (predictions["model"] == model_name)]
            model = joblib.load(run_dir / "models" / f"{model_name}_fold{fold_number:02d}.joblib")
            reproduced = model.predict(by_id.loc[saved["row_id"]].reset_index())
            np.testing.assert_array_equal(reproduced, saved["forecast"].to_numpy())
    print(f"verified {len(predictions):,} saved forecasts exactly for {run_dir.name}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run = Path(sys.argv[1])
    else:
        run = max(config.RUNS_DIR.glob("*-returns-exploratory*"), key=lambda p: p.stat().st_mtime)
    verify(run)
