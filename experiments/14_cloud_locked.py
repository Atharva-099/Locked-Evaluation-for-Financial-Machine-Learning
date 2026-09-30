"""Run one frozen, sealed-period task from a portable Kaggle/Colab bundle.

The lock must be created before this entry point is invoked.  The script copies
that immutable specification into the cloud result root, verifies it through
the normal task runner, evaluates the reserved years once, and writes the
corresponding report beside the predictions and fitted models.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p3_modellab import config  # noqa: E402


def _configure_result_root(output: Path, task: str, lock: Path) -> None:
    """Point the standard lock/run checks at writable cloud storage."""
    config.RESULTS_DIR = output
    config.RUNS_DIR = output / "runs"
    config.LOCKS_DIR = output / "locks"
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lock, config.LOCKS_DIR / f"{task}.json")


def _volatility(panel: pd.DataFrame, audit: dict, output: Path, names: Path | None, seed: int) -> Path:
    if names is None:
        raise ValueError("--names is required for the volatility report")
    from p3_modellab import plots
    from p3_modellab.models import MODEL_NAMES
    from p3_modellab.report import build_report
    from p3_modellab.report_text import summary_markdown
    from p3_modellab.runner import run

    listed = pd.read_parquet(names, columns=["permno", "namedt"])
    listed["namedt"] = pd.to_datetime(listed["namedt"])
    first_listed = listed.groupby("permno")["namedt"].min()
    run_dir = run(models=list(MODEL_NAMES), run_type="locked", seed=seed,
                  panel=panel, audit=audit, out_root=config.RUNS_DIR, progress=print)
    report_dir = build_report(run_dir, panel=panel, first_listed=first_listed)
    plots.write_all(report_dir)
    (report_dir / "REPORT.md").write_text(summary_markdown(report_dir))
    return run_dir


def _returns(panel: pd.DataFrame, audit: dict, seed: int) -> Path:
    from p3_modellab.return_models import RETURN_MODEL_NAMES
    from p3_modellab.return_report import build_report
    from p3_modellab.return_runner import run

    run_dir = run(models=list(RETURN_MODEL_NAMES), run_type="locked", seed=seed,
                  panel=panel, audit=audit, out_root=config.RUNS_DIR, progress=print)
    build_report(run_dir)
    return run_dir


def _delisting(panel: pd.DataFrame, audit: dict, seed: int, lock_spec: dict) -> Path:
    from p3_modellab import delisting_models

    # The frozen delisting spec records the training thread count.  Reapply it
    # explicitly because cloud CPU discovery can differ from the machine that
    # wrote the lock; thread availability is an execution detail, not a reason
    # to rewrite a sealed specification.
    locked_cores = {
        int(model["n_jobs"])
        for model in lock_spec["models"]
        if "n_jobs" in model
    }
    if len(locked_cores) != 1:
        raise ValueError(f"expected one frozen delisting n_jobs value, found {sorted(locked_cores)}")
    delisting_models.MODEL_CORES = locked_cores.pop()

    from p3_modellab.delisting_models import DELISTING_MODEL_NAMES
    from p3_modellab.delisting_report import build_report
    from p3_modellab.delisting_runner import run

    run_dir = run(models=list(DELISTING_MODEL_NAMES), run_type="locked", seed=seed,
                  panel=panel, audit=audit, out_root=config.RUNS_DIR, progress=print)
    build_report(run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["volatility", "returns", "delisting"])
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--names", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    lock_spec = json.loads(args.lock.read_text())["spec"]
    _configure_result_root(args.output, args.task, args.lock)
    panel = pd.read_parquet(args.panel)
    audit = json.loads(args.audit.read_text())

    if args.task == "volatility":
        run_dir = _volatility(panel, audit, args.output, args.names, args.seed)
    elif args.task == "returns":
        run_dir = _returns(panel, audit, args.seed)
    else:
        run_dir = _delisting(panel, audit, args.seed, lock_spec)
    print(f"sealed {args.task} run saved to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
