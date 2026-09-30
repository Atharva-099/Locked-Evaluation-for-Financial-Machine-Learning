"""Run Stage 9 from a portable private panel bundle on Kaggle or Colab.

Only the saved delisting panel and audit JSON are required. Raw licensed CRSP
files stay outside the cloud job.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p3_modellab.delisting_models import DELISTING_MODEL_NAMES  # noqa: E402
from p3_modellab.delisting_report import build_report  # noqa: E402
from p3_modellab.delisting_runner import run  # noqa: E402


def years(value: str) -> list[int]:
    first, last = value.split("-", 1)
    return list(range(int(first), int(last) + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True, help="private delisting.parquet")
    parser.add_argument("--audit", type=Path, required=True, help="audit.json saved with the panel")
    parser.add_argument("--output", type=Path, required=True, help="folder for run outputs")
    parser.add_argument("--years", type=years, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    audit = json.loads(args.audit.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    run_dir = run(
        models=list(DELISTING_MODEL_NAMES), test_years=args.years, quick=args.quick,
        seed=args.seed, panel=panel, audit=audit, out_root=args.output, progress=print,
    )
    report_dir = build_report(run_dir)
    print(f"run saved to {run_dir}")
    print(f"report saved to {report_dir}")


if __name__ == "__main__":
    main()
