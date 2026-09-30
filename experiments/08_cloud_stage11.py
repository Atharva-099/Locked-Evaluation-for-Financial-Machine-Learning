"""Run the eight-model volatility comparison from a portable private panel bundle.

This entry point is intended for Kaggle or Colab. It bypasses panel rebuilding,
so the cloud runtime needs the saved volatility panel, audit JSON, and CRSP names
file rather than every daily source file.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p3_modellab import plots  # noqa: E402
from p3_modellab.models import MODEL_NAMES  # noqa: E402
from p3_modellab.report import build_report  # noqa: E402
from p3_modellab.report_text import summary_markdown  # noqa: E402
from p3_modellab.runner import run  # noqa: E402


def years(value: str) -> list[int]:
    first, last = value.split("-", 1)
    return list(range(int(first), int(last) + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True, help="private volatility.parquet")
    parser.add_argument("--audit", type=Path, required=True, help="audit.json saved with the panel")
    parser.add_argument("--names", type=Path, required=True, help="private CRSP msenames parquet")
    parser.add_argument("--output", type=Path, required=True, help="folder for run outputs")
    parser.add_argument("--years", type=years, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    audit = json.loads(args.audit.read_text())
    names = pd.read_parquet(args.names, columns=["permno", "namedt"])
    names["namedt"] = pd.to_datetime(names["namedt"])
    first_listed = names.groupby("permno")["namedt"].min()
    args.output.mkdir(parents=True, exist_ok=True)

    run_dir = run(
        models=list(MODEL_NAMES), test_years=args.years, quick=args.quick,
        seed=args.seed, panel=panel, audit=audit, out_root=args.output, progress=print,
    )
    report_dir = build_report(run_dir, panel=panel, first_listed=first_listed)
    plots.write_all(report_dir)
    (report_dir / "REPORT.md").write_text(summary_markdown(report_dir))
    print(f"run saved to {run_dir}")
    print(f"report saved to {report_dir}")


if __name__ == "__main__":
    main()
