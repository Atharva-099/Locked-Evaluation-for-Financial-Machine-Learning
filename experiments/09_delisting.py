"""Stage 9: train adverse-delisting models and build their report.

Examples:
  python experiments/09_delisting.py --quick
  python experiments/09_delisting.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3_modellab.delisting_report import build_report
from p3_modellab.delisting_runner import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--years", default=None, help="single year, comma list, or inclusive range")
    args = parser.parse_args()
    years = None
    if args.years:
        if "-" in args.years:
            first, last = map(int, args.years.split("-"))
            years = list(range(first, last + 1))
        else:
            years = [int(value) for value in args.years.split(",")]
    out = run(quick=args.quick, test_years=years)
    print(f"run saved to {out}")
    report = build_report(out)
    print(f"report written to {report}")


if __name__ == "__main__":
    main()
