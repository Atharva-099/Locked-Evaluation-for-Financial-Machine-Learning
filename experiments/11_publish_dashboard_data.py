"""Export one or more saved reports as a small Streamlit deployment bundle."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3_modellab.dashboard_bundle import export_run, export_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", type=Path, help="saved run folders with completed reports")
    parser.add_argument("--sweeps", nargs="*", type=Path, default=[], help="completed sensitivity sweep folders")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    for run in args.runs:
        destination = export_run(run, args.output)
        print(f"dashboard bundle written to {destination}")
    for sweep in args.sweeps:
        destination = export_sweep(sweep, args.output)
        print(f"dashboard sweep written to {destination}")


if __name__ == "__main__":
    main()
