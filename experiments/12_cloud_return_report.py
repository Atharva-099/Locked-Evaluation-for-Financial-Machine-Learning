"""Rebuild the Stage 8 report, including turnover and cost diagnostics, on cloud CPU."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p3_modellab.return_report import build_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    run_dir = args.output / manifest["run_id"]
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.manifest, run_dir / "manifest.json")
    shutil.copy2(args.predictions, run_dir / "predictions.parquet")
    report_dir = build_report(run_dir)
    print(f"turnover report saved to {report_dir}")


if __name__ == "__main__":
    main()
