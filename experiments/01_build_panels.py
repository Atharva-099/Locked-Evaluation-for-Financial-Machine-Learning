"""Step 1: build the volatility table (one row per stock per month). Writes results/panels/."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3_modellab.tasks import volatility  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild even if the saved table is up to date")
    args = ap.parse_args()
    t0 = time.time()
    panel, meta = volatility.load_or_build(rebuild=args.rebuild)
    print(json.dumps(meta["summary"], indent=2))
    print(f"{len(panel):,} rows, {panel.shape[1]} columns, {time.time() - t0:.0f}s")
