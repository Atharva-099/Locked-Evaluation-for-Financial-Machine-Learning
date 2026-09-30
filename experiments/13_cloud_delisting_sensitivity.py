"""Run prespecified adverse-delisting label sensitivities on Kaggle/Colab."""
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
from p3_modellab.tasks.delisting import (  # noqa: E402
    ADVERSE_CODE_REASONS,
    SENSITIVITY_CODE_SETS,
    relabel_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True, help="primary delisting panel")
    parser.add_argument("--delist", type=Path, required=True, help="raw CRSP delisting-event parquet")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = pd.read_parquet(args.panel)
    events = pd.read_parquet(args.delist)
    audit = json.loads(args.audit.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for label, codes in SENSITIVITY_CODE_SETS.items():
        panel = relabel_panel(base, events, codes)
        reasons = {code: ADVERSE_CODE_REASONS[code] for code in sorted(codes)}
        print(f"starting {label}: {int(panel['target_adverse_delisting'].sum()):,} positive rows", flush=True)
        run_dir = run(
            models=list(DELISTING_MODEL_NAMES), run_type="exploratory", seed=args.seed,
            panel=panel, audit=audit, out_root=args.output,
            label_definition=label, adverse_code_reasons=reasons, progress=print,
        )
        build_report(run_dir)
        print(f"completed {label}: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
