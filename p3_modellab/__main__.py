"""Command line: python -m p3_modellab <command> ...

  build-panel [--rebuild]
  run [--models persistence,har,ridge,lightgbm] [--type exploratory|locked] [--years 2000-2021] [--quick] [--seed 0] [--allow-rerun]
  lock [--models ...] [--note "..."]
  report --run <run folder>
  sweep --kind settings|greedy|data|noise [--demo] [--seeds 0,1,2] [--firm-percent 25]
"""
import argparse
import json

from .models import MODEL_NAMES


def _years(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(prog="p3_modellab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-panel")
    b.add_argument("--rebuild", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--models", default=",".join(MODEL_NAMES))
    r.add_argument("--type", default="exploratory", choices=["exploratory", "locked"])
    r.add_argument("--years", type=_years, default=None)
    r.add_argument("--quick", action="store_true")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--allow-rerun", action="store_true")
    lk = sub.add_parser("lock")
    lk.add_argument("--models", default=",".join(MODEL_NAMES))
    lk.add_argument("--note", default="")
    rp = sub.add_parser("report")
    rp.add_argument("--run", required=True, help="run folder to score")
    sw = sub.add_parser("sweep")
    sw.add_argument("--kind", required=True, choices=["settings", "greedy", "data", "noise"])
    sw.add_argument("--seeds", default="0,1,2")
    sw.add_argument("--firm-percent", type=float, default=None)
    sw.add_argument("--demo", action="store_true", help="tiny size: only checks the calculations run (about a minute)")
    args = ap.parse_args()

    if args.cmd == "build-panel":
        from .tasks import volatility
        _, meta = volatility.load_or_build(rebuild=args.rebuild)
        print(json.dumps(meta["summary"], indent=2))
    elif args.cmd == "run":
        from .runner import run
        out = run(models=args.models.split(","), run_type=args.type, test_years=args.years, quick=args.quick,
                  seed=args.seed, allow_rerun=args.allow_rerun)
        print(f"run saved to {out}")
    elif args.cmd == "lock":
        from .runner import create_lock
        print(f"lock written to {create_lock(args.models.split(','), note=args.note)}")
    elif args.cmd == "report":
        from pathlib import Path

        from . import plots
        from .report import build_report
        from .report_text import summary_markdown
        out = build_report(Path(args.run))
        plots.write_all(out)
        (out / "REPORT.md").write_text(summary_markdown(out))
        print(f"report written to {out}")
    elif args.cmd == "sweep":
        from .sweep_plots import write_figures
        from .sweeps import run_sweep
        seeds = None if args.demo and args.seeds == "0,1,2" else [int(s) for s in args.seeds.split(",")]
        out = run_sweep(args.kind, seeds=seeds, firm_percent=args.firm_percent, demo=args.demo)
        write_figures(out)
        print(f"sweep saved to {out}")


if __name__ == "__main__":
    main()
