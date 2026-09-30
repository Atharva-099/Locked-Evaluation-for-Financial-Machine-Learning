"""Command line: python -m p3_modellab <command> ...

  build-panel [--task volatility|returns|delisting] [--rebuild]
  run [--task volatility|returns|delisting] [--models ...] [--type exploratory|locked] [--years 2000-2021] [--quick] [--seed 0] [--allow-rerun]
  lock [--task volatility|returns|delisting] [--models ...] [--note "..."]
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
    b.add_argument("--task", default="volatility", choices=["volatility", "returns", "delisting"])
    b.add_argument("--rebuild", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--task", default="volatility", choices=["volatility", "returns", "delisting"])
    r.add_argument("--models", default=None)
    r.add_argument("--type", default="exploratory", choices=["exploratory", "locked"])
    r.add_argument("--years", type=_years, default=None)
    r.add_argument("--quick", action="store_true")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--allow-rerun", action="store_true")
    lk = sub.add_parser("lock")
    lk.add_argument("--task", default="volatility", choices=["volatility", "returns", "delisting"])
    lk.add_argument("--models", default=None)
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
        from .tasks import delisting, returns, volatility
        task = {"volatility": volatility, "returns": returns, "delisting": delisting}[args.task]
        _, meta = task.load_or_build(rebuild=args.rebuild)
        print(json.dumps(meta["summary"], indent=2))
    elif args.cmd == "run":
        if args.task == "returns":
            from .return_models import RETURN_MODEL_NAMES
            from .return_runner import run
            defaults = RETURN_MODEL_NAMES
        elif args.task == "delisting":
            from .delisting_models import DELISTING_MODEL_NAMES
            from .delisting_runner import run
            defaults = DELISTING_MODEL_NAMES
        else:
            from .runner import run
            defaults = MODEL_NAMES
        models = args.models.split(",") if args.models else list(defaults)
        out = run(models=models, run_type=args.type, test_years=args.years, quick=args.quick,
                  seed=args.seed, allow_rerun=args.allow_rerun)
        print(f"run saved to {out}")
    elif args.cmd == "lock":
        if args.task == "returns":
            from .return_models import RETURN_MODEL_NAMES
            from .return_runner import create_lock
            defaults = RETURN_MODEL_NAMES
        elif args.task == "delisting":
            from .delisting_models import DELISTING_MODEL_NAMES
            from .delisting_runner import create_lock
            defaults = DELISTING_MODEL_NAMES
        else:
            from .runner import create_lock
            defaults = MODEL_NAMES
        models = args.models.split(",") if args.models else list(defaults)
        print(f"lock written to {create_lock(models, note=args.note)}")
    elif args.cmd == "report":
        from pathlib import Path
        run_dir = Path(args.run)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest.get("task") == "returns":
            from .return_report import build_report
            out = build_report(run_dir)
        elif manifest.get("task") == "delisting":
            from .delisting_report import build_report
            out = build_report(run_dir)
        else:
            from . import plots
            from .report import build_report
            from .report_text import summary_markdown
            out = build_report(run_dir)
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
