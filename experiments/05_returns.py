"""Stage 8: build returns data, run the four-model exploratory comparison, and report it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3_modellab.return_report import build_report
from p3_modellab.return_runner import run
from p3_modellab.tasks import returns


if __name__ == "__main__":
    returns.load_or_build(progress=print)
    run_dir = run(progress=print)
    report_dir = build_report(Path(run_dir))
    print(f"returns report written to {report_dir}")
