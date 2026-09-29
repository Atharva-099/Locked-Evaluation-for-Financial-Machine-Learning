"""Step 3: score a saved run, break results down, and write charts and a plain-language summary.

Usage: python experiments/03_report.py [run_folder]   (default: newest non-quick exploratory run)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3_modellab import config, plots  # noqa: E402
from p3_modellab.report import build_report  # noqa: E402
from p3_modellab.report_text import summary_markdown  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run = Path(sys.argv[1])
    else:
        run = max((p for p in config.RUNS_DIR.glob("*-volatility-exploratory")), key=lambda p: p.stat().st_mtime)
    out = build_report(run)
    plots.write_all(out)
    (out / "REPORT.md").write_text(summary_markdown(out))
    print((out / "REPORT.md").read_text())
    print(f"report written to {out}")
