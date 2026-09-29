"""Step 0: measure what the saved WRDS data really covers. Writes results/audit/."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3_modellab.data.audit import run_audit, write_audit  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hash", action="store_true", help="skip sha256 of each file (faster, weaker fingerprint)")
    args = ap.parse_args()
    report = run_audit(full_hash=not args.no_hash)
    j, m = write_audit(report)
    print(m.read_text())
    print(f"wrote {j} and {m}")
