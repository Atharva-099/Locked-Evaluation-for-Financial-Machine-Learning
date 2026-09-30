"""Stage 9: build and audit the twelve-month adverse-delisting panel."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3_modellab.tasks import delisting


if __name__ == "__main__":
    _, meta = delisting.load_or_build(progress=print)
    print(json.dumps(meta["summary"], indent=2))
