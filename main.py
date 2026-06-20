"""Run ESPM-3D scalability benchmarks from a source checkout.

Examples
--------
Quick run:
    python main.py --profile smoke

Larger run:
    python main.py --profile full --match-limit 1000
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from espm3d.benchmark import main


if __name__ == "__main__":
    main()
