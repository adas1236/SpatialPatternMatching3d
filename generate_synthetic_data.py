"""Command-line wrapper for ``espm3d.generate_synthetic_data``.

Run from a source checkout without installing first:

    python generate_synthetic_data.py list-patterns
    python generate_synthetic_data.py generate --n-objects 10000
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from espm3d.generate_synthetic_data import main


if __name__ == "__main__":
    main()
