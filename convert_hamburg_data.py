"""Command-line wrapper for ``espm3d.convert_hamburg_data``.

Run from a source checkout without installing first:

    python convert_hamburg_data.py convert hamburg_buildings_facade_amenities_trees.csv
    python convert_hamburg_data.py list-patterns
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from espm3d.convert_hamburg_data import main


if __name__ == "__main__":
    main()
