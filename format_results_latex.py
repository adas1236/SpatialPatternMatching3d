#!/usr/bin/env python3
"""Command-line wrapper for ``espm3d.format_results_latex``.

Run from a source checkout without installing first:

    python format_results_latex.py --results-root results --output results/table.tex
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from espm3d.format_results_latex import main


if __name__ == "__main__":
    raise SystemExit(main())
