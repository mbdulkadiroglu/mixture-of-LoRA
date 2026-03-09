"""
CLI wrapper for cascade storage cleanup.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cascade.storage_cleanup import main


if __name__ == "__main__":
    raise SystemExit(main())
