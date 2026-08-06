"""Launch H5Grid as a desktop app.

    python app.py [optional_file.h5]

This opens H5Grid in a native window rather than a browser. It's the same thing
the packaged executable runs. For the browser version, use the `h5grid` command
(see the README).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when run straight from a source checkout, before
# `pip install`.
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from h5grid.desktop import run_desktop  # noqa: E402


def main() -> int:
    file = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    return run_desktop(file)


if __name__ == "__main__":
    raise SystemExit(main())
