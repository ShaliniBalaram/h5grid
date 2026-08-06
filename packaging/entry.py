"""Entry point for the packaged executable — opens the desktop window.

A double-clicked binary has no command-line arguments, so this defaults to the
desktop window with no file loaded. A path may still be passed (e.g. from a
"Open with" association), and it's honoured.
"""

from __future__ import annotations

import sys


def _selftest(path: str) -> int:
    """Prove the frozen binary is functional without opening a window.

    Reads a slice of a (blosc-compressed) file through the real reader, which is
    exactly what shakes out a missing hdf5plugin — the failure mode this whole
    build is most likely to hit. CI runs this on every platform's binary.
    """
    from h5grid.files import FileRegistry

    reg = FileRegistry()
    entry = reg.open(path)
    reader = entry.reader("/reservoir") if "/reservoir" else None
    frame = reader.read(0, 5)
    reg.close_all()
    if frame.empty:
        print("SELFTEST FAIL: read returned no rows")
        return 1
    print(f"SELFTEST OK: read {frame.shape[0]} rows x {frame.shape[1]} cols")
    return 0


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        return _selftest(sys.argv[2])

    from h5grid.desktop import run_desktop

    file = sys.argv[1] if len(sys.argv) > 1 else None
    return run_desktop(file)


if __name__ == "__main__":
    raise SystemExit(main())
