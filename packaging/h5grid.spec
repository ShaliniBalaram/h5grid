# PyInstaller spec for the H5Grid desktop app.
#
#   pyinstaller packaging/h5grid.spec
#
# Produces a standalone binary (a .exe on Windows, a .app on macOS, an ELF on
# Linux) that opens H5Grid in a native window — no Python, no pip, no browser.
#
# The two things that break a naive build, both handled below:
#   1. the built web frontend (backend/h5grid/static) must be bundled as data,
#      or the app serves nothing;
#   2. hdf5plugin ships compiled filter libraries (blosc, zstd, …) that
#      PyInstaller does not discover, and without them reading a PyTables/pywr
#      file fails with "can't open directory .../plugin".

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files

# The spec runs with CWD at the repo root (that is how the CI invokes it).
ROOT = Path.cwd()
PKG = ROOT / "backend" / "h5grid"

datas = []
binaries = []
hiddenimports = []

# 1. The built single-page app.
static = PKG / "static"
if not (static / "index.html").exists():
    raise SystemExit(
        "packaging/h5grid.spec: backend/h5grid/static/index.html is missing.\n"
        "Build the frontend first:  cd frontend && npm ci && npm run build"
    )
datas += [(str(static), "h5grid/static")]

# 2. hdf5plugin — grab its data, binaries and submodules wholesale so the filter
#    libraries travel with the app.
for collector in ("hdf5plugin",):
    d, b, h = collect_all(collector)
    datas += d
    binaries += b
    hiddenimports += h

# h5py and tables carry compiled extensions and, on some platforms, bundled HDF5
# libraries; collect their data to be safe.
for pkg in ("tables", "h5py"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

# Uvicorn/anyio/click pull parts in dynamically; naming them avoids missing
# imports at runtime in the frozen app.
hiddenimports += [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "hdf5plugin",
]

block_cipher = None

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "backend")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pytest"],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="H5Grid",
    debug=False,
    strip=False,
    upx=False,
    console=False,  # no terminal window; it's a GUI app
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# On macOS, wrap the binary in a .app bundle so it behaves like a normal app.
if sys.platform == "darwin":
    app = BUNDLE(
        exe,
        name="H5Grid.app",
        bundle_identifier="com.h5grid.app",
        info_plist={
            "NSHighResolutionCapable": True,
            "CFBundleShortVersionString": "0.1.2",
        },
    )
