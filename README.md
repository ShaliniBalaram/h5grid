# H5Grid

[![PyPI](https://img.shields.io/pypi/v/h5grid)](https://pypi.org/project/h5grid/)
[![CI](https://github.com/ShaliniBalaram/h5grid/actions/workflows/ci.yml/badge.svg)](https://github.com/ShaliniBalaram/h5grid/actions/workflows/ci.yml)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/h5grid/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Open an HDF5 file and actually read it.** A tree on the left, a spreadsheet in
the middle, metadata on the right — running locally in your browser.

```bash
pip install h5grid
h5grid my_model_outputs.h5
```

---

## What it does

`.h5` files are the standard container for time series in water resource models
(pywr, and anything using pandas or PyTables). They're fast and compact — and
almost unreadable with existing free tools.

H5Grid fixes the three things that make them painful:

**1. It shows your table, not the storage internals.**
A DataFrame saved with `to_hdf()` is stored internally as `axis0`, `axis1`,
`block0_items`, `block0_values`. Other viewers show you exactly that. H5Grid
reconstructs the table you actually saved, with your real column names and
dtypes.

**2. Dates look like dates.**
A `DatetimeIndex` is stored as a 19-digit integer. H5Grid shows `1975-01-01`.
For pywr outputs it reads the `/time` table and uses it as a frozen date column,
so you can see *when* something happened without cross-referencing anything.

**3. pywr scenarios get their real names.**
A `TablesRecorder` run stores one array per node, with one axis per scenario.
Other viewers show anonymous columns `0, 1, 2 …`. H5Grid reads the `/scenarios`
table and labels them — `climate[0]`, `demand[2]` — and tags each node with what
it records (flow, volume, parameter). *No other HDF5 viewer does this.*

It also stays responsive on large files: nothing is ever loaded whole, every read
is a bounded slice, and there's no file-size limit.

### What you can do with it

| | |
|---|---|
| **Browse** | Group/dataset tree, with pandas tables and pywr nodes decoded |
| **Read** | Virtualised grid — scroll millions of rows smoothly |
| **Slice** | N-dimensional arrays get a selector per extra dimension |
| **Inspect** | Shape, dtype, chunking, compression, and all attributes |
| **Analyse** | Per-column min/max/mean/std/NaN-count |
| **Search** | Server-side value search (`>100`, `<=0`, `3..7`, or text) |
| **Plot** | Line chart with zoom that re-fetches at higher resolution |
| **Export** | CSV or XLSX, with dates preserved as real dates |
| **Copy** | Select a range, ⌘C / Ctrl+C, paste straight into Excel |

**It is read-only.** There is no write path anywhere, so it cannot corrupt a
results file — and it can open a file while a model run is still writing to it.

---

## Install

You need **Python 3.11, 3.12 or 3.13**. The package includes the web interface,
so you don't need Node.js or anything else.

<details open>
<summary><b>Windows</b></summary>

Open **PowerShell** or **Command Prompt**:

```powershell
py -m pip install h5grid
h5grid C:\path\to\model_outputs.h5
```

If `h5grid` isn't recognised afterwards, use `py -m h5grid.cli` instead, or add
Python's Scripts folder to your PATH. Don't have Python? Install it from
[python.org](https://www.python.org/downloads/) or the Microsoft Store, and tick
**"Add Python to PATH"** during setup.
</details>

<details open>
<summary><b>macOS</b></summary>

Open **Terminal**:

```bash
python3 -m pip install h5grid
h5grid ~/path/to/model_outputs.h5
```

macOS ships an old Python; if you hit trouble, install a current one with
[Homebrew](https://brew.sh): `brew install python@3.13`.
</details>

<details open>
<summary><b>Linux</b></summary>

```bash
python3 -m pip install h5grid
h5grid ~/path/to/model_outputs.h5
```

On Debian/Ubuntu you may need `sudo apt install python3-pip python3-venv` first.
</details>

### Recommended: install in a virtual environment

This keeps H5Grid from interfering with other Python projects.

```bash
# Windows
py -m venv h5grid-env
h5grid-env\Scripts\activate
pip install h5grid

# macOS / Linux
python3 -m venv h5grid-env
source h5grid-env/bin/activate
pip install h5grid
```

### Using it

```bash
h5grid path/to/file.h5   # open a file straight away
h5grid serve             # start with no file, then browse to one
```

Either way it starts a small local server, prints a URL, and opens your browser.
Nothing is uploaded anywhere — the server runs on your own machine and the files
never leave it.

Options: `--port`, `--host`, `--no-browser`, `--no-token`.

### Try it without your own data

```bash
git clone https://github.com/ShaliniBalaram/h5grid.git
cd h5grid
python -m pip install h5grid
python examples/make_examples.py
h5grid examples/example_3d_pywr.h5
```

| Example | Shape | Shows |
|---|---|---|
| `example_3d_pywr.h5` | 3653 × 5 × 4 | A pywr run: named scenarios, date index, node tags |
| `example_3d_plain.h5` | 365 × 24 × 12 | A plain 3D array with no metadata |
| `example_4d.h5` | 500 × 8 × 6 × 4 | 4D, so two dimension selectors appear |

---

## Finding your files

Because a browser can't hand a web page a file path, H5Grid provides its own
file picker. It offers shortcuts to **Home, Desktop, Documents**, your current
folder, and **every drive on your machine** — `C:`, `D:` and network drives on
Windows; mounted volumes on macOS and Linux. Model data usually lives on an
external or shared drive, so those are one click away.

The drive names you see are read from your computer when the picker opens.
Nothing is stored or hardcoded.

You can also paste a full path (to a file *or* a folder) into the box at the
bottom, and use **Back** and the clickable breadcrumbs to move around.

---

## What it understands

| Layout | How it's read |
|---|---|
| Plain h5py datasets | Direct slicing. 1D, 2D, ND with a slice selector, and compound dtypes as multi-column tables. |
| `pandas_type = 'frame_table'` | `HDFStore.select(start, stop)` — genuinely lazy row access. |
| `pandas_type = 'frame'` (fixed) | Can't be row-sliced, so it's decoded once under a 500 MB guard and cached. Above that it falls back to raw blocks with an explanation. |
| pywr `TablesRecorder` | One array per node, shaped `[timesteps] + scenarios.shape` — one axis per scenario, so often 3D. |

Any group with a `pandas_type` attribute is treated as one logical table and its
internals are hidden. The **Raw structure** toggle reveals them.

### pywr specifics

- **`/scenarios`** names the scenario axes, so columns read `climate[0]` rather
  than `col_0`, and the dimension selectors say "demand" rather than "dim 2".
- **`/scenario_combinations`** (when a model used explicit combinations)
  collapses the axes into one — columns then read `climate=2, demand=3`.
- **`PYWR_ATTRIBUTE`** and **`PYWR_TYPE`** tag each node with what it records and
  its model class, shown as a badge in the tree. Older pywr files spell these
  `pywr-attribute`/`pywr-type`; both are read.

Labels are only applied when a dataset's dimensions actually match the scenario
sizes, so an unrelated summary array in the same file keeps positional names.

Integer epoch columns are decoded in seconds, milliseconds, microseconds or
nanoseconds — pandas 2 wrote nanoseconds, pandas 3 defaults to microseconds, and
both turn up in real files. A column only becomes dates if its *name* suggests
time **and** every value falls inside one epoch band, so a row counter is never
silently converted.

---

## The plot

Pick columns with **Choose series** — a filter box over a checkbox list, because
these files routinely have a hundred or more columns. Up to 12 series at once.

Zooming re-requests the visible window rather than magnifying what's drawn, so
**zooming in shows real rows**: a 3,653-row series arrives min–max decimated at
~37 rows per point, and a zoom to 100 rows comes back undecimated.

| Action | |
|---|---|
| Drag across the plot | zoom to that span |
| Scroll wheel | zoom about the cursor |
| `+` / `−` | zoom about the centre |
| `‹` / `›` | pan by half a window |
| `Reset` | back to the whole series |

Decimation is min–max per bucket, so a one-row spike in a 50M-row series still
appears rather than being strided over.

---

## Security

The server binds to `127.0.0.1`, but that alone isn't protection — any website
you visit could otherwise reach it. So:

- every request needs the session token generated at launch;
- the `Host` header must resolve to loopback, which blocks DNS rebinding;
- the token is stripped from the address bar so it doesn't reach browser history.

Files are opened read-only and there is no write path.

---

## Things that deliberately don't happen

Each of these is a defect reported against a shipped HDF5 viewer, and each has a
test in `backend/tests/test_robustness.py`.

- **Nothing fails silently.** A bad export returns a visible error rather than
  the browser saving the error text as your "export".
- **Unresolvable links are shown**, greyed and labelled, not dropped — a node
  that vanishes is indistinguishable from one the model never wrote.
- **`node_2` sorts before `node_10`.** Model names are rarely zero-padded.
- **File locking is disabled before the HDF5 library loads**, so a file a model
  run is writing can still be read.
- **Compression plugins are registered**, so blosc-compressed PyTables output
  reads rather than failing with a missing-plugin error.
- **The drive list is read from your machine**, never hardcoded, and works on
  Windows, macOS and Linux.

### Limits

- **No sorting.** Sorting server-side would need a fully sorted index on the
  column, which pywr outputs don't have, and sorting only the loaded page would
  be misleading. Use the value search instead.
- **No editing**, by design.
- If a model rewrites the file while it's open, the API returns `409` and the UI
  offers a Reload.

---

## Development

From a clone (needs [Node.js](https://nodejs.org) to build the interface):

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
cd frontend && npm install && npm run build && cd ..
.venv/bin/h5grid path/to/file.h5
```

Working on it:

```bash
.venv/bin/python -m pytest backend/tests -q     # 248 tests; fixtures self-generate

.venv/bin/h5grid serve --no-token --port 8765   # terminal 1
cd frontend && npm run dev                      # terminal 2, proxies /api
```

`npm run build` writes the interface into `backend/h5grid/static/`, which is why
a `pip install` needs no Node at runtime.

### Layout

```
backend/h5grid/
  main.py       FastAPI app and endpoints
  cli.py        the `h5grid` command
  files.py      open-file registry, mtime guard, drive discovery
  tree.py       tree walker and pandas-node detection
  readers.py    RawDatasetReader / PandasTableReader / PandasFixedReader
  timeindex.py  /time tables and integer epoch decoding
  pywr.py       scenario axes, combinations, node metadata
  service.py    shared request handling for data, stats, plot, export
  stats.py      chunked column statistics and value search
  export.py     CSV and XLSX
  jsonsafe.py   NaN → null, ±Inf → strings, bytes → UTF-8
  security.py   session token and host guard
frontend/src/   React app (Glide Data Grid, TanStack Query, uPlot)
```

The design document is [h5-viewer-spec.md](h5-viewer-spec.md).

## Licence

MIT — see [LICENSE](LICENSE).
