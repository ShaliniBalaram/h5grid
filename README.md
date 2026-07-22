# H5Grid

[![PyPI](https://img.shields.io/pypi/v/h5grid)](https://pypi.org/project/h5grid/)
[![CI](https://github.com/ShaliniBalaram/h5grid/actions/workflows/ci.yml/badge.svg)](https://github.com/ShaliniBalaram/h5grid/actions/workflows/ci.yml)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/h5grid/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A lightweight local HDF5 viewer for water resource model files.

Opening an `.h5` file should feel like opening a workbook: a tree on the left, a
scrollable grid in the middle, metadata on the right, one click to export.

What makes it different from HDFView, ViTables, myHDF5 and Panoply:

- **Pandas stores are decoded, not dumped.** A group written by
  `DataFrame.to_hdf` shows up as the table the modeller saved — real column
  names, real dtypes — instead of `axis0`, `block0_items`, `block0_values`.
- **Dates are dates.** A `DatetimeIndex` renders as `1975-01-01`, not as a
  19-digit integer. A pywr `TablesRecorder` output picks up the file's `/time`
  table as a frozen first column.
- **Large files stay responsive.** Nothing is ever read whole: every request is
  a bounded row slice, capped at 200,000 cells.

The full design document is [h5-viewer-spec.md](h5-viewer-spec.md).

---

## Install and run

Requires **Python 3.11–3.13**. The published package bundles the web frontend,
so no Node is needed:

```bash
pip install h5grid
h5grid path/to/model_outputs.h5
```

That starts a server on localhost, prints a URL carrying a one-off session
token, and opens your browser. `h5grid serve` starts it with no file loaded.

Flags: `--port`, `--host`, `--no-browser`, `--no-token`.

To install from source instead (needs Node to build the frontend), see
[Development](#development) below.

## Example files

```bash
.venv/bin/python examples/make_examples.py
.venv/bin/h5grid examples/example_3d_pywr.h5
```

| File | Shape | What it shows |
|---|---|---|
| `example_3d_pywr.h5` | `(3653, 5, 4)` | A ten-year run over 5 climate × 4 demand scenarios. Columns are named from `/scenarios`, the extra axis gets a **demand** selector, dates come from `/time`, and nodes are tagged `flow` / `volume` / `idx`. |
| `example_3d_plain.h5` | `(365, 24, 12)` | Hourly demand by zone, with no naming metadata at all — the generic path, where the selectors fall back to "dim 2". |
| `example_4d.h5` | `(500, 8, 6, 4)` | An ensemble forecast: two dimensions beyond rows and columns, so two selectors appear. |

Open a 3D dataset and the toolbar gains a selector for each dimension past the
first two. Dimension 0 is always the rows; the next free one becomes the
columns; the rest are pinned to a single index you choose.

## Development

From a clone (needs Node, since the frontend must be built):

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
cd frontend && npm install && npm run build && cd ..   # builds into backend/h5grid/static/
.venv/bin/h5grid path/to/model_outputs.h5
```

Working on it:

```bash
.venv/bin/python backend/tests/make_fixtures.py   # writes backend/tests/fixtures/
.venv/bin/python -m pytest backend/tests -q

.venv/bin/h5grid serve --no-token --port 8765     # terminal 1
cd frontend && npm run dev                        # terminal 2, proxies /api
```

`npm run build` writes the SPA into `backend/h5grid/static/`, so a `pip install`
ships one artifact and needs no Node at runtime.

---

## What it understands

| Layout | How it is read |
|---|---|
| Plain h5py datasets | `dset[start:stop]`. 1D, 2D, ND with a slice selector, and compound dtypes as multi-column tables. |
| `pandas_type = 'frame_table'` | `HDFStore.select(key, start, stop)` — genuinely lazy row access. |
| `pandas_type = 'frame'` (fixed) | Cannot be row-sliced. Decoded once under a 500 MB guard and cached; above that it falls back to raw blocks with a banner explaining why. |
| pywr `TablesRecorder` output | One array per node, shaped `[timesteps] + scenarios.shape` — one axis per Scenario, so often 3D. The root `/time` table becomes a date index whenever the row counts match. |

Any group carrying a `pandas_type` attribute is treated as one logical table and
its internals are hidden. The **Raw structure** toggle reveals them.

### pywr scenarios

`TablesRecorder` writes more structure than any other viewer reads, and H5Grid
uses all of it:

- **`/scenarios`** names the scenario axes, so columns read `climate[0]` rather
  than `col_0`, and the slice selectors say "demand" rather than "dim 2".
- **`/scenario_combinations`**, present when a model used explicit
  combinations, collapses the scenario axes into one — columns then read
  `climate=2, demand=3`.
- **`PYWR_ATTRIBUTE`** and **`PYWR_TYPE`** tag each node with what it records
  (`flow`, `volume`, `parameter`) and its model class, shown as a badge in the
  tree and in the inspector. Files written by older pywr spell these
  `pywr-attribute`/`pywr-type`; both are read.

Labels are only applied when a dataset's trailing dimensions actually match the
scenario sizes, so a summary array that happens to sit in the same file is left
with positional column names.

Integer epoch columns are decoded in seconds, milliseconds, microseconds or
nanoseconds — pandas 2 wrote nanoseconds, pandas 3 defaults to microseconds, and
both turn up in real files. A column is only treated as dates if its *name*
suggests time and *every* value falls inside one epoch band, so a row counter is
never silently converted.

`hdf5plugin` is a hard dependency, not an optional one: PyTables writes
blosc-compressed chunks by default, and plain h5py cannot decompress them.

---

## Security

The server binds to `127.0.0.1`, but that alone does not protect it — any
website you visit can issue requests to `http://127.0.0.1:<port>`. So:

- every `/api` request needs the session token generated at launch (sent as an
  `X-H5Grid-Token` header, or a query parameter for download links);
- the `Host` header must resolve to loopback, which blocks DNS rebinding;
- the frontend strips the token from the address bar on load so it does not
  reach browser history.

Files are opened read-only and there is no write path.

## Opening files

A browser cannot hand a web page a filesystem path, so **Open file…** is served
by the backend rather than by the OS dialog. It offers:

- **Shortcuts** for Home, Desktop, Documents, the working directory, the
  filesystem root, and every **mounted volume** — model data usually lives on an
  external or network drive, and reaching `/Volumes/…` by walking up from home
  is not a reasonable way to get there.
- **Breadcrumbs**, so any ancestor is one click away.
- **Back**, which returns to the previous folder — not the same as **Up** once
  you have jumped via a shortcut or breadcrumb.
- A path box that takes either a file or a folder.

## The plot

Pick columns with **Choose series** — a filter box over a checkbox list, because
these files routinely have a hundred or more columns. Up to 12 series at once.

Zooming re-requests the visible window rather than magnifying what is already
drawn, so **zooming in shows real rows**: a 3,653-row series arrives min–max
decimated at ~37 rows per point, and a zoom to 100 rows comes back undecimated.
The header says which you are looking at.

| Action | |
|---|---|
| Drag across the plot | zoom to that span |
| Scroll wheel | zoom about the cursor |
| `+` / `−` | zoom about the centre |
| `‹` / `›` | pan by half a window |
| `Reset` | back to the whole series |

Decimation is min–max per bucket, so a one-row spike in a 50M-row series still
appears rather than being strided over.

## Failure modes we deliberately guard against

Each of these is a defect reported against a shipped HDF5 viewer, and each has a
test in `backend/tests/test_robustness.py`.

- **Nothing fails silently.** A bad export returns an error the UI shows, rather
  than the browser saving the error body as the "export" file. An over-large
  request says so and gives the limit instead of hanging or truncating.
- **Links that cannot be resolved are shown as nodes**, greyed and labelled,
  not dropped. A node that vanishes from the tree is indistinguishable from one
  the model never wrote.
- **`node_2` sorts before `node_10`.** Model node names are rarely zero-padded.
- **File locking is disabled before the HDF5 library loads**, so a file a model
  run currently holds open can still be read. A test asserts the import order,
  because reordering imports undoes it silently.
- **Compression plugins are registered**, so blosc-compressed PyTables output
  reads rather than failing with a missing-plugin path error.
- **Read-only, always.** There is no write path, so the viewer cannot damage a
  results file.

## Notes and limits

- **Sorting is not implemented.** Sorting server-side would need a completely
  sorted index on the column, which pywr outputs do not have, and sorting only
  the loaded page would be misleading. Filtering is a server-side value search
  (`>100`, `<=0`, `=5`, `3..7`, or substring) that returns row numbers.
- **Editing is not implemented and is not planned for pandas stores.** See the
  spec's Editing scope note.
- `file_id` folds in the file's mtime, so once a model run rewrites the file the
  API answers `409` and the UI offers a Reload.

## Layout

```
backend/h5grid/
  main.py       FastAPI app and endpoints
  cli.py        `h5grid` entry point
  files.py      open-file registry, mtime guard, per-file locking
  tree.py       tree walker and pandas-node detection
  readers.py    RawDatasetReader / PandasTableReader / PandasFixedReader
  timeindex.py  /time tables and integer epoch decoding
  service.py    request handling shared by data, stats, plot and export
  stats.py      chunked column statistics and value search
  export.py     CSV and XLSX
  jsonsafe.py   NaN → null, ±Inf → strings, bytes → UTF-8
  security.py   session token and host guard
backend/tests/  make_fixtures.py plus the suite that runs against it
frontend/src/   React app (Glide Data Grid, TanStack Query, uPlot)
```
