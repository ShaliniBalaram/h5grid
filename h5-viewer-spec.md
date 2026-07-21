# H5Grid: A Lightweight HDF5 Viewer for Water Resource Model Files

**Version:** 0.2 draft
**Owner:** Shalini Balaram
**Audience:** frontend and backend developers building the MVP

---

## 1. Problem Statement

HDF5 (.h5) files are the standard container for time series inputs and outputs in
water resource models such as Pywr (via `TablesArrayParameter`, `TablesRecorder`,
and `pd.read_hdf` tables). Existing free viewers (HDFView, ViTables, myHDF5,
Panoply) fail on three counts:

1. They display pandas-format HDF5 stores as raw internal blocks
   (`axis0`, `block0_items`, `block0_values`) instead of reconstructing the
   table the modeller actually saved.
2. They do not render datetime indexes as human-readable dates.
3. They are clunky for large arrays. No smooth scrolling, no filtering,
   no copy-paste into Excel.

The goal is a small, fast, local tool where opening an H5 file feels like
opening a workbook in Excel: a file tree on the left, a scrollable grid in the
middle, metadata on the side, and one-click export.

---

## 2. Target Users and Core Use Cases

Primary user: a water resources modeller inspecting model inputs and outputs.

| # | Use case | Priority |
|---|----------|----------|
| U1 | Open an .h5 file and browse its group/dataset tree | Must |
| U2 | Click a dataset and see it as a spreadsheet-style grid | Must |
| U3 | Correctly decode pandas HDFStore tables (fixed and table format) | Must |
| U4 | Render datetime indexes and Pywr `/time` tables as real dates | Must |
| U5 | Handle multi-GB files without freezing (lazy slicing, paging) | Must |
| U6 | View attributes of any group or dataset | Must |
| U7 | Sort, filter, and search within the visible table | Must |
| U8 | Copy a selection to clipboard in Excel-compatible TSV | Must |
| U9 | Export any dataset or slice to CSV or XLSX | Should |
| U10 | Quick summary stats per column (min, max, mean, NaN count) | Should |
| U11 | Quick time series plot of selected columns | Should |
| U12 | Open two files side by side and diff datasets | Later |
| U13 | Edit cells and write back to the file | Later, off by default |

Editing is deliberately deferred. A read-only tool is safe, simple, and covers
90% of daily need. Corrupting a model input file is the worst possible outcome.

---

## 3. What the Tool Must Understand About the Files

This section is the domain knowledge the backend developer needs. Get this
right and the tool is already better than everything free.

### 3.1 Plain HDF5 (h5py-native)

- Hierarchy of **groups** (folders) and **datasets** (n-dimensional arrays).
- Datasets have a `shape`, `dtype`, optional `chunks`, and `attrs`.
- Dtypes to support: float32/64, int32/64, bool, fixed-length bytes,
  variable-length UTF-8 strings, and **compound dtypes** (record arrays,
  which render as multi-column tables).
- 1D dataset renders as a single column. 2D renders as rows x columns.
  3D and higher: let the user pick which 2D slice to view via index
  selectors for the extra dimensions (e.g. scenario axis in Pywr outputs).

### 3.2 Pandas HDFStore layout (critical)

Files written with `DataFrame.to_hdf(path, key)` contain per key either:

- **Fixed format:** a group with attribute `pandas_type = 'frame'` containing
  `axis0`, `axis1`, `block0_items`, `block0_values`, etc.
- **Table format:** a group with `pandas_type = 'frame_table'` containing a
  PyTables `table` dataset with an `index` column and value columns.

**Rule:** when the backend walks the tree, any group carrying a `pandas_type`
attribute is treated as a **logical table node**. Its internal children are
hidden from the tree by default (a "show raw structure" toggle reveals them).
Reading goes through `pandas.read_hdf` (fixed format loads whole object, see
size guard in 5.3) or `HDFStore.select(key, start, stop)` (table format,
which supports true row slicing).

### 3.3 Pywr-specific conventions

- `TablesArrayParameter` inputs: a 2D float dataset under some node path,
  shape `(timesteps, columns)`, with a sibling or root-level `/time` table.
- `TablesRecorder` outputs: one array per model node, shape
  `[len(timestepper)] + model.scenarios.shape` — **one axis per Scenario
  object**, so a model with two scenarios writes a 3D array, not a 2D one.
  Plus a `/time` table with columns `year, month, day` (and usually `index`).
- **`/scenarios` names those axes**: a table of `name` and `size`, in axis
  order. Decode it and the columns read `climate[0]` instead of `col_0`, and
  the slice selectors read "demand" instead of "dim 2". No other viewer does
  this, and without it every scenario axis is an anonymous integer.
- **`/scenario_combinations`** appears when the model used explicit
  combinations. The scenario axes are then collapsed into a single axis, and
  this table maps each column back to one index per scenario, so headers become
  `climate=2, demand=3`.
- **Per-array `PYWR_ATTRIBUTE`** (`flow`, `volume`, `parameter`,
  `parameter-index`) and **`PYWR_TYPE`** (the node's Python class). Older files
  spell these lower-case and hyphenated (`pywr-attribute`); support both.
- **Rule:** if a `/time` table exists at file root, offer a per-dataset toggle
  "use /time as row index" that renders dates as the first (frozen) column of
  the grid. Auto-enable it when the row count of the dataset matches the
  length of `/time`.
- Datetime detection elsewhere: int64 columns named like `time`, `date`,
  `index` holding an epoch should be decoded via `pd.to_datetime(values)`.
  Show the raw ints only if the user asks.
- **The epoch unit varies and must be inferred.** pandas 2 stored a
  DatetimeIndex as nanoseconds; pandas 3 defaults to microseconds, and other
  tools write seconds or milliseconds. All four appear in real files, so the
  unit is inferred from magnitude bands (which do not overlap for plausible
  model dates) rather than assumed.
- **Guard against false positives.** Require both a time-like column name and
  that *every* value sits inside one band. A row counter starts near zero, so
  its minimum falls below any band floor — which is what stops the `index`
  column of a pywr `/time` table being rendered as dates.

### 3.4 Scale assumptions

- Files up to ~10 GB, individual datasets up to ~50M rows.
- Backend must never call `dataset[...]` (full read) on anything above a
  size threshold. All reads are `dataset[start:stop, col_start:col_stop]`.
- HDF5 slicing is fast when aligned with chunking. Do not over-engineer:
  simple row-range slicing is fine for the MVP.

---

## 4. Architecture

Local-first desktop-style app. No cloud, no accounts, files never leave the
machine.

```
+---------------------------+          +------------------------------+
|  Frontend (React SPA)     |  HTTP    |  Backend (FastAPI, Python)   |
|  - Glide Data Grid table  | <------> |  - h5py / PyTables / pandas  |
|  - Tree view              |  JSON    |  - slicing, decoding, stats  |
|  - Plot panel (uPlot)     |          |  - CSV/XLSX export           |
+---------------------------+          +------------------------------+
                                             |
                                        local .h5 files
```

- **Why a web stack for a local tool:** free, cross-platform, and the grid
  problem is already solved by mature JS libraries. Ship it three ways with
  the same code: (a) `pip install h5grid && h5grid open file.h5` which starts
  the server and opens the browser, (b) later a Tauri wrapper for a real
  desktop app, (c) optionally hosted on an internal server for a team.
- Backend binds to `127.0.0.1` only by default.

### 4.1 Tech choices

| Layer | Choice | Why |
|-------|--------|-----|
| Backend | Python 3.11+, FastAPI, uvicorn | async, trivial JSON APIs, every hydrologist can read it |
| HDF5 | h5py + hdf5plugin (raw), PyTables + pandas (pandas stores) | h5py for slicing; hdf5plugin so h5py can decompress blosc/zstd chunks in PyTables-written files (Pywr outputs are often blosc-compressed); pandas for HDFStore decoding |
| Frontend | React 18 + TypeScript + Vite | standard, fast dev loop |
| Grid | Glide Data Grid (MIT) | virtual scrolling of millions of rows, built-in range selection and clipboard copy. NOT AG Grid Community: range selection, clipboard copy, and the selection status bar are AG Grid Enterprise-only, which would break U8 |
| Server state | TanStack Query | caching and request dedup for page fetches |
| Plot | uPlot | renders 1M points without breaking a sweat, ~40 kB |
| Export | pandas `to_csv` / `to_excel` (openpyxl) server-side | correct dtypes, streams to download |
| Packaging | pipx / pip entry point, later Tauri | zero-install friction first |

Alternative for a one-person prototype: the entire backend logic in a
Streamlit app in 1-2 days. Useful to validate the decoding rules in section 3
before the real frontend is built. The FastAPI service code is reusable either way.

---

## 5. Backend Specification

### 5.1 Process model

- One session = one or more open files, tracked by a `file_id` (hash of path
  + mtime). Files opened read-only (`h5py.File(path, 'r')`,
  `HDFStore(path, 'r')`). Handles cached in an LRU dict, closed on idle
  timeout (10 min) and reopened transparently.
- If the file changed on disk (mtime mismatch), return HTTP 409 with
  `{"error": "file_changed"}` so the frontend can prompt a reload.
- **Local API protection:** binding to 127.0.0.1 stops the network but not
  the user's own browser — any website they visit can fire requests at
  `http://127.0.0.1:<port>` and read local files through this API. On
  startup the server generates a random session token; the CLI opens the
  browser at `/?token=...`, the frontend sends it on every request
  (`X-H5Grid-Token` header), and the server rejects requests with a missing
  token or an unexpected `Host`/`Origin` header (DNS-rebinding guard).

### 5.2 REST API

All responses JSON unless noted. Errors: `{"error": str, "detail": str}`.

```
GET  /api/browse?dir=/models
     -> {"dir", "parent", "entries": [{"name","is_dir","size_bytes","is_h5"}]}
     Minimal server-side file picker backing the "Open file" button —
     browsers cannot hand a web page a filesystem path, so the frontend
     browses via this endpoint and then calls /api/files/open.

POST /api/files/open        body: {"path": "C:/models/inputs.h5"}
                            -> {"file_id", "path", "size_bytes", "mtime"}

GET  /api/files/{fid}/tree
     -> nested tree. Node schema:
        {
          "name": "flows",
          "path": "/inputs/flows",
          "kind": "group" | "dataset" | "pandas_frame" | "pandas_table",
          "shape": [30000, 12] | null,
          "dtype": "float64" | "compound" | null,
          "nrows": 30000 | null,          # logical rows after decoding
          "children": [...]
        }
     pandas nodes report the DECODED shape (rows x named columns),
     and their raw children are omitted unless ?raw=true.

GET  /api/files/{fid}/node/meta?path=/inputs/flows
     -> {"path", "kind", "shape", "dtype", "chunks", "compression",
         "columns": [{"name","dtype","is_datetime"}],
         "attrs": {...},                  # values JSON-safe, numpy coerced
         "time_index_available": true}

GET  /api/files/{fid}/node/data?path=...&start=0&stop=1000
        &cols=0:20            # optional column window
        &slice=,,3            # for >2D: fixed indices of extra dims
        &use_time_index=true
     -> {
          "start": 0, "stop": 1000, "total_rows": 30000,
          "columns": ["date", "node_A", "node_B", ...],
          "rows": [["1975-01-01", 12.4, 8.1, ...], ...]
        }
     Hard cap: (stop-start) * ncols <= 200_000 cells per request.
     NaN -> null, Inf -> "Infinity" string, bytes -> utf-8 with
     replacement. Datetimes ISO-8601 strings.

GET  /api/files/{fid}/node/stats?path=...&col=node_A
     -> {"min","max","mean","std","nan_count","count"}
     Computed in row chunks of 1M, cached per (fid, path, col).

GET  /api/files/{fid}/node/export?path=...&format=csv|xlsx
        &start=&stop=&cols=
     -> streamed file download. XLSX refused above 1M rows with a
        clear error suggesting CSV.

GET  /api/files/{fid}/node/plotdata?path=...&cols=1,4&max_points=4000
     -> decimated {x: [...], series: [{name, y: [...]}]}
     Decimation: min-max downsampling per bucket so spikes survive.

POST /api/files/{fid}/close
```

### 5.3 Decoding rules (the heart of the backend)

Implement as a `NodeReader` abstraction with three concrete readers:

1. **RawDatasetReader (h5py):** slices `dset[start:stop]`. Compound dtype
   fields become columns. 1D becomes a single column named `value`.
2. **PandasTableReader (PyTables `frame_table`):** uses
   `store.select(key, start=start, stop=stop)`. True lazy row access.
3. **PandasFixedReader (`frame`, fixed format):** fixed format cannot be
   row-sliced by pandas. Guard: if the decoded frame is under 500 MB
   estimated, load once, cache the DataFrame, serve slices from memory.
   Above that, fall back to raw block display with a banner explaining why.

Reader selection happens once per node during tree build and is stored in the
node metadata. Unit-test all three against fixture files (see 8).

### 5.4 Non-goals for the backend MVP

No auth, no multi-user, no write path, no HDF5 repacking, no remote files
(S3 etc.). Each of these is a bolt-on later and a distraction now.

---

## 6. Frontend Specification

### 6.1 Layout

```
+------------------------------------------------------------------+
| Toolbar: [Open file] [file tabs]        [Export] [Plot] [Search]  |
+---------------+--------------------------------------+-----------+
| Tree panel    |  Data grid (Glide Data Grid)         | Inspector |
| (resizable)   |  - frozen index/date column          | - dtype   |
|               |  - virtual scroll via infinite       | - shape   |
| /inputs       |    row model -> /node/data pages     | - chunks  |
|   flows  [T]  |  - column sort + text/number filter  | - attrs   |
|   demand [T]  |  - range selection + Ctrl+C (TSV)    | - stats   |
| /time         |  - status bar: sum/avg/count of      |   button  |
|               |    current selection (like Excel)    |           |
+---------------+--------------------------------------+-----------+
```

- Tree icons distinguish group, raw dataset, and decoded pandas table.
  Badge shows shape, e.g. `flows  30000 x 12  f64`.
- Clicking a >2D dataset shows small dropdowns above the grid for the extra
  dimension indices ("dim 2: [0..19]"), defaulting to 0.
- Inspector shows attributes as a key-value table. Long values truncated with
  expand-on-click.
- Plot panel opens as a bottom drawer. User selects columns via checkboxes in
  the column headers or from a modal, gets an instant line chart with zoom.

### 6.2 Grid behaviour details

- **Virtualized paging:** Glide Data Grid requests cells on demand via its
  callback API; fetch in pages of 1000 rows, prefetch one page ahead,
  TanStack Query cache keyed by `(fid, path, page, cols, slice)`.
- **Sorting and filtering:** client-side sorting within loaded pages is
  NOT acceptable (misleading), and server-side sorting is deferred to Later
  entirely — PyTables index-sorts only work with a completely-sorted index
  (CSI) on the column, which Pywr outputs will not have. MVP: sorting
  disabled everywhere with a tooltip ("sorting arrives in a later version").
  Filtering in MVP = a "jump to row" box + a column value search that scans
  server-side in chunks and returns matching row numbers.
- **Copy:** Glide Data Grid range selection, serialized as TSV so paste into
  Excel is native.
- **Formatting:** floats default to 6 significant digits, toggle for full
  precision. NaN shown as empty cell in light red. Dates as `YYYY-MM-DD`
  (with time part only if nonzero anywhere in the loaded page).
- Keyboard: arrows, PgUp/PgDn, Ctrl+Home/End, Ctrl+C. No editing keys.

### 6.3 States to design

Empty (no file), loading tree, loading page (skeleton rows), decode-fallback
banner (fixed-format too large), file-changed-on-disk banner with Reload
button, backend-down error.

---

## 7. MVP Cut and Milestones

**Milestone 1, backend core (1 week):** open/tree/meta/data endpoints,
three readers, fixture-file test suite. Definition of done: `curl` can page
through a 5 GB Pywr recorder file at <200 ms per page.

**Milestone 2, minimal UI (1 week):** tree + grid + inspector, copy to
clipboard, `/time` index toggle. Definition of done: a modeller can answer
"what is the flow at node X on 1976-03-15" in under 10 seconds from double-
clicking the file.

**Milestone 3, comfort (1 week):** export CSV/XLSX, column stats, plot
drawer, value search, packaging as `pipx install h5grid`.

**Later:** file diff view, Tauri desktop wrapper, opt-in editing, NetCDF
(.nc) support via the same reader abstraction (xarray), which would extend
the tool to climate model outputs.

**Editing scope (when it comes):** raw h5py datasets only — HDF5 supports
in-place cell writes (`dset[i, j] = value`) without rewriting the file, so
with an explicit "enable editing" toggle plus an automatic backup copy this
is low-risk. Pandas-format stores stay read-only indefinitely: fixed format
cannot be modified in place (the whole frame must be rewritten), and a bug
there corrupts the exact model input files this tool exists to protect.

---

## 8. Test Fixtures (backend dev: build these first)

A `make_fixtures.py` script generating:

1. `plain.h5`: 1D, 2D, 3D float datasets, compound dtype, vlen strings,
   nested groups, attributes of every scalar type.
2. `pandas_fixed.h5`: `df.to_hdf(key='monthly', format='fixed')` with a
   DatetimeIndex.
3. `pandas_table.h5`: same data, `format='table'`.
4. `pywr_style.h5`: root `/time` table (year/month/day) + several
   `(timesteps, scenarios)` float arrays, mimicking `TablesRecorder` output.
4b. `pywr_scenarios.h5`: two named Scenarios, so node arrays are genuinely 3D
   `(timesteps, climate, demand)`, with `PYWR_ATTRIBUTE`/`PYWR_TYPE` on each
   array and one node using the legacy lower-case attribute spelling.
4c. `pywr_combinations.h5`: explicit scenario combinations, so arrays collapse
   to `(timesteps, N)` and `/scenario_combinations` names the columns.
5. `big.h5`: a 20M x 10 chunked float dataset for performance tests
   (generated on demand, gitignored).

Every decoding rule in 5.3 gets a test against these files. This suite is the
project's real specification.

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Fixed-format pandas stores can't be lazily sliced | 500 MB guard + raw fallback with explanatory banner (5.3) |
| HDF5 C library is not thread-safe by default | serialize file access per file_id with an asyncio lock, do reads in a thread pool |
| Users open files while a model run is writing them | mtime check + 409 + Reload prompt (5.1) |
| Grid library outgrown (needs beyond Glide Data Grid) | wrap the grid in one thin component so the library can be swapped; keep a written list of missing-feature pain points |
| Dtype edge cases (int dates, byte strings) | fixtures + a "show raw" escape hatch on every node |

---

## 10. Repository Skeleton

```
h5grid/
  backend/
    h5grid/
      main.py            # FastAPI app, static file serving
      files.py           # open/close, handle cache, mtime guard
      tree.py            # walker + pandas-node detection
      readers.py         # RawDatasetReader, PandasTableReader, PandasFixedReader
      timeindex.py       # /time table + epoch-int datetime decoding
      export.py
      stats.py
    tests/
      make_fixtures.py
      test_readers.py
      test_tree.py
  frontend/
    src/
      components/ (Tree, Grid, Inspector, PlotDrawer, Toolbar)
      api/ (typed client, TanStack Query hooks)
  pyproject.toml         # entry point: h5grid = backend.h5grid.cli:main
```

`h5grid open path/to/file.h5` starts uvicorn on a free localhost port,
serves the built frontend as static files, and opens the default browser.
One installable artifact, no Node required at runtime.

---

## 11. Implementation Status

Milestones 1–3 are built against this document. See README.md to run it.

**Done:** all Must-have use cases (U1–U8) plus export (U9), column statistics
(U10) and the plot drawer (U11). Backend readers, tree walker, time-index
decoding, full pywr scenario decoding (3.3), stats, search, CSV/XLSX export,
session-token and host guards, the React UI, and a 238-test suite over the
section 8 fixtures.

**Deliberately not done:** sorting (U7 partial). Server-side sorting needs a
completely sorted index on the column, which pywr outputs do not have, and
sorting only the loaded page would be misleading — so the MVP ships jump-to-row
and a server-side value search instead, and sorting moves to Later. Editing
(U13) and the diff view (U12) remain Later as planned.

**Found during implementation, worth knowing:**

- `hdf5plugin` must be *imported*, not merely installed. It registers the blosc
  filter with the HDF5 library; without the import, every read of a
  PyTables-written file fails with a misleading "can't open directory .../plugin".
- h5py reports `compression = None` for blosc. Reading the dataset creation
  property list directly is the only way to report it honestly.
- Fixed-format storers expose `.shape` without touching the data, so the 500 MB
  guard in 5.3 can run *before* anything is decoded rather than after.
- Column statistics combine across chunks with Chan's parallel update. Summing
  squares loses most significant digits on a long column of near-identical
  reservoir levels and can even yield a negative variance.
