"""The three readers that turn an HDF5 node into a table of rows and columns.

Every reader returns a plain `pd.DataFrame` for a row range, with a RangeIndex
and one column per `self.columns` entry. Keeping that contract uniform is what
lets the data, stats, export and plot endpoints share one code path.

  RawDatasetReader    h5py slicing. 1D, 2D, ND-with-a-slice, and compound dtypes.
  PandasTableReader   HDFStore.select(start, stop). Genuinely lazy.
  PandasFixedReader   fixed format cannot be row-sliced; load once under a size
                      guard, cache, and serve slices from memory.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd

from . import pywr, timeindex

# Fixed-format frames above this estimated in-memory size are not decoded; the
# node falls back to showing its raw pandas blocks instead.
FIXED_FORMAT_SIZE_LIMIT_BYTES = 500 * 1024 * 1024

# Attribute names that, by convention, carry column labels for a 2D array.
COLUMN_NAME_ATTRS = ("column_names", "columns", "col_names", "labels", "names")

_HDF5_FILTER_NAMES = {
    1: "gzip",
    2: "shuffle",
    3: "fletcher32",
    4: "szip",
    5: "nbit",
    6: "scaleoffset",
    32000: "lzf",
    32001: "blosc",
    32004: "lz4",
    32008: "bitshuffle",
    32013: "zfp",
    32015: "zstd",
    32026: "blosc2",
}


@dataclass
class ColumnSpec:
    name: str
    dtype: str
    is_datetime: bool = False
    epoch_unit: str | None = None  # set when the raw values are an integer epoch

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "is_datetime": self.is_datetime}


@dataclass
class ColumnSelection:
    """A parsed `cols=` parameter: either a contiguous window or explicit list."""

    indices: list[int] | None = None  # None means "all columns"

    def apply(self, all_names: Sequence[str]) -> list[int]:
        if self.indices is None:
            return list(range(len(all_names)))
        return [i for i in self.indices if 0 <= i < len(all_names)]


def parse_cols(spec: str | None) -> ColumnSelection:
    """Parse `cols`, accepting a window ("0:20") or a list ("1,4,9")."""
    if spec is None or spec.strip() == "":
        return ColumnSelection()

    text = spec.strip()
    if ":" in text:
        start_s, _, stop_s = text.partition(":")
        start = int(start_s) if start_s.strip() else 0
        stop = int(stop_s) if stop_s.strip() else None
        if stop is None:
            raise ValueError("open-ended column windows need an upper bound, e.g. 0:20")
        if stop < start:
            raise ValueError("column window stop must be >= start")
        return ColumnSelection(list(range(max(0, start), stop)))

    indices = [int(part) for part in re.split(r"[,\s]+", text) if part]
    return ColumnSelection(indices)


def parse_dim_slice(spec: str | None, ndim: int) -> list[int | None]:
    """Parse `slice=,,3` into per-dimension fixed indices.

    An empty entry means the dimension stays free. Dimension 0 is always rows.
    For 2D and above exactly one further dimension may stay free (the columns);
    when the caller leaves several free, the first is taken and the rest pinned
    to 0, which matches the UI defaulting extra selectors to index 0.
    """
    free_needed = 1 if ndim <= 1 else 2

    if spec is None or spec.strip() == "":
        result: list[int | None] = [None] * ndim
        for dim in range(free_needed, ndim):
            result[dim] = 0
        return result

    parts = [p.strip() for p in spec.split(",")]
    if len(parts) < ndim:
        parts += [""] * (ndim - len(parts))
    elif len(parts) > ndim:
        raise ValueError(f"slice has {len(parts)} entries but dataset has {ndim} dims")

    result = [None if p == "" else int(p) for p in parts]
    if result[0] is not None:
        raise ValueError("dimension 0 is the row axis and cannot be pinned")

    free = [i for i, v in enumerate(result) if v is None]
    if len(free) > free_needed:
        for dim in free[free_needed:]:
            result[dim] = 0
    elif len(free) < free_needed:
        raise ValueError(
            f"{free_needed} free dimensions required, slice leaves {len(free)}"
        )
    return result


def describe_compression(dset: h5py.Dataset) -> str | None:
    """Human-readable filter chain, including filters h5py does not name.

    `dset.compression` returns None for blosc and friends, which is exactly what
    PyTables (and therefore pywr) writes, so we read the creation property list
    directly rather than reporting "no compression" on a compressed file.
    """
    try:
        plist = dset.id.get_create_plist()
        names = []
        for i in range(plist.get_nfilters()):
            filter_id = plist.get_filter(i)[0]
            names.append(_HDF5_FILTER_NAMES.get(filter_id, f"filter_{filter_id}"))
        if names:
            return ", ".join(names)
    except Exception:
        pass
    return dset.compression


class NodeReader(ABC):
    """Common interface over every kind of readable node."""

    kind: str = "dataset"
    path: str = "/"
    nrows: int = 0
    columns: list[ColumnSpec] = []
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    chunks: tuple[int, ...] | None = None
    compression: str | None = None
    supports_row_slicing: bool = True
    decode_fallback: str | None = None
    ndim: int = 2

    @abstractmethod
    def read(
        self,
        start: int,
        stop: int,
        cols: ColumnSelection | None = None,
        dim_slice: list[int | None] | None = None,
    ) -> pd.DataFrame:
        """Return rows [start, stop) as a DataFrame with a RangeIndex."""

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


class RawDatasetReader(NodeReader):
    """h5py-native datasets: plain arrays of any rank, and compound dtypes."""

    kind = "dataset"

    def __init__(
        self,
        dset: h5py.Dataset,
        path: str,
        scenario_layout: "pywr.ScenarioLayout | None" = None,
    ) -> None:
        self._dset = dset
        self.path = path
        self._layout = scenario_layout
        self.pywr = pywr.node_metadata(dset)
        self.shape = tuple(int(s) for s in dset.shape)
        self.ndim = len(self.shape)
        self.chunks = tuple(int(c) for c in dset.chunks) if dset.chunks else None
        self.compression = describe_compression(dset)
        self.nrows = int(self.shape[0]) if self.shape else 1

        self._fields = dset.dtype.names
        self._is_compound = self._fields is not None
        self.dtype = "compound" if self._is_compound else str(dset.dtype)
        self.columns = self._build_columns()

    def _build_columns(self) -> list[ColumnSpec]:
        if self._is_compound:
            specs = []
            for name in self._fields:
                sub = self._dset.dtype[name]
                unit = timeindex.detect_epoch_unit(self._sample_field(name), name)
                specs.append(
                    ColumnSpec(
                        name=name,
                        dtype="datetime64[ns]" if unit else str(sub),
                        is_datetime=unit is not None,
                        epoch_unit=unit,
                    )
                )
            return specs

        if self.ndim <= 1:
            values = self._sample_flat()
            unit = timeindex.detect_epoch_unit(values, _leaf_name(self.path))
            return [
                ColumnSpec(
                    "value",
                    "datetime64[ns]" if unit else str(self._dset.dtype),
                    unit is not None,
                    epoch_unit=unit,
                )
            ]

        ncols = self._default_ncols()
        return [
            ColumnSpec(n, str(self._dset.dtype), False)
            for n in self._labels_for_dim(1, ncols)
        ]

    def _default_ncols(self) -> int:
        """Column count for the default view (dim 1 free, later dims pinned)."""
        return int(self.shape[1]) if self.ndim >= 2 else 1

    def _labels_for_dim(self, col_dim: int, ncols: int) -> list[str]:
        """Headers for whichever dimension is currently the column axis.

        A pywr scenario name beats a `column_names` attribute, which beats a
        positional fallback — so a recorder array reads `climate[0] … climate[2]`
        instead of `col_0 … col_2`.
        """
        if self._layout is not None and self._layout.describes(self.shape):
            labels = self._layout.labels_for_dim(col_dim, ncols)
            if labels is not None:
                return labels
        return self._labels_from_attrs(ncols) or [f"col_{i}" for i in range(ncols)]

    def dimension_names(self) -> list[str | None]:
        """A name per dataset dimension, for labelling the slice selectors."""
        names: list[str | None] = [None] * self.ndim
        if self._layout is not None and self._layout.describes(self.shape):
            for dim in range(1, self.ndim):
                names[dim] = self._layout.axis_name(dim)
        return names

    def _labels_from_attrs(self, ncols: int) -> list[str] | None:
        for attr in COLUMN_NAME_ATTRS:
            if attr not in self._dset.attrs:
                continue
            raw = self._dset.attrs[attr]
            try:
                values = list(np.atleast_1d(raw))
            except Exception:
                continue
            if len(values) != ncols:
                continue
            return [
                v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
                for v in values
            ]
        return None

    def _sample_field(self, name: str) -> np.ndarray:
        n = min(self.nrows, 1000)
        if n == 0:
            return np.empty(0, dtype="i8")
        return self._dset[0:n][name]

    def _sample_flat(self) -> np.ndarray:
        n = min(self.nrows, 1000)
        if n == 0:
            return np.empty(0, dtype="i8")
        return np.asarray(self._dset[0:n]).ravel()

    def read(
        self,
        start: int,
        stop: int,
        cols: ColumnSelection | None = None,
        dim_slice: list[int | None] | None = None,
    ) -> pd.DataFrame:
        start = max(0, start)
        stop = min(self.nrows, stop)
        if stop <= start:
            return pd.DataFrame({c.name: [] for c in self.columns})

        if self._is_compound:
            return self._read_compound(start, stop, cols)
        if self.ndim <= 1:
            return self._read_1d(start, stop)
        return self._read_nd(start, stop, cols, dim_slice)

    def _read_compound(
        self, start: int, stop: int, cols: ColumnSelection | None
    ) -> pd.DataFrame:
        block = self._dset[start:stop]
        wanted = (cols or ColumnSelection()).apply(self.column_names())
        data = {}
        for i in wanted:
            spec = self.columns[i]
            values = block[spec.name]
            if spec.is_datetime:
                values = timeindex.decode_epoch(values, spec.epoch_unit or "ns")
            elif values.dtype.kind == "S":
                values = np.char.decode(values, "utf-8", "replace")
            data[spec.name] = values
        return pd.DataFrame(data)

    def _read_1d(self, start: int, stop: int) -> pd.DataFrame:
        values = np.asarray(self._dset[start:stop]).ravel()
        spec = self.columns[0]
        if spec.is_datetime:
            values = timeindex.decode_epoch(values, spec.epoch_unit or "ns")
        elif values.dtype.kind == "S":
            values = np.char.decode(values, "utf-8", "replace")
        elif values.dtype.kind == "O":
            values = np.array(
                [v.decode("utf-8", "replace") if isinstance(v, bytes) else v for v in values],
                dtype=object,
            )
        return pd.DataFrame({spec.name: values})

    def _read_nd(
        self,
        start: int,
        stop: int,
        cols: ColumnSelection | None,
        dim_slice: list[int | None] | None,
    ) -> pd.DataFrame:
        pinned = dim_slice or parse_dim_slice(None, self.ndim)
        free = [i for i, v in enumerate(pinned) if v is None]
        col_dim = free[1]
        total_cols = int(self.shape[col_dim])

        wanted = (cols or ColumnSelection()).apply([""] * total_cols)
        if not wanted:
            wanted = list(range(total_cols))

        # One contiguous HDF5 read covering the requested columns, then take the
        # offsets we need. Contiguous beats fancy indexing for chunked reads.
        col_lo, col_hi = min(wanted), max(wanted) + 1

        selector: list[Any] = [0] * self.ndim
        for dim in range(self.ndim):
            if dim == 0:
                selector[dim] = slice(start, stop)
            elif dim == col_dim:
                selector[dim] = slice(col_lo, col_hi)
            else:
                selector[dim] = int(pinned[dim] or 0)

        block = np.asarray(self._dset[tuple(selector)])
        if block.ndim == 1:  # every non-row dim collapsed to a scalar index
            block = block.reshape(-1, 1)

        offsets = [c - col_lo for c in wanted]
        block = block[:, offsets]

        names = self.column_names()
        labels = [
            names[c] if c < len(names) else f"col_{c}" for c in wanted
        ]
        if block.dtype.kind == "S":
            block = np.char.decode(block, "utf-8", "replace")
        return pd.DataFrame(block, columns=labels)

    def columns_for_slice(self, dim_slice: list[int | None] | None) -> list[ColumnSpec]:
        """Column specs for a non-default slice, where the free dim may differ."""
        if self._is_compound or self.ndim <= 1 or dim_slice is None:
            return self.columns
        free = [i for i, v in enumerate(dim_slice) if v is None]
        col_dim = free[1]
        if col_dim == 1:
            return self.columns
        ncols = int(self.shape[col_dim])
        return [
            ColumnSpec(name, str(self._dset.dtype), False)
            for name in self._labels_for_dim(col_dim, ncols)
        ]


class PandasTableReader(NodeReader):
    """PyTables `frame_table` nodes. Supports true lazy row slicing."""

    kind = "pandas_table"

    def __init__(self, store: pd.HDFStore, key: str) -> None:
        self._store = store
        self.path = key if key.startswith("/") else "/" + key
        storer = store.get_storer(key)
        self.nrows = int(storer.nrows)
        self.chunks = None
        self.compression = _storer_compression(storer)

        head = store.select(key, start=0, stop=1)
        self._index_name = head.index.name or "index"
        self.columns = [
            ColumnSpec(
                self._index_name,
                str(head.index.dtype),
                bool(pd.api.types.is_datetime64_any_dtype(head.index)),
            )
        ] + [
            ColumnSpec(
                str(name),
                str(head[name].dtype),
                bool(pd.api.types.is_datetime64_any_dtype(head[name])),
            )
            for name in head.columns
        ]
        self.shape = (self.nrows, len(self.columns) - 1)
        self.dtype = "frame_table"
        self.ndim = 2

    def read(
        self,
        start: int,
        stop: int,
        cols: ColumnSelection | None = None,
        dim_slice: list[int | None] | None = None,
    ) -> pd.DataFrame:
        start = max(0, start)
        stop = min(self.nrows, stop)
        if stop <= start:
            return pd.DataFrame({c.name: [] for c in self.columns})

        frame = self._store.select(self.path, start=start, stop=stop)
        frame = frame.reset_index()
        if frame.columns[0] != self._index_name:
            frame = frame.rename(columns={frame.columns[0]: self._index_name})

        wanted = (cols or ColumnSelection()).apply(self.column_names())
        names = self.column_names()
        return frame[[names[i] for i in wanted]]


class PandasFixedReader(NodeReader):
    """Fixed-format `frame` nodes: whole-object read, guarded by size."""

    kind = "pandas_frame"

    def __init__(
        self,
        store: pd.HDFStore,
        key: str,
        frame_cache: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self._store = store
        self._cache = frame_cache if frame_cache is not None else {}
        self.path = key if key.startswith("/") else "/" + key
        self.supports_row_slicing = False
        self.dtype = "frame"
        self.ndim = 2

        storer = store.get_storer(key)
        self.compression = _storer_compression(storer)
        shape = getattr(storer, "shape", None)

        # Fixed-format storers report [nrows, ncols] without touching the data,
        # so the size guard runs before anything large is read.
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            self.nrows, ncols = int(shape[0]), int(shape[1])
        else:
            self.nrows, ncols = 0, 0

        estimated = self.nrows * max(ncols, 1) * 8
        if estimated > FIXED_FORMAT_SIZE_LIMIT_BYTES:
            self.decode_fallback = (
                f"This fixed-format table is about {estimated / 1e9:.1f} GB decoded, "
                f"over the {FIXED_FORMAT_SIZE_LIMIT_BYTES / 1e6:.0f} MB limit. Fixed "
                "format cannot be read row by row, so it is shown as raw pandas "
                "blocks. Re-save it with format='table' to browse it directly."
            )
            self.columns = []
            self.shape = (self.nrows, ncols)
            return

        frame = self._frame()
        self.nrows = len(frame)
        self.shape = (self.nrows, frame.shape[1])
        self._index_name = frame.index.name or "index"
        self.columns = [
            ColumnSpec(
                self._index_name,
                str(frame.index.dtype),
                bool(pd.api.types.is_datetime64_any_dtype(frame.index)),
            )
        ] + [
            ColumnSpec(
                str(name),
                str(frame[name].dtype),
                bool(pd.api.types.is_datetime64_any_dtype(frame[name])),
            )
            for name in frame.columns
        ]

    def _frame(self) -> pd.DataFrame:
        cached = self._cache.get(self.path)
        if cached is None:
            cached = self._store.get(self.path)
            self._cache[self.path] = cached
        return cached

    def read(
        self,
        start: int,
        stop: int,
        cols: ColumnSelection | None = None,
        dim_slice: list[int | None] | None = None,
    ) -> pd.DataFrame:
        if self.decode_fallback:
            raise ValueError(self.decode_fallback)

        start = max(0, start)
        stop = min(self.nrows, stop)
        if stop <= start:
            return pd.DataFrame({c.name: [] for c in self.columns})

        frame = self._frame().iloc[start:stop].reset_index()
        if frame.columns[0] != self._index_name:
            frame = frame.rename(columns={frame.columns[0]: self._index_name})

        wanted = (cols or ColumnSelection()).apply(self.column_names())
        names = self.column_names()
        return frame[[names[i] for i in wanted]]


def _storer_compression(storer: Any) -> str | None:
    try:
        filters = storer.group._v_filters
        if filters is not None and filters.complevel:
            return f"{filters.complib} (level {filters.complevel})"
    except Exception:
        pass
    return None


def _leaf_name(path: str) -> str:
    return path.rstrip("/").rsplit("/", 1)[-1]
