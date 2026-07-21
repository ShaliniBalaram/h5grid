"""Composes readers, the time index, and JSON coercion into API payloads.

The endpoints in main.py stay thin: they parse query parameters and call in
here. Everything that knows about the shape of a response lives in this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import jsonsafe, tree
from .files import OpenFile
from .readers import ColumnSelection, NodeReader, parse_cols, parse_dim_slice

MAX_CELLS_PER_REQUEST = 200_000
DEFAULT_PAGE_ROWS = 1000
TIME_COLUMN_NAME = "date"


class RequestTooLargeError(Exception):
    pass


def _reader_columns(reader: NodeReader, dim_slice: list[int | None] | None):
    """Column specs, allowing for a non-default free dimension on ND arrays."""
    getter = getattr(reader, "columns_for_slice", None)
    if getter is not None:
        return getter(dim_slice)
    return reader.columns


def time_index_matches(open_file: OpenFile, reader: NodeReader) -> bool:
    index = open_file.time_index()
    return index is not None and len(index) == reader.nrows


def node_meta(
    open_file: OpenFile, path: str, dim_slice_spec: str | None = None
) -> dict[str, Any]:
    reader = open_file.reader(path)
    dim_slice = parse_dim_slice(dim_slice_spec, reader.ndim) if reader.ndim > 2 else None
    columns = _reader_columns(reader, dim_slice)

    h5 = open_file.h5()
    lookup = path.strip("/")
    obj = h5 if lookup == "" else h5[lookup]

    layout = open_file.scenario_layout()
    dimension_names = getattr(reader, "dimension_names", None)

    return {
        "path": reader.path,
        "kind": reader.kind,
        "shape": list(reader.shape) if reader.shape else None,
        "ndim": reader.ndim,
        "dtype": reader.dtype,
        "chunks": list(reader.chunks) if reader.chunks else None,
        "compression": reader.compression,
        "nrows": reader.nrows,
        "columns": [c.to_json() for c in columns],
        "attrs": jsonsafe.coerce_attrs(obj.attrs),
        "time_index_available": time_index_matches(open_file, reader),
        "supports_row_slicing": reader.supports_row_slicing,
        "decode_fallback": reader.decode_fallback,
        "dim_sizes": list(reader.shape) if reader.shape else None,
        # Names the scenario axes so the slice selectors can say "demand"
        # rather than "dim 2".
        "dim_names": dimension_names() if dimension_names else None,
        "pywr": getattr(reader, "pywr", None) or None,
        "scenarios": layout.to_json() if layout is not None else None,
    }


def node_data(
    open_file: OpenFile,
    path: str,
    *,
    start: int = 0,
    stop: int | None = None,
    cols_spec: str | None = None,
    dim_slice_spec: str | None = None,
    use_time_index: bool = False,
) -> dict[str, Any]:
    reader = open_file.reader(path)
    if reader.decode_fallback:
        raise ValueError(reader.decode_fallback)

    if stop is None:
        stop = start + DEFAULT_PAGE_ROWS
    start = max(0, start)
    stop = max(start, min(stop, reader.nrows))

    dim_slice = parse_dim_slice(dim_slice_spec, reader.ndim) if reader.ndim > 2 else None
    selection = parse_cols(cols_spec)
    all_columns = _reader_columns(reader, dim_slice)
    wanted = selection.apply([c.name for c in all_columns])

    cells = (stop - start) * max(len(wanted), 1)
    if cells > MAX_CELLS_PER_REQUEST:
        raise RequestTooLargeError(
            f"Requested {cells:,} cells, over the {MAX_CELLS_PER_REQUEST:,} limit. "
            "Narrow the row range or the column window."
        )

    frame = reader.read(start, stop, selection, dim_slice)
    columns = [all_columns[i].name for i in wanted] if wanted else list(frame.columns)

    date_column: list[Any] | None = None
    if use_time_index and time_index_matches(open_file, reader):
        index = open_file.time_index()
        dates = pd.Series(index[start:stop])
        date_column = jsonsafe._coerce_column(dates)

    rows = jsonsafe.frame_to_rows(frame)
    if date_column is not None:
        columns = [TIME_COLUMN_NAME] + columns
        rows = [[date, *row] for date, row in zip(date_column, rows)]

    return {
        "path": reader.path,
        "start": start,
        "stop": stop,
        "total_rows": reader.nrows,
        "columns": columns,
        "column_types": _column_types(all_columns, wanted, date_column is not None),
        "rows": rows,
        "time_index_applied": date_column is not None,
    }


def _column_types(
    all_columns, wanted: list[int], with_date: bool
) -> list[dict[str, Any]]:
    specs = [all_columns[i].to_json() for i in wanted]
    if with_date:
        specs.insert(
            0, {"name": TIME_COLUMN_NAME, "dtype": "datetime64[ns]", "is_datetime": True}
        )
    return specs


def plot_data(
    open_file: OpenFile,
    path: str,
    *,
    cols_spec: str | None,
    max_points: int = 4000,
    dim_slice_spec: str | None = None,
    use_time_index: bool = True,
    start: int = 0,
    stop: int | None = None,
) -> dict[str, Any]:
    """Decimated series for the plot drawer.

    Min-max downsampling: each bucket contributes its minimum and its maximum,
    placed at the x positions where they occurred, so a one-row spike in a
    50M-row series still shows up. Plain striding would drop it.
    """
    reader = open_file.reader(path)
    if reader.decode_fallback:
        raise ValueError(reader.decode_fallback)

    dim_slice = parse_dim_slice(dim_slice_spec, reader.ndim) if reader.ndim > 2 else None
    all_columns = _reader_columns(reader, dim_slice)
    selection = parse_cols(cols_spec)
    wanted = selection.apply([c.name for c in all_columns])
    if not wanted:
        wanted = [i for i, c in enumerate(all_columns) if not c.is_datetime][:4]
    if not wanted:
        raise ValueError("No plottable columns selected.")

    start = max(0, start)
    stop = reader.nrows if stop is None else min(stop, reader.nrows)
    total = max(0, stop - start)
    if total == 0:
        return {
            "x": [],
            "rows": [],
            "series": [],
            "x_is_date": False,
            "decimated": False,
            "start": start,
            "stop": stop,
            "window_rows": 0,
            "total_rows": reader.nrows,
        }

    max_points = max(20, min(max_points, 20_000))
    buckets = max(1, max_points // 2)
    # Round the bucket size up, so the bucket count never exceeds the budget and
    # the response stays within max_points.
    bucket_size = max(1, -(-total // buckets))

    time_index = open_file.time_index() if use_time_index else None
    x_is_date = time_index is not None and len(time_index) == reader.nrows

    # A pandas frame carries its dates in its own index column rather than in a
    # /time table, so fall back to that — otherwise the grid shows dates while
    # the plot shows row numbers for the very same rows.
    index_column: int | None = None
    if not x_is_date and all_columns and all_columns[0].is_datetime:
        index_column = 0
        x_is_date = True

    index_values: list[Any] = []
    x_positions: list[int] = []
    series_values: dict[int, list[float | None]] = {i: [] for i in wanted}

    chunk_rows = max(bucket_size, 200_000 // max(len(wanted), 1))
    chunk_rows = (chunk_rows // bucket_size) * bucket_size or bucket_size

    column_selection = ColumnSelection(wanted)
    names = [c.name for c in all_columns]

    for chunk_start in range(start, stop, chunk_rows):
        chunk_stop = min(chunk_start + chunk_rows, stop)
        frame = reader.read(chunk_start, chunk_stop, column_selection, dim_slice)
        if frame.empty:
            continue

        arrays = {}
        for position, col_index in enumerate(wanted):
            column_name = names[col_index]
            series = (
                frame[column_name]
                if column_name in frame.columns
                else frame.iloc[:, position]
            )
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
                series
            ):
                arrays[col_index] = series.to_numpy().astype("float64", copy=False)
            else:
                arrays[col_index] = np.full(len(series), np.nan)

        dates_in_chunk = None
        if index_column is not None:
            dates_in_chunk = reader.read(
                chunk_start, chunk_stop, ColumnSelection([index_column]), dim_slice
            ).iloc[:, 0]

        for offset in range(0, chunk_stop - chunk_start, bucket_size):
            end = min(offset + bucket_size, chunk_stop - chunk_start)
            if end <= offset:
                continue
            lo_row = chunk_start + offset
            hi_row = chunk_start + end - 1
            x_positions.extend([lo_row, hi_row])
            if dates_in_chunk is not None:
                index_values.extend(
                    [dates_in_chunk.iloc[offset], dates_in_chunk.iloc[end - 1]]
                )

            for col_index in wanted:
                window = arrays[col_index][offset:end]
                target = series_values[col_index]
                if window.size == 0 or np.all(np.isnan(window)):
                    target.extend([None, None])
                    continue
                i_min = int(np.nanargmin(window))
                i_max = int(np.nanargmax(window))
                v_min = float(window[i_min])
                v_max = float(window[i_max])
                # Emit in the order the two extremes actually occurred so the
                # line does not zigzag backwards through time.
                if i_min <= i_max:
                    target.extend([v_min, v_max])
                else:
                    target.extend([v_max, v_min])

    if index_column is not None:
        x_values = [jsonsafe.coerce_scalar(v) for v in index_values]
    elif x_is_date and time_index is not None:
        x_values = [jsonsafe.coerce_scalar(time_index[i]) for i in x_positions]
    else:
        x_values = x_positions

    return {
        "x": x_values,
        "x_is_date": bool(x_is_date),
        # The row each point came from. The client needs these to turn a zoom
        # selection back into a row range and re-request that window at higher
        # resolution — without them a zoom could only stretch existing points.
        "rows": x_positions,
        "series": [
            {
                "name": names[i],
                "y": [jsonsafe.coerce_scalar(v) for v in series_values[i]],
            }
            for i in wanted
        ],
        "decimated": bucket_size > 1,
        "bucket_size": bucket_size,
        "start": start,
        "stop": stop,
        "window_rows": total,
        "total_rows": reader.nrows,
    }


def tree_payload(open_file: OpenFile, *, raw: bool = False) -> dict[str, Any]:
    return tree.build_tree(open_file.h5(), raw=raw)
