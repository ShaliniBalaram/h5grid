"""Per-column summary statistics, computed in chunks so memory stays flat.

Mean and standard deviation combine across chunks with Chan's parallel update
rather than by accumulating sum and sum-of-squares. On a 50M-row column of
reservoir levels that all sit near 25.0, the naive approach loses most of its
significant digits to cancellation and can even produce a negative variance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .readers import ColumnSelection, NodeReader

CHUNK_ROWS = 1_000_000


@dataclass
class RunningStats:
    count: int = 0          # finite values only
    total: int = 0          # every value seen, including NaN
    nan_count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: np.ndarray) -> None:
        self.total += int(values.size)
        if values.size == 0:
            return

        finite_mask = np.isfinite(values)
        self.nan_count += int(np.count_nonzero(np.isnan(values)))
        finite = values[finite_mask]
        if finite.size == 0:
            return

        self.minimum = min(self.minimum, float(finite.min()))
        self.maximum = max(self.maximum, float(finite.max()))

        n_b = int(finite.size)
        mean_b = float(finite.mean())
        m2_b = float(((finite - mean_b) ** 2).sum())

        n_a, mean_a, m2_a = self.count, self.mean, self.m2
        n_ab = n_a + n_b
        delta = mean_b - mean_a
        self.mean = mean_a + delta * (n_b / n_ab)
        self.m2 = m2_a + m2_b + (delta**2) * n_a * n_b / n_ab
        self.count = n_ab

    def to_json(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "nan_count": self.nan_count,
                "count": 0,
                "total": self.total,
            }
        variance = self.m2 / (self.count - 1) if self.count > 1 else 0.0
        return {
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std": math.sqrt(max(variance, 0.0)),
            "nan_count": self.nan_count,
            "count": self.count,
            "total": self.total,
        }


def column_stats(
    reader: NodeReader,
    column: str,
    dim_slice: list[int | None] | None = None,
    chunk_rows: int = CHUNK_ROWS,
) -> dict[str, Any]:
    """Summary statistics for one column, read in chunks."""
    names = reader.column_names()
    if column not in names:
        raise KeyError(f"No column {column!r}. Available: {', '.join(names[:20])}")

    index = names.index(column)
    selection = ColumnSelection([index])
    running = RunningStats()
    non_numeric = False

    for start in range(0, reader.nrows, chunk_rows):
        stop = min(start + chunk_rows, reader.nrows)
        frame = reader.read(start, stop, selection, dim_slice)
        if frame.empty:
            continue
        series = frame.iloc[:, 0]

        if pd.api.types.is_datetime64_any_dtype(series):
            values = series.to_numpy().astype("datetime64[ns]").astype("float64")
            values[pd.isna(series).to_numpy()] = np.nan
        elif pd.api.types.is_bool_dtype(series):
            values = series.to_numpy().astype("float64")
        elif pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy().astype("float64", copy=False)
        else:
            non_numeric = True
            break

        running.update(values)

    if non_numeric:
        return _non_numeric_stats(reader, column, index, dim_slice, chunk_rows)

    result = running.to_json()
    result["column"] = column
    result["numeric"] = True

    spec = reader.columns[index]
    if spec.is_datetime and result["min"] is not None:
        result["min"] = str(pd.Timestamp(int(result["min"])))
        result["max"] = str(pd.Timestamp(int(result["max"])))
        result["mean"] = str(pd.Timestamp(int(running.mean)))
        result["std"] = None
    return result


def _non_numeric_stats(
    reader: NodeReader,
    column: str,
    index: int,
    dim_slice: list[int | None] | None,
    chunk_rows: int,
) -> dict[str, Any]:
    """For text columns: counts and distinct values instead of min/max/mean."""
    selection = ColumnSelection([index])
    total = 0
    missing = 0
    distinct: set[str] = set()
    truncated = False

    for start in range(0, reader.nrows, chunk_rows):
        stop = min(start + chunk_rows, reader.nrows)
        series = reader.read(start, stop, selection, dim_slice).iloc[:, 0]
        total += int(series.size)
        missing += int(series.isna().sum())
        if not truncated:
            for value in series.dropna().unique():
                distinct.add(str(value))
                if len(distinct) > 1000:
                    truncated = True
                    break

    return {
        "column": column,
        "numeric": False,
        "count": total - missing,
        "total": total,
        "nan_count": missing,
        "distinct_count": len(distinct),
        "distinct_truncated": truncated,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
    }


def search_column(
    reader: NodeReader,
    column: str,
    query: str,
    *,
    limit: int = 500,
    chunk_rows: int = CHUNK_ROWS,
    dim_slice: list[int | None] | None = None,
) -> dict[str, Any]:
    """Scan a column server-side and return the row numbers that match.

    Numeric queries support comparisons (`>100`, `<=0`, `=5`, `3..7`); anything
    else is a case-insensitive substring match on the formatted value.
    """
    names = reader.column_names()
    if column not in names:
        raise KeyError(f"No column {column!r}")

    index = names.index(column)
    selection = ColumnSelection([index])
    predicate = _build_predicate(query)

    matches: list[int] = []
    scanned = 0
    for start in range(0, reader.nrows, chunk_rows):
        stop = min(start + chunk_rows, reader.nrows)
        series = reader.read(start, stop, selection, dim_slice).iloc[:, 0]
        scanned = stop
        hits = predicate(series)
        if hits is None:
            continue
        for offset in np.flatnonzero(np.asarray(hits)):
            matches.append(start + int(offset))
            if len(matches) >= limit:
                return {
                    "column": column,
                    "query": query,
                    "rows": matches,
                    "truncated": True,
                    "scanned_rows": scanned,
                }

    return {
        "column": column,
        "query": query,
        "rows": matches,
        "truncated": False,
        "scanned_rows": scanned,
    }


def _build_predicate(query: str):
    text = query.strip()

    for op in (">=", "<=", "!=", ">", "<", "="):
        if text.startswith(op):
            try:
                threshold = float(text[len(op) :].strip())
            except ValueError:
                break
            return lambda s, op=op, t=threshold: _compare(s, op, t)

    if ".." in text:
        lo_s, _, hi_s = text.partition("..")
        try:
            lo, hi = float(lo_s), float(hi_s)
        except ValueError:
            pass
        else:
            return lambda s: _numeric(s) is not None and (
                (_numeric(s) >= lo) & (_numeric(s) <= hi)
            )

    needle = text.lower()
    return lambda s: s.astype(str).str.lower().str.contains(needle, regex=False).to_numpy()


def _numeric(series: pd.Series):
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy().astype("float64", copy=False)
    return None


def _compare(series: pd.Series, op: str, threshold: float):
    values = _numeric(series)
    if values is None:
        return None
    with np.errstate(invalid="ignore"):
        if op == ">":
            return values > threshold
        if op == "<":
            return values < threshold
        if op == ">=":
            return values >= threshold
        if op == "<=":
            return values <= threshold
        if op == "!=":
            return values != threshold
        return values == threshold
