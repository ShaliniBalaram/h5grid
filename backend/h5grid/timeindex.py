"""Turning the various ways a model stores time into real dates.

Two separate jobs:

  * `/time` tables, as written by pywr's TablesRecorder: a compound dataset with
    year/month/day columns (and usually an `index` column of row numbers). Any
    output array whose row count matches becomes date-indexed.
  * loose int64 columns holding a nanosecond epoch, which pandas writes for
    DatetimeIndexes and which every other viewer shows as 19-digit integers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIME_TABLE_PATHS = ("/time", "/times", "/timestep", "/timesteps")

# Column names that may hold an encoded datetime. `index` is included because
# pandas names a DatetimeIndex column that way inside HDFStore tables.
DATETIME_COLUMN_NAMES = {
    "time",
    "times",
    "date",
    "dates",
    "datetime",
    "datetimes",
    "timestamp",
    "timestamps",
    "index",
    "date_time",
    "time_index",
}

# Epoch magnitude bands, one per unit, covering roughly 1973 to 2224. pandas 2
# stored DatetimeIndexes as nanoseconds; pandas 3 defaults to microseconds, and
# other tools write seconds or milliseconds, so all four turn up in real files.
# The bands do not overlap, which is what lets the unit be inferred.
#
# A row counter is never mistaken for a date because both the smallest and the
# largest value must fall inside a band: an index column starts at 0 or 1, so
# its minimum lands far below any band floor.
_EPOCH_BANDS: tuple[tuple[str, float, float], ...] = (
    ("s", 1e8, 8e9),
    ("ms", 1e11, 8e12),
    ("us", 1e14, 8e15),
    ("ns", 1e17, 8e18),
)


def find_time_table(h5file) -> str | None:
    """Return the path of a usable `/time` table, if the file has one."""
    for path in TIME_TABLE_PATHS:
        name = path.lstrip("/")
        if name not in h5file:
            continue
        node = h5file[name]
        if not hasattr(node, "dtype"):
            continue
        names = node.dtype.names
        if names and {"year", "month", "day"} <= {n.lower() for n in names}:
            return path
    return None


def read_time_index(h5file, path: str) -> pd.DatetimeIndex:
    """Build a DatetimeIndex from a year/month/day table."""
    data = h5file[path.lstrip("/")][...]
    fields = {n.lower(): n for n in data.dtype.names}

    parts = {
        "year": data[fields["year"]].astype("int64"),
        "month": data[fields["month"]].astype("int64"),
        "day": data[fields["day"]].astype("int64"),
    }
    for optional in ("hour", "minute", "second"):
        if optional in fields:
            parts[optional] = data[fields[optional]].astype("int64")

    return pd.DatetimeIndex(pd.to_datetime(pd.DataFrame(parts)))


def detect_epoch_unit(values: np.ndarray, name: str = "") -> str | None:
    """Return the epoch unit ('s'/'ms'/'us'/'ns') of an integer column, or None.

    Requires both a matching column name and a plausible magnitude. Magnitude
    alone is not enough: a column of large integers that happens to sit in range
    should not silently become dates.
    """
    if values.dtype.kind not in "iu" or values.size == 0:
        return None
    if name and name.lower() not in DATETIME_COLUMN_NAMES:
        return None

    nonzero = values[values != 0]
    if nonzero.size == 0:
        return None

    magnitudes = np.abs(nonzero.astype("float64"))
    lo = float(magnitudes.min())
    hi = float(magnitudes.max())

    for unit, floor, ceiling in _EPOCH_BANDS:
        if lo >= floor and hi <= ceiling:
            return unit
    return None


def looks_like_epoch(values: np.ndarray, name: str = "") -> bool:
    return detect_epoch_unit(values, name) is not None


def decode_epoch(values: np.ndarray, unit: str = "ns") -> pd.DatetimeIndex:
    return pd.to_datetime(values.astype("int64"), unit=unit)


def maybe_decode_datetime_column(values: np.ndarray, name: str):
    """Decode a column to datetimes when it looks like one, else leave it."""
    unit = detect_epoch_unit(values, name)
    if unit is not None:
        return decode_epoch(values, unit), True
    return values, False
