"""Coercion of numpy/pandas values into strictly JSON-safe Python objects.

Every value leaving the API passes through here. The rules are fixed by the
spec: NaN becomes null, +/-Inf become the strings "Infinity"/"-Infinity",
bytes decode as UTF-8 with replacement, and datetimes become ISO-8601.

Standard `json` would happily emit bare `NaN` and `Infinity` tokens, which are
not valid JSON and make `JSON.parse` throw in the browser. So we never hand raw
floats to the encoder without checking them first.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any

import numpy as np
import pandas as pd

INFINITY = "Infinity"
NEG_INFINITY = "-Infinity"


def coerce_scalar(value: Any) -> Any:
    """Convert a single numpy/pandas/Python value into something JSON-safe."""
    # Order matters: pd.isna on an array raises, so handle containers first.
    if value is None:
        return None

    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")

    if isinstance(value, np.ndarray):
        return [coerce_scalar(v) for v in value.tolist()]

    if isinstance(value, (list, tuple)):
        return [coerce_scalar(v) for v in value]

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if isinstance(value, (pd.Timestamp, _dt.datetime)):
        if pd.isna(value):
            return None
        return _format_timestamp(value)

    if isinstance(value, _dt.date):
        return value.isoformat()

    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        return _format_timestamp(pd.Timestamp(value))

    if isinstance(value, (np.timedelta64, pd.Timedelta)):
        return str(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        f = float(value)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return INFINITY if f > 0 else NEG_INFINITY
        return f

    if isinstance(value, int):
        return value

    if isinstance(value, (np.str_, str)):
        return str(value)

    if isinstance(value, np.void):  # a compound-dtype record
        return [coerce_scalar(v) for v in value.tolist()]

    if value is pd.NaT:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def _format_timestamp(ts: pd.Timestamp) -> str:
    """ISO-8601, dropping the time part when it is exactly midnight."""
    if ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.microsecond == 0:
        if getattr(ts, "nanosecond", 0) == 0:
            return ts.strftime("%Y-%m-%d")
    return ts.isoformat()


def coerce_attrs(attrs: Any) -> dict[str, Any]:
    """Turn an h5py AttributeManager into a JSON-safe dict."""
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        try:
            out[str(key)] = coerce_scalar(value)
        except Exception as exc:  # never let one odd attribute break the node
            out[str(key)] = f"<unreadable: {type(exc).__name__}>"
    return out


def frame_to_rows(df: pd.DataFrame) -> list[list[Any]]:
    """Serialize a DataFrame's body to JSON-safe row lists.

    Column-at-a-time so the fast paths (clean float and int columns) avoid a
    per-cell Python call, which matters at 200k cells per request.
    """
    if df.empty:
        return []

    columns: list[list[Any]] = []
    for name in df.columns:
        columns.append(_coerce_column(df[name]))

    return [list(row) for row in zip(*columns)]


def _coerce_column(series: pd.Series) -> list[Any]:
    values = series.to_numpy()
    kind = values.dtype.kind

    if kind == "f":
        finite = np.isfinite(values)
        if finite.all():
            return values.tolist()
        out: list[Any] = values.tolist()
        for i, ok in enumerate(finite):
            if not ok:
                v = out[i]
                if math.isnan(v):
                    out[i] = None
                else:
                    out[i] = INFINITY if v > 0 else NEG_INFINITY
        return out

    if kind in "iu":
        return values.tolist()

    if kind == "b":
        return [bool(v) for v in values.tolist()]

    if kind == "M":  # datetime64
        return [None if pd.isna(v) else _format_timestamp(pd.Timestamp(v)) for v in values]

    if kind == "S":
        return [v.decode("utf-8", errors="replace") for v in values.tolist()]

    return [coerce_scalar(v) for v in values.tolist()]
