"""JSON coercion: the rules that keep responses parseable in the browser."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from h5grid import jsonsafe


class TestScalars:
    def test_nan_becomes_null(self):
        assert jsonsafe.coerce_scalar(np.float64("nan")) is None
        assert jsonsafe.coerce_scalar(float("nan")) is None

    def test_infinities_become_strings(self):
        assert jsonsafe.coerce_scalar(np.inf) == "Infinity"
        assert jsonsafe.coerce_scalar(-np.inf) == "-Infinity"

    def test_bytes_decode_as_utf8(self):
        assert jsonsafe.coerce_scalar(b"hello") == "hello"
        assert jsonsafe.coerce_scalar(np.bytes_(b"caf\xc3\xa9")) == "café"

    def test_invalid_utf8_is_replaced_not_raised(self):
        assert jsonsafe.coerce_scalar(b"\xff\xfe") == "��"

    def test_numpy_integers_become_python_ints(self):
        value = jsonsafe.coerce_scalar(np.int64(2**40))
        assert value == 2**40 and isinstance(value, int)

    def test_numpy_bool(self):
        assert jsonsafe.coerce_scalar(np.bool_(True)) is True

    def test_datetime_midnight_drops_time_part(self):
        assert jsonsafe.coerce_scalar(pd.Timestamp("1975-03-04")) == "1975-03-04"

    def test_datetime_with_time_keeps_it(self):
        assert (
            jsonsafe.coerce_scalar(pd.Timestamp("1975-03-04 06:30:00"))
            == "1975-03-04T06:30:00"
        )

    def test_nat_becomes_null(self):
        assert jsonsafe.coerce_scalar(pd.NaT) is None
        assert jsonsafe.coerce_scalar(np.datetime64("NaT")) is None

    def test_arrays_become_lists(self):
        assert jsonsafe.coerce_scalar(np.arange(3)) == [0, 1, 2]

    def test_output_is_strictly_json_serialisable(self):
        values = [np.nan, np.inf, -np.inf, np.int64(5), b"x", pd.Timestamp("2000-01-01")]
        encoded = json.dumps([jsonsafe.coerce_scalar(v) for v in values], allow_nan=False)
        assert json.loads(encoded) == [
            None,
            "Infinity",
            "-Infinity",
            5,
            "x",
            "2000-01-01",
        ]


class TestAttrs:
    def test_every_scalar_type_survives(self, plain):
        attrs = jsonsafe.coerce_attrs(plain.h5()["attributed"].attrs)
        assert attrs["int32"] == 32
        assert attrs["int64"] == 2**40
        assert attrs["uint16"] == 65535
        assert attrs["bool_true"] is True
        assert attrs["bytes"] == "raw bytes"
        assert attrs["unicode"] == "unicode éè 中文"
        assert attrs["int_array"] == [0, 1, 2, 3, 4]
        assert attrs["nan_attr"] is None
        assert attrs["inf_attr"] == "Infinity"
        assert attrs["empty_str"] == ""
        assert math.isclose(attrs["float64"], math.pi)

    def test_attrs_are_json_serialisable(self, plain):
        attrs = jsonsafe.coerce_attrs(plain.h5()["attributed"].attrs)
        json.dumps(attrs, allow_nan=False)


class TestFrameToRows:
    def test_clean_numeric_frame(self):
        frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
        assert jsonsafe.frame_to_rows(frame) == [[1.0, 3], [2.0, 4]]

    def test_nan_and_inf_in_a_float_column(self):
        frame = pd.DataFrame({"a": [1.0, np.nan, np.inf, -np.inf]})
        assert jsonsafe.frame_to_rows(frame) == [[1.0], [None], ["Infinity"], ["-Infinity"]]

    def test_datetime_column(self):
        frame = pd.DataFrame({"d": pd.to_datetime(["1975-01-01", "1975-01-02"])})
        assert jsonsafe.frame_to_rows(frame) == [["1975-01-01"], ["1975-01-02"]]

    def test_empty_frame(self):
        assert jsonsafe.frame_to_rows(pd.DataFrame({"a": []})) == []

    def test_bytes_column(self):
        frame = pd.DataFrame({"s": np.array([b"ab", b"cd"], dtype="S2")})
        assert jsonsafe.frame_to_rows(frame) == [["ab"], ["cd"]]

    def test_row_order_and_width_preserved(self):
        frame = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        rows = jsonsafe.frame_to_rows(frame)
        assert rows == [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
