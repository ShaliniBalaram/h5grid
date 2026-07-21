"""Column statistics and server-side value search."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from h5grid import stats


class TestColumnStats:
    def test_matches_numpy_on_a_clean_column(self, plain):
        reader = plain.reader("/matrix")
        result = stats.column_stats(reader, "node_0")
        expected = plain.h5()["matrix"][:, 0]

        assert math.isclose(result["min"], float(expected.min()))
        assert math.isclose(result["max"], float(expected.max()))
        assert math.isclose(result["mean"], float(expected.mean()), rel_tol=1e-12)
        assert math.isclose(result["std"], float(expected.std(ddof=1)), rel_tol=1e-9)
        assert result["nan_count"] == 0
        assert result["count"] == 5000

    def test_nan_and_inf_excluded_from_min_max(self, plain):
        result = stats.column_stats(plain.reader("/matrix"), "node_3")
        assert result["nan_count"] == 1
        assert math.isfinite(result["min"]) and math.isfinite(result["max"])
        assert result["count"] == 4997  # 5000 less one NaN and two infinities

    def test_all_nan_column(self, plain):
        result = stats.column_stats(plain.reader("/matrix"), "node_11")
        assert result["count"] == 0
        assert result["nan_count"] == 5000
        assert result["min"] is None and result["mean"] is None

    def test_chunked_result_matches_single_pass(self, plain):
        reader = plain.reader("/matrix")
        chunked = stats.column_stats(reader, "node_1", chunk_rows=97)
        whole = stats.column_stats(reader, "node_1", chunk_rows=10**9)
        for key in ("min", "max", "count", "nan_count"):
            assert chunked[key] == whole[key]
        assert math.isclose(chunked["mean"], whole["mean"], rel_tol=1e-12)
        assert math.isclose(chunked["std"], whole["std"], rel_tol=1e-9)

    def test_precision_holds_for_tightly_clustered_values(self):
        # Naive sum-of-squares loses this to cancellation; Chan's update does not.
        values = 25.0 + np.random.default_rng(0).normal(0, 1e-4, 200_000)
        running = stats.RunningStats()
        for start in range(0, values.size, 1000):
            running.update(values[start : start + 1000])
        result = running.to_json()
        assert math.isclose(result["std"], float(values.std(ddof=1)), rel_tol=1e-6)
        assert result["std"] > 0

    def test_datetime_column_reports_date_bounds(self, pandas_table):
        result = stats.column_stats(pandas_table.reader("/monthly"), "date")
        assert result["min"].startswith("1975-01-01")
        assert result["max"].startswith("1984-12-31")

    def test_text_column_reports_distinct_counts(self, plain):
        result = stats.column_stats(plain.reader("/labels"), "value")
        assert result["numeric"] is False
        assert result["distinct_count"] == 5
        assert result["total"] == 100

    def test_unknown_column_raises(self, plain):
        with pytest.raises(KeyError):
            stats.column_stats(plain.reader("/matrix"), "not_a_column")

    def test_integer_column(self, pandas_table):
        result = stats.column_stats(pandas_table.reader("/monthly"), "count")
        assert result["count"] == 3653
        assert 0 <= result["min"] <= result["max"] <= 100


class TestSearch:
    def test_greater_than(self, plain):
        reader = plain.reader("/matrix")
        result = stats.search_column(reader, "node_0", ">180", limit=10)
        column = plain.h5()["matrix"][:, 0]
        for row in result["rows"]:
            assert column[row] > 180

    def test_returns_row_numbers_in_order(self, plain):
        result = stats.search_column(plain.reader("/matrix"), "node_0", ">150")
        assert result["rows"] == sorted(result["rows"])

    def test_range_query(self, plain):
        reader = plain.reader("/matrix")
        result = stats.search_column(reader, "node_0", "99..101", limit=50)
        column = plain.h5()["matrix"][:, 0]
        for row in result["rows"]:
            assert 99 <= column[row] <= 101

    def test_equality_on_integers(self, pandas_table):
        reader = pandas_table.reader("/monthly")
        result = stats.search_column(reader, "count", "=42", limit=20)
        frame = reader.read(0, reader.nrows)
        for row in result["rows"]:
            assert frame["count"].iloc[row] == 42

    def test_substring_match_on_text(self, plain):
        result = stats.search_column(plain.reader("/labels"), "value", "graf")
        assert result["rows"], "case-insensitive substring should match Grafham"
        frame = plain.reader("/labels").read(0, 100)
        for row in result["rows"]:
            assert "graf" in frame["value"].iloc[row].lower()

    def test_limit_is_honoured_and_flagged(self, plain):
        result = stats.search_column(plain.reader("/matrix"), "node_0", ">0", limit=5)
        assert len(result["rows"]) == 5
        assert result["truncated"] is True

    def test_no_matches(self, plain):
        result = stats.search_column(plain.reader("/matrix"), "node_0", ">1e9")
        assert result["rows"] == []
        assert result["truncated"] is False

    def test_chunk_boundaries_do_not_lose_rows(self, plain):
        reader = plain.reader("/matrix")
        small = stats.search_column(reader, "node_0", ">150", limit=5000, chunk_rows=13)
        large = stats.search_column(reader, "node_0", ">150", limit=5000, chunk_rows=10**9)
        assert small["rows"] == large["rows"]
