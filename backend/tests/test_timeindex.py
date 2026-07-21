"""Date decoding: /time tables and integer epoch columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from h5grid import timeindex


class TestTimeTable:
    def test_found_in_pywr_files(self, pywr):
        assert timeindex.find_time_table(pywr.h5()) == "/time"

    def test_absent_elsewhere(self, plain, pandas_table):
        assert timeindex.find_time_table(plain.h5()) is None
        assert timeindex.find_time_table(pandas_table.h5()) is None

    def test_decodes_year_month_day_to_dates(self, pywr):
        index = timeindex.read_time_index(pywr.h5(), "/time")
        assert isinstance(index, pd.DatetimeIndex)
        assert len(index) == 3653
        assert index[0] == pd.Timestamp("1975-01-01")
        assert index[-1] == pd.Timestamp("1984-12-31")

    def test_cached_on_the_open_file(self, pywr):
        assert pywr.time_index() is pywr.time_index()

    def test_none_when_file_has_no_time_table(self, plain):
        assert plain.time_index() is None


class TestEpochUnitDetection:
    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_each_unit_detected(self, unit):
        dates = pd.date_range("1975-01-01", periods=100, freq="D")
        divisor = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}[unit]
        values = (dates.astype("int64") * 1000 // divisor).to_numpy()
        assert timeindex.detect_epoch_unit(values, "date") == unit

    def test_round_trip_through_decode(self):
        dates = pd.date_range("2001-06-15", periods=10, freq="D")
        values = dates.astype("int64").to_numpy() * 1000  # nanoseconds
        unit = timeindex.detect_epoch_unit(values, "time")
        decoded = pd.DatetimeIndex(timeindex.decode_epoch(values, unit))
        # Compare instants, not storage resolution: pandas 3 builds date_range
        # at microsecond precision while the decode yields nanoseconds.
        assert list(decoded.astype("datetime64[ns]")) == list(
            dates.astype("datetime64[ns]")
        )

    def test_row_counter_is_not_a_date(self):
        # The single most important negative case: pywr's /time table has an
        # `index` column of row numbers that must stay integers.
        assert timeindex.detect_epoch_unit(np.arange(3653), "index") is None
        assert timeindex.detect_epoch_unit(np.arange(50_000_000), "index") is None

    def test_large_values_in_a_non_date_column_are_left_alone(self):
        dates = pd.date_range("1975-01-01", periods=10).astype("int64").to_numpy()
        assert timeindex.detect_epoch_unit(dates, "flow") is None
        assert timeindex.detect_epoch_unit(dates, "reservoir_volume") is None

    def test_float_columns_are_never_dates(self):
        values = np.linspace(1e18, 2e18, 50)
        assert timeindex.detect_epoch_unit(values, "date") is None

    def test_empty_and_all_zero(self):
        assert timeindex.detect_epoch_unit(np.array([], dtype="i8"), "date") is None
        assert timeindex.detect_epoch_unit(np.zeros(10, dtype="i8"), "date") is None

    def test_mixed_magnitudes_rejected(self):
        # A column that starts near zero cannot be an epoch, whatever its max.
        values = np.array([0, 1, 2, 10**18], dtype="i8")
        assert timeindex.detect_epoch_unit(values, "date") is None


class TestDecodingInReaders:
    def test_epoch_column_renders_as_dates(self, plain):
        # /timestamps holds an integer epoch written by pandas.
        frame = plain.reader("/timestamps").read(0, 3)
        assert str(frame["value"].iloc[0].date()) == "1975-01-01"

    def test_time_table_index_column_stays_numeric(self, pywr):
        frame = pywr.reader("/time").read(0, 3)
        assert frame["index"].tolist() == [0, 1, 2]
        assert frame["year"].tolist() == [1975, 1975, 1975]


class TestTimeIndexApplication:
    def test_matching_row_count_makes_it_available(self, pywr):
        from h5grid import service

        assert service.time_index_matches(pywr, pywr.reader("/reservoir")) is True

    def test_mismatched_row_count_does_not(self, pywr):
        from h5grid import service

        # /monthly_summary has 120 rows against a 3653-row /time table.
        assert service.time_index_matches(pywr, pywr.reader("/monthly_summary")) is False

    def test_applied_as_a_leading_date_column(self, pywr):
        from h5grid import service

        payload = service.node_data(
            pywr, "/reservoir", stop=3, cols_spec="0:2", use_time_index=True
        )
        # Columns carry the scenario name from /scenarios; see test_pywr.py.
        assert payload["columns"] == ["date", "climate[0]", "climate[1]"]
        assert payload["rows"][0][0] == "1975-01-01"
        assert payload["time_index_applied"] is True

    def test_not_applied_when_not_requested(self, pywr):
        from h5grid import service

        payload = service.node_data(pywr, "/reservoir", stop=3, cols_spec="0:2")
        assert payload["columns"] == ["climate[0]", "climate[1]"]
        assert payload["time_index_applied"] is False

    def test_requesting_it_where_it_does_not_fit_is_ignored(self, pywr):
        from h5grid import service

        payload = service.node_data(
            pywr, "/monthly_summary", stop=3, use_time_index=True
        )
        assert payload["time_index_applied"] is False
        assert "date" not in payload["columns"]

    def test_dates_align_with_the_row_offset(self, pywr):
        from h5grid import service

        payload = service.node_data(
            pywr, "/reservoir", start=365, stop=367, cols_spec="0:1", use_time_index=True
        )
        assert payload["rows"][0][0] == "1976-01-01"
