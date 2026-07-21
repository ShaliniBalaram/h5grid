"""Every decoding rule in spec section 5.3, against the fixture files."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from h5grid.readers import (
    ColumnSelection,
    PandasFixedReader,
    PandasTableReader,
    RawDatasetReader,
    parse_cols,
    parse_dim_slice,
)


class TestRawDataset:
    def test_1d_becomes_single_value_column(self, plain):
        reader = plain.reader("/vector")
        assert isinstance(reader, RawDatasetReader)
        assert reader.column_names() == ["value"]
        assert reader.nrows == 1000

        frame = reader.read(0, 5)
        assert list(frame["value"]) == [0.0, 1.5, 3.0, 4.5, 6.0]

    def test_2d_rows_by_columns(self, plain):
        reader = plain.reader("/matrix")
        assert reader.shape == (5000, 12)
        assert reader.nrows == 5000

        frame = reader.read(0, 10)
        assert frame.shape == (10, 12)

    def test_2d_column_names_from_attribute(self, plain):
        # plain.h5 stores a `column_names` attribute on /matrix.
        reader = plain.reader("/matrix")
        assert reader.column_names()[:3] == ["node_0", "node_1", "node_2"]

    def test_2d_falls_back_to_generated_names(self, pywr):
        # /monthly_summary is (120, 4): no column_names attribute, and its shape
        # does not match the file's scenario sizes, so there is nothing to name
        # the columns after.
        reader = pywr.reader("/monthly_summary")
        assert reader.column_names() == ["col_0", "col_1", "col_2", "col_3"]

    def test_row_slicing_is_a_window_not_a_full_read(self, plain):
        reader = plain.reader("/matrix")
        frame = reader.read(100, 110)
        assert len(frame) == 10

        whole = reader.read(0, 5000)
        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True),
            whole.iloc[100:110].reset_index(drop=True),
        )

    def test_slice_beyond_end_is_clamped(self, plain):
        reader = plain.reader("/vector")
        assert len(reader.read(995, 5000)) == 5
        assert reader.read(5000, 6000).empty

    def test_compound_dtype_becomes_columns(self, plain):
        reader = plain.reader("/records")
        assert reader.dtype == "compound"
        assert reader.column_names() == ["id", "name", "flow", "active"]

        frame = reader.read(0, 3)
        assert frame["id"].tolist() == [0, 1, 2]
        # Fixed-length bytes decode to text rather than showing as b'...'.
        assert frame["name"].tolist() == ["reservoir_0", "reservoir_1", "reservoir_2"]
        assert frame["active"].dtype == bool

    def test_fixed_length_byte_strings_decode(self, plain):
        reader = plain.reader("/byte_labels")
        assert reader.read(0, 3)["value"].tolist() == ["AAA", "BBB", "CCC"]

    def test_variable_length_strings_decode(self, plain):
        reader = plain.reader("/labels")
        assert reader.read(0, 2)["value"].tolist() == ["Rutland Water", "Grafham"]

    def test_3d_uses_default_slice_of_zero(self, plain):
        reader = plain.reader("/cube")
        assert reader.ndim == 3
        assert reader.shape == (365, 8, 4)

        frame = reader.read(0, 2)
        assert frame.shape == (2, 8)  # dim1 free, dim2 pinned to 0

    def test_3d_slice_selects_a_different_plane(self, plain):
        reader = plain.reader("/cube")
        first = reader.read(0, 2, None, parse_dim_slice(",,0", 3))
        third = reader.read(0, 2, None, parse_dim_slice(",,2", 3))
        assert not np.allclose(first.to_numpy(), third.to_numpy())

        raw = plain.h5()["cube"][0:2, :, 2]
        assert np.allclose(third.to_numpy(), raw)

    def test_column_window(self, plain):
        reader = plain.reader("/matrix")
        frame = reader.read(0, 3, parse_cols("2:5"))
        assert list(frame.columns) == ["node_2", "node_3", "node_4"]

    def test_explicit_column_list(self, plain):
        reader = plain.reader("/matrix")
        frame = reader.read(0, 3, parse_cols("0,5,9"))
        assert list(frame.columns) == ["node_0", "node_5", "node_9"]

    def test_nan_and_inf_survive_the_read(self, plain):
        reader = plain.reader("/matrix")
        values = reader.read(10, 13, parse_cols("3:4"))["node_3"].tolist()
        assert math.isnan(values[0])
        assert values[1] == math.inf
        assert values[2] == -math.inf

    def test_blosc_compressed_dataset_reads(self, pywr):
        # PyTables writes blosc by default; without hdf5plugin registered this
        # raises "can't open directory .../plugin" instead of returning data.
        reader = pywr.reader("/reservoir")
        frame = reader.read(0, 5)
        assert frame.shape == (5, 20)
        assert np.isfinite(frame.to_numpy()).all()

    def test_compression_is_reported_even_when_h5py_cannot_name_it(self, pywr):
        reader = pywr.reader("/reservoir")
        assert reader.compression is not None
        assert "blosc" in reader.compression

    def test_chunks_reported(self, plain):
        assert plain.reader("/matrix").chunks == (500, 12)


class TestPandasTableReader:
    def test_selected_for_frame_table_nodes(self, pandas_table):
        reader = pandas_table.reader("/monthly")
        assert isinstance(reader, PandasTableReader)
        assert reader.kind == "pandas_table"

    def test_index_becomes_first_column(self, pandas_table):
        reader = pandas_table.reader("/monthly")
        assert reader.column_names() == [
            "date",
            "flow",
            "level",
            "demand",
            "spill",
            "count",
        ]
        assert reader.columns[0].is_datetime is True

    def test_true_lazy_row_slicing(self, pandas_table):
        reader = pandas_table.reader("/monthly")
        assert reader.supports_row_slicing is True
        assert reader.nrows == 3653

        frame = reader.read(100, 103)
        assert len(frame) == 3
        assert str(frame["date"].iloc[0].date()) == "1975-04-11"

    def test_slice_matches_pandas_read_hdf(self, pandas_table, fixture_files):
        expected = pd.read_hdf(fixture_files / "pandas_table.h5", "monthly")
        frame = pandas_table.reader("/monthly").read(500, 505)
        np.testing.assert_allclose(
            frame["flow"].to_numpy(), expected["flow"].to_numpy()[500:505]
        )

    def test_node_under_a_group_path(self, pandas_table):
        reader = pandas_table.reader("/results/deep/nested")
        assert reader.nrows == 100
        assert reader.read(0, 2)["date"].iloc[0] == pd.Timestamp("1975-01-01")

    def test_nan_preserved(self, pandas_table):
        frame = pandas_table.reader("/monthly").read(5, 10)
        assert frame["flow"].isna().all()


class TestPandasFixedReader:
    def test_selected_for_frame_nodes(self, pandas_fixed):
        reader = pandas_fixed.reader("/monthly")
        assert isinstance(reader, PandasFixedReader)
        assert reader.kind == "pandas_frame"
        assert reader.supports_row_slicing is False

    def test_decodes_to_the_saved_table_not_raw_blocks(self, pandas_fixed):
        reader = pandas_fixed.reader("/monthly")
        assert reader.column_names() == [
            "date",
            "flow",
            "level",
            "demand",
            "spill",
            "count",
        ]
        # The whole point: no axis0 / block0_values leaking through.
        assert not any("block" in name for name in reader.column_names())

    def test_serves_slices_from_the_cached_frame(self, pandas_fixed):
        reader = pandas_fixed.reader("/monthly")
        first = reader.read(0, 5)
        again = reader.read(0, 5)
        pd.testing.assert_frame_equal(first, again)
        assert len(reader.read(3000, 3010)) == 10

    def test_matches_pandas_read_hdf(self, pandas_fixed, fixture_files):
        expected = pd.read_hdf(fixture_files / "pandas_fixed.h5", "monthly")
        frame = pandas_fixed.reader("/monthly").read(10, 20)
        np.testing.assert_allclose(
            frame["level"].to_numpy(), expected["level"].to_numpy()[10:20]
        )

    def test_non_datetime_index_left_alone(self, pandas_fixed):
        reader = pandas_fixed.reader("/plain_index")
        assert reader.column_names() == ["index", "a", "b"]
        assert reader.columns[0].is_datetime is False
        assert reader.read(0, 3)["index"].tolist() == [0, 1, 2]

    def test_size_guard_trips_and_reports_why(self, pandas_fixed, monkeypatch):
        import h5grid.readers as readers_module

        monkeypatch.setattr(readers_module, "FIXED_FORMAT_SIZE_LIMIT_BYTES", 100)
        store = pandas_fixed.store()
        reader = PandasFixedReader(store, "/monthly", {})

        assert reader.decode_fallback is not None
        assert "format='table'" in reader.decode_fallback
        assert reader.columns == []
        with pytest.raises(ValueError):
            reader.read(0, 10)

    def test_size_guard_does_not_read_the_frame(self, pandas_fixed, monkeypatch):
        import h5grid.readers as readers_module

        monkeypatch.setattr(readers_module, "FIXED_FORMAT_SIZE_LIMIT_BYTES", 100)
        cache: dict = {}
        PandasFixedReader(pandas_fixed.store(), "/monthly", cache)
        assert cache == {}, "the guard must trip before anything is decoded"


class TestReaderSelection:
    def test_selection_happens_once_and_is_cached(self, pandas_table):
        first = pandas_table.reader("/monthly")
        assert pandas_table.reader("/monthly") is first

    def test_group_is_not_readable(self, plain):
        from h5grid.files import NodeNotFoundError

        with pytest.raises(NodeNotFoundError):
            plain.reader("/inputs")

    def test_missing_path_raises(self, plain):
        from h5grid.files import NodeNotFoundError

        with pytest.raises(NodeNotFoundError):
            plain.reader("/nope")


class TestParsing:
    @pytest.mark.parametrize(
        "spec,expected",
        [
            (None, None),
            ("", None),
            ("0:3", [0, 1, 2]),
            ("2:5", [2, 3, 4]),
            ("1,4,9", [1, 4, 9]),
            ("3", [3]),
        ],
    )
    def test_parse_cols(self, spec, expected):
        assert parse_cols(spec).indices == expected

    def test_parse_cols_rejects_open_window(self):
        with pytest.raises(ValueError):
            parse_cols("5:")

    def test_parse_cols_rejects_reversed_window(self):
        with pytest.raises(ValueError):
            parse_cols("9:2")

    def test_parse_dim_slice_defaults_extra_dims_to_zero(self):
        assert parse_dim_slice(None, 3) == [None, None, 0]
        assert parse_dim_slice(None, 4) == [None, None, 0, 0]
        assert parse_dim_slice(None, 2) == [None, None]

    def test_parse_dim_slice_explicit(self):
        assert parse_dim_slice(",,2", 3) == [None, None, 2]
        assert parse_dim_slice(",1,", 3) == [None, 1, None]

    def test_parse_dim_slice_rejects_pinned_row_axis(self):
        with pytest.raises(ValueError):
            parse_dim_slice("0,,", 3)

    def test_parse_dim_slice_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            parse_dim_slice(",,,,", 2)

    def test_parse_dim_slice_pins_surplus_free_dims(self):
        assert parse_dim_slice(",,,", 4) == [None, None, 0, 0]


class TestColumnSelection:
    def test_all_columns_by_default(self):
        assert ColumnSelection().apply(["a", "b", "c"]) == [0, 1, 2]

    def test_out_of_range_indices_dropped(self):
        assert ColumnSelection([0, 99]).apply(["a", "b"]) == [0]
