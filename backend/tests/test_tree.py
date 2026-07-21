"""Tree walking: pandas node detection, hidden internals, decoded shapes."""

from __future__ import annotations

import pytest

from h5grid import tree


def flatten(node, out=None):
    out = {} if out is None else out
    if node["path"] != "/":
        out[node["path"]] = node
    for child in node["children"]:
        flatten(child, out)
    return out


@pytest.fixture
def plain_tree(plain):
    return flatten(tree.build_tree(plain.h5()))


@pytest.fixture
def fixed_tree(pandas_fixed):
    return flatten(tree.build_tree(pandas_fixed.h5()))


@pytest.fixture
def table_tree(pandas_table):
    return flatten(tree.build_tree(pandas_table.h5()))


@pytest.fixture
def pywr_tree(pywr):
    return flatten(tree.build_tree(pywr.h5()))


class TestPlainFiles:
    def test_datasets_and_groups_classified(self, plain_tree):
        assert plain_tree["/vector"]["kind"] == "dataset"
        assert plain_tree["/inputs"]["kind"] == "group"

    def test_nested_groups_are_walked(self, plain_tree):
        assert "/inputs/nested/deeper/deepest/leaf" in plain_tree
        assert plain_tree["/inputs/catchment_a/rainfall"]["shape"] == [3653, 3]

    def test_shape_and_dtype_reported(self, plain_tree):
        node = plain_tree["/matrix"]
        assert node["shape"] == [5000, 12]
        assert node["dtype"] == "float64"
        assert node["nrows"] == 5000
        assert node["ncols"] == 12

    def test_compound_dtype_labelled(self, plain_tree):
        node = plain_tree["/records"]
        assert node["dtype"] == "compound"
        assert node["ncols"] == 4

    def test_3d_shape_and_ndim(self, plain_tree):
        assert plain_tree["/cube"]["shape"] == [365, 8, 4]
        assert plain_tree["/cube"]["ndim"] == 3


class TestPandasFixedNodes:
    def test_group_with_pandas_type_becomes_a_table_node(self, fixed_tree):
        node = fixed_tree["/monthly"]
        assert node["kind"] == "pandas_frame"
        assert node["dtype"] == "frame"

    def test_internal_blocks_are_hidden(self, fixed_tree):
        assert fixed_tree["/monthly"]["children"] == []
        for path in fixed_tree:
            assert "axis0" not in path
            assert "block0_values" not in path

    def test_decoded_shape_not_block_shape(self, fixed_tree):
        # 5 columns across two blocks (4 float + 1 int); the node must report 5,
        # not the width of any single block.
        assert fixed_tree["/monthly"]["shape"] == [3653, 5]
        assert fixed_tree["/monthly"]["nrows"] == 3653

    def test_raw_toggle_reveals_internals(self, pandas_fixed):
        raw = flatten(tree.build_tree(pandas_fixed.h5(), raw=True))
        assert "/monthly/axis0" in raw
        assert "/monthly/block0_values" in raw

    def test_shape_computed_without_decoding(self, pandas_fixed):
        # Building the tree must not populate the decoded-frame cache.
        tree.build_tree(pandas_fixed.h5())
        assert pandas_fixed.frame_cache() == {}


class TestPandasTableNodes:
    def test_frame_table_detected(self, table_tree):
        node = table_tree["/monthly"]
        assert node["kind"] == "pandas_table"
        assert node["dtype"] == "frame_table"
        assert node["shape"] == [3653, 5]

    def test_pytables_index_containers_are_hidden(self, table_tree):
        # data_columns=True creates a _i_table group full of index arrays.
        for path in table_tree:
            assert "_i_table" not in path

    def test_table_dataset_itself_is_hidden(self, table_tree):
        assert "/monthly/table" not in table_tree
        assert table_tree["/monthly"]["children"] == []

    def test_table_node_nested_under_groups(self, table_tree):
        assert table_tree["/results"]["kind"] == "group"
        assert table_tree["/results/deep/nested"]["kind"] == "pandas_table"
        assert table_tree["/results/deep/nested"]["nrows"] == 100


class TestPywrFiles:
    def test_recorder_arrays_are_plain_datasets(self, pywr_tree):
        node = pywr_tree["/reservoir"]
        assert node["kind"] == "dataset"
        assert node["shape"] == [3653, 20]

    def test_time_table_present_as_compound_dataset(self, pywr_tree):
        assert pywr_tree["/time"]["dtype"] == "compound"
        assert pywr_tree["/time"]["nrows"] == 3653

    def test_nested_node_group(self, pywr_tree):
        assert pywr_tree["/subcatchment"]["kind"] == "group"
        assert pywr_tree["/subcatchment/tributary"]["shape"] == [3653, 20]


class TestFallback:
    def test_oversized_fixed_frame_shown_as_raw_group(self, pandas_fixed, monkeypatch):
        import h5grid.tree as tree_module

        monkeypatch.setattr(tree_module, "FIXED_FORMAT_SIZE_LIMIT_BYTES", 100)
        nodes = flatten(tree.build_tree(pandas_fixed.h5()))

        node = nodes["/monthly"]
        assert node["decode_fallback"] is True
        assert node["kind"] == "group"
        # The data stays reachable as raw blocks rather than vanishing.
        assert "/monthly/block0_values" in nodes


class TestFindNodeKind:
    def test_classifies_each_kind(self, plain, pandas_fixed, pandas_table):
        assert tree.find_node_kind(plain.h5(), "/matrix") == "dataset"
        assert tree.find_node_kind(plain.h5(), "/inputs") == "group"
        assert tree.find_node_kind(pandas_fixed.h5(), "/monthly") == "pandas_frame"
        assert tree.find_node_kind(pandas_table.h5(), "/monthly") == "pandas_table"

    def test_root_is_a_group(self, plain):
        assert tree.find_node_kind(plain.h5(), "/") == "group"
