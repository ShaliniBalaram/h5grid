"""Guards against failure modes other HDF5 viewers are known to have.

Each test here corresponds to a defect reported against a shipped viewer. They
are cheap to keep and they protect properties that are easy to break silently
while refactoring.
"""

from __future__ import annotations

import subprocess
import sys

import h5py
import numpy as np
import pytest

from h5grid import tree


class TestFileLockingOrder:
    """HDF5 file locking must be disabled before the C library loads.

    Every viewer does this, and silx shipped a bugfix release because an import
    reorder silently undid it — after which users could no longer open a file
    that a running job held open. Ours is set in h5grid/__init__.py; this test
    fails if that ordering is ever disturbed.
    """

    def test_env_var_set_after_importing_h5grid(self):
        code = (
            "import h5grid, os, sys;"
            "print(os.environ.get('HDF5_USE_FILE_LOCKING'));"
            "print('h5py' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        locking, h5py_loaded = result.stdout.split()
        assert locking == "FALSE"
        # hdf5plugin pulls h5py in, which is exactly why the env var has to be
        # set above that import rather than below it.
        assert h5py_loaded == "True"

    def test_env_var_precedes_h5py_import_in_source(self):
        source = (__import__("h5grid").__file__)
        text = open(source).read()
        assert text.index("HDF5_USE_FILE_LOCKING") < text.index("import hdf5plugin")


class TestCompressionPluginsRegistered:
    """Reading a blosc-compressed file must work.

    PyTables (and so pywr) writes blosc by default. Without hdf5plugin imported,
    h5py fails with "can't open directory .../plugin" — the same class of error
    HDFView users hit with a missing LZF plugin.
    """

    def test_blosc_dataset_reads(self, pywr):
        values = pywr.reader("/reservoir").read(0, 5)
        assert values.shape == (5, 20)
        assert np.isfinite(values.to_numpy()).all()


class TestNaturalSort:
    """`node_2` must come before `node_10`.

    Plain lexicographic ordering interleaves unpadded model node names, which is
    an open complaint against h5web (#1760) and makes a long tree hard to scan.
    """

    def test_numeric_suffixes_order_numerically(self):
        names = ["node_1", "node_10", "node_100", "node_2", "node_20"]
        assert tree.natural_sorted(names) == [
            "node_1",
            "node_2",
            "node_10",
            "node_20",
            "node_100",
        ]

    def test_mixed_names_still_sort(self):
        names = ["reservoir", "abstraction_2", "abstraction_10", "Weir"]
        assert tree.natural_sorted(names) == [
            "abstraction_2",
            "abstraction_10",
            "reservoir",
            "Weir",
        ]

    def test_applied_when_walking(self, tmp_path):
        path = tmp_path / "unpadded.h5"
        with h5py.File(path, "w") as f:
            for i in (1, 2, 10, 20, 100):
                f.create_dataset(f"node_{i}", data=np.zeros(3))

        with h5py.File(path, "r") as f:
            names = [c["name"] for c in tree.build_tree(f)["children"]]
        assert names == ["node_1", "node_2", "node_10", "node_20", "node_100"]


class TestBrokenLinks:
    """A link that cannot be resolved must be shown, not silently dropped.

    A node that vanishes from the tree is indistinguishable from one the model
    never wrote. h5py's own `visit` skips external links silently, and h5web
    had the opposite failure — a broken link taking over the whole screen.
    """

    @pytest.fixture
    def file_with_broken_links(self, tmp_path):
        path = tmp_path / "links.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("real", data=np.arange(10, dtype="f8"))
            f["dangling"] = h5py.SoftLink("/nowhere")
            f["missing_file"] = h5py.ExternalLink("no_such_file.h5", "/data")
        return path

    def test_broken_links_appear_as_nodes(self, file_with_broken_links):
        with h5py.File(file_with_broken_links, "r") as f:
            children = {c["name"]: c for c in tree.build_tree(f)["children"]}

        assert set(children) == {"real", "dangling", "missing_file"}
        assert children["real"]["kind"] == "dataset"
        assert children["dangling"]["kind"] == "broken_link"
        assert children["missing_file"]["kind"] == "broken_link"

    def test_broken_link_explains_itself(self, file_with_broken_links):
        with h5py.File(file_with_broken_links, "r") as f:
            children = {c["name"]: c for c in tree.build_tree(f)["children"]}

        assert "/nowhere" in children["dangling"]["error"]
        assert "no_such_file.h5" in children["missing_file"]["error"]

    def test_a_broken_link_does_not_break_the_tree(self, file_with_broken_links):
        # The whole point: valid siblings still resolve normally.
        with h5py.File(file_with_broken_links, "r") as f:
            children = {c["name"]: c for c in tree.build_tree(f)["children"]}
        assert children["real"]["shape"] == [10]


class TestErrorsAreVisible:
    """Failures must surface as errors, never as plausible-looking output."""

    def test_bad_export_format_is_an_error_not_a_file(self, client, opened):
        # Panoply shipped a fix for CSV export "failing silently"; the web
        # equivalent is an <a download> saving the JSON error body as the file.
        fid = opened["pandas_table.h5"]
        response = client.get(
            f"/api/files/{fid}/node/export?path=/monthly&format=parquet"
        )
        assert response.status_code == 400
        assert response.json()["error"] == "bad_request"

    def test_oversized_request_explains_itself(self, client, opened, monkeypatch):
        # HDFView's complaint was "open it, or tell me it is too large" —
        # it did neither.
        import h5grid.service as service_module

        monkeypatch.setattr(service_module, "MAX_CELLS_PER_REQUEST", 100)
        fid = opened["plain.h5"]
        response = client.get(
            f"/api/files/{fid}/node/data?path=/matrix&start=0&stop=5000"
        )
        assert response.status_code == 413
        detail = response.json()["detail"]
        assert "limit" in detail.lower()

    def test_group_read_is_an_error_not_an_empty_table(self, client, opened):
        fid = opened["plain.h5"]
        response = client.get(f"/api/files/{fid}/node/data?path=/inputs")
        assert response.status_code == 404
