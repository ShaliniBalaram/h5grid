"""Endpoint behaviour: payload shapes, guards, and error codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestOpenAndTree:
    def test_open_returns_identity(self, client, fixture_files):
        response = client.post(
            "/api/files/open", json={"path": str(fixture_files / "plain.h5")}
        )
        assert response.status_code == 200
        body = response.json()
        assert set(body) >= {"file_id", "path", "size_bytes", "mtime"}
        assert body["path"].endswith("plain.h5")

    def test_open_missing_file_is_404(self, client):
        response = client.post("/api/files/open", json={"path": "/nope/missing.h5"})
        assert response.status_code == 404
        assert response.json()["error"] == "not_found"

    def test_open_non_hdf5_is_400(self, client, tmp_path):
        junk = tmp_path / "notes.txt"
        junk.write_text("this is not HDF5")
        response = client.post("/api/files/open", json={"path": str(junk)})
        assert response.status_code == 400

    def test_reopening_the_same_file_is_idempotent(self, client, fixture_files):
        path = str(fixture_files / "plain.h5")
        first = client.post("/api/files/open", json={"path": path}).json()
        second = client.post("/api/files/open", json={"path": path}).json()
        assert first["file_id"] == second["file_id"]

    def test_tree_shape(self, client, opened):
        response = client.get(f"/api/files/{opened['plain.h5']}/tree")
        assert response.status_code == 200
        root = response.json()
        assert root["kind"] == "group"
        names = {c["name"] for c in root["children"]}
        assert {"matrix", "vector", "inputs"} <= names

    def test_tree_hides_pandas_internals(self, client, opened):
        root = client.get(f"/api/files/{opened['pandas_fixed.h5']}/tree").json()
        monthly = next(c for c in root["children"] if c["name"] == "monthly")
        assert monthly["kind"] == "pandas_frame"
        assert monthly["children"] == []
        assert monthly["shape"] == [3653, 5]

    def test_tree_raw_reveals_internals(self, client, opened):
        root = client.get(f"/api/files/{opened['pandas_fixed.h5']}/tree?raw=true").json()
        monthly = next(c for c in root["children"] if c["name"] == "monthly")
        assert {c["name"] for c in monthly["children"]} >= {"axis0", "block0_values"}

    def test_unknown_file_id_is_404(self, client):
        assert client.get("/api/files/deadbeef/tree").status_code == 404


class TestMeta:
    def test_dataset_meta(self, client, opened):
        fid = opened["plain.h5"]
        body = client.get(f"/api/files/{fid}/node/meta?path=/matrix").json()
        assert body["shape"] == [5000, 12]
        assert body["dtype"] == "float64"
        assert body["chunks"] == [500, 12]
        assert body["nrows"] == 5000
        assert len(body["columns"]) == 12
        assert body["columns"][0]["name"] == "node_0"

    def test_attrs_included_and_json_safe(self, client, opened):
        fid = opened["plain.h5"]
        body = client.get(f"/api/files/{fid}/node/meta?path=/attributed").json()
        assert body["attrs"]["unicode"] == "unicode éè 中文"
        assert body["attrs"]["nan_attr"] is None
        json.dumps(body, allow_nan=False)

    def test_time_index_flag(self, client, opened):
        fid = opened["pywr_style.h5"]
        assert (
            client.get(f"/api/files/{fid}/node/meta?path=/reservoir").json()[
                "time_index_available"
            ]
            is True
        )
        assert (
            client.get(f"/api/files/{fid}/node/meta?path=/monthly_summary").json()[
                "time_index_available"
            ]
            is False
        )

    def test_meta_on_a_group_is_404(self, client, opened):
        fid = opened["plain.h5"]
        assert client.get(f"/api/files/{fid}/node/meta?path=/inputs").status_code == 404


class TestData:
    def test_payload_shape(self, client, opened):
        fid = opened["plain.h5"]
        body = client.get(
            f"/api/files/{fid}/node/data?path=/matrix&start=0&stop=10&cols=0:3"
        ).json()
        assert body["start"] == 0
        assert body["stop"] == 10
        assert body["total_rows"] == 5000
        assert body["columns"] == ["node_0", "node_1", "node_2"]
        assert len(body["rows"]) == 10
        assert all(len(row) == 3 for row in body["rows"])

    def test_nan_is_null_and_inf_is_a_string(self, client, opened):
        fid = opened["plain.h5"]
        raw = client.get(
            f"/api/files/{fid}/node/data?path=/matrix&start=10&stop=13&cols=3:4"
        ).text
        # The response body must be valid JSON with no bare NaN/Infinity tokens.
        body = json.loads(raw)
        assert body["rows"] == [[None], ["Infinity"], ["-Infinity"]]
        assert "NaN" not in raw

    def test_cell_cap_is_enforced(self, client, opened, monkeypatch):
        import h5grid.service as service_module

        assert service_module.MAX_CELLS_PER_REQUEST == 200_000
        monkeypatch.setattr(service_module, "MAX_CELLS_PER_REQUEST", 100)

        fid = opened["plain.h5"]
        response = client.get(
            f"/api/files/{fid}/node/data?path=/matrix&start=0&stop=5000"
        )
        assert response.status_code == 413
        assert response.json()["error"] == "request_too_large"

    def test_request_within_the_cap_succeeds(self, client, opened):
        fid = opened["plain.h5"]
        # 5000 rows x 12 columns = 60,000 cells, comfortably under the limit.
        response = client.get(
            f"/api/files/{fid}/node/data?path=/matrix&start=0&stop=5000"
        )
        assert response.status_code == 200
        assert len(response.json()["rows"]) == 5000

    def test_pandas_table_paging(self, client, opened):
        fid = opened["pandas_table.h5"]
        body = client.get(
            f"/api/files/{fid}/node/data?path=/monthly&start=100&stop=103"
        ).json()
        assert body["columns"][0] == "date"
        assert body["rows"][0][0] == "1975-04-11"

    def test_pandas_fixed_paging(self, client, opened):
        fid = opened["pandas_fixed.h5"]
        body = client.get(
            f"/api/files/{fid}/node/data?path=/monthly&start=0&stop=3"
        ).json()
        assert body["rows"][0][0] == "1975-01-01"
        assert body["total_rows"] == 3653

    def test_time_index_toggle(self, client, opened):
        fid = opened["pywr_style.h5"]
        url = f"/api/files/{fid}/node/data?path=/reservoir&stop=3&cols=0:2"
        without = client.get(url).json()
        with_dates = client.get(url + "&use_time_index=true").json()
        assert without["columns"] == ["climate[0]", "climate[1]"]
        assert with_dates["columns"] == ["date", "climate[0]", "climate[1]"]
        assert with_dates["rows"][0][0] == "1975-01-01"

    def test_3d_slice_parameter(self, client, opened):
        fid = opened["plain.h5"]
        first = client.get(
            f"/api/files/{fid}/node/data?path=/cube&stop=2&cols=0:3&slice=,,0"
        ).json()
        third = client.get(
            f"/api/files/{fid}/node/data?path=/cube&stop=2&cols=0:3&slice=,,2"
        ).json()
        assert first["rows"] != third["rows"]

    def test_bad_slice_is_400(self, client, opened):
        fid = opened["plain.h5"]
        response = client.get(
            f"/api/files/{fid}/node/data?path=/cube&stop=2&slice=0,,"
        )
        assert response.status_code == 400

    def test_stop_beyond_end_is_clamped(self, client, opened):
        fid = opened["plain.h5"]
        body = client.get(
            f"/api/files/{fid}/node/data?path=/vector&start=995&stop=99999"
        ).json()
        assert body["stop"] == 1000
        assert len(body["rows"]) == 5


class TestStatsEndpoint:
    def test_stats(self, client, opened):
        fid = opened["plain.h5"]
        body = client.get(
            f"/api/files/{fid}/node/stats?path=/matrix&col=node_0"
        ).json()
        assert set(body) >= {"min", "max", "mean", "std", "nan_count", "count"}
        assert body["count"] == 5000

    def test_unknown_column_is_404(self, client, opened):
        fid = opened["plain.h5"]
        response = client.get(f"/api/files/{fid}/node/stats?path=/matrix&col=zzz")
        assert response.status_code == 404

    def test_result_is_cached(self, client, opened):
        fid = opened["plain.h5"]
        url = f"/api/files/{fid}/node/stats?path=/matrix&col=node_1"
        assert client.get(url).json() == client.get(url).json()


class TestSearchEndpoint:
    def test_search_returns_rows(self, client, opened):
        fid = opened["plain.h5"]
        body = client.get(
            f"/api/files/{fid}/node/search?path=/matrix&col=node_0&q=>180&limit=5"
        ).json()
        assert body["rows"] == sorted(body["rows"])
        assert len(body["rows"]) <= 5


class TestPlotData:
    def test_series_returned(self, client, opened):
        fid = opened["pywr_style.h5"]
        body = client.get(
            f"/api/files/{fid}/node/plotdata?path=/reservoir&cols=0,1&max_points=200"
        ).json()
        assert [s["name"] for s in body["series"]] == ["climate[0]", "climate[1]"]
        assert len(body["x"]) == len(body["series"][0]["y"])
        assert len(body["x"]) <= 200 + 2

    def test_x_axis_uses_dates_when_available(self, client, opened):
        fid = opened["pywr_style.h5"]
        body = client.get(
            f"/api/files/{fid}/node/plotdata?path=/reservoir&cols=0&max_points=100"
        ).json()
        assert body["x_is_date"] is True
        assert body["x"][0] == "1975-01-01"

    def test_spikes_survive_decimation(self, client, opened):
        # A min-max decimator must keep the extremes of the full series.
        fid = opened["pywr_style.h5"]
        plot = client.get(
            f"/api/files/{fid}/node/plotdata?path=/reservoir&cols=0&max_points=100"
        ).json()
        stats_body = client.get(
            f"/api/files/{fid}/node/stats?path=/reservoir&col=climate%5B0%5D"
        ).json()
        values = [v for v in plot["series"][0]["y"] if isinstance(v, (int, float))]
        assert min(values) == pytest.approx(stats_body["min"])
        assert max(values) == pytest.approx(stats_body["max"])

    def test_pandas_frame_uses_its_own_datetime_index(self, client, opened):
        # A pandas frame has no /time table; its dates live in the index column.
        # Without this the grid showed dates while the plot showed row numbers.
        fid = opened["pandas_fixed.h5"]
        body = client.get(
            f"/api/files/{fid}/node/plotdata?path=/monthly&cols=1,2&max_points=40"
        ).json()
        assert body["x_is_date"] is True
        assert body["x"][0] == "1975-01-01"
        assert body["x"][-1] == "1984-12-31"
        assert [s["name"] for s in body["series"]] == ["flow", "level"]

    def test_x_is_row_numbers_when_there_is_no_date_index(self, client, opened):
        fid = opened["plain.h5"]
        body = client.get(
            f"/api/files/{fid}/node/plotdata?path=/matrix&cols=0&max_points=40"
        ).json()
        assert body["x_is_date"] is False
        assert body["x"][0] == 0
        assert all(isinstance(v, int) for v in body["x"])

    def test_x_length_matches_every_series(self, client, opened):
        fid = opened["pandas_fixed.h5"]
        body = client.get(
            f"/api/files/{fid}/node/plotdata?path=/monthly&cols=1,2,3&max_points=200"
        ).json()
        for series in body["series"]:
            assert len(series["y"]) == len(body["x"])

    def test_zoom_window_returns_finer_data(self, client, opened):
        # The point of re-requesting on zoom: a narrow window comes back
        # undecimated, so zooming in reveals real rows instead of magnifying
        # the points already drawn.
        fid = opened["pywr_style.h5"]
        base = f"/api/files/{fid}/node/plotdata?path=/reservoir&cols=0&max_points=200"

        whole = client.get(base).json()
        assert whole["decimated"] is True
        assert whole["bucket_size"] > 1
        assert whole["window_rows"] == whole["total_rows"]

        zoomed = client.get(base + "&start=1000&stop=1100").json()
        assert zoomed["decimated"] is False
        assert zoomed["bucket_size"] == 1
        assert zoomed["window_rows"] == 100
        assert zoomed["total_rows"] == whole["total_rows"]

    def test_window_reports_its_own_bounds(self, client, opened):
        fid = opened["pywr_style.h5"]
        body = client.get(
            f"/api/files/{fid}/node/plotdata?path=/reservoir&cols=0&start=500&stop=700"
        ).json()
        assert body["start"] == 500
        assert body["stop"] == 700
        assert min(body["rows"]) >= 500
        assert max(body["rows"]) < 700

    def test_row_positions_align_with_points(self, client, opened):
        # The client maps a drag selection back to rows through this array, so a
        # length mismatch would silently zoom to the wrong place.
        fid = opened["pywr_style.h5"]
        body = client.get(
            f"/api/files/{fid}/node/plotdata?path=/reservoir&cols=0,1&max_points=120"
        ).json()
        assert len(body["rows"]) == len(body["x"])
        assert body["rows"] == sorted(body["rows"])
        for series in body["series"]:
            assert len(series["y"]) == len(body["rows"])

    def test_window_dates_track_the_time_index(self, client, opened):
        fid = opened["pywr_style.h5"]
        body = client.get(
            f"/api/files/{fid}/node/plotdata?path=/reservoir&cols=0"
            "&use_time_index=true&start=365&stop=395"
        ).json()
        assert body["x_is_date"] is True
        assert body["x"][0] == "1976-01-01"

    def test_no_decimation_for_short_series(self, client, opened):
        fid = opened["pywr_style.h5"]
        body = client.get(
            f"/api/files/{fid}/node/plotdata?path=/monthly_summary&cols=0&max_points=4000"
        ).json()
        assert body["decimated"] is False


class TestExport:
    def test_csv_download(self, client, opened):
        fid = opened["pandas_table.h5"]
        response = client.get(
            f"/api/files/{fid}/node/export?path=/monthly&format=csv&start=0&stop=5"
        )
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "attachment" in response.headers["content-disposition"]

        lines = response.text.strip().splitlines()
        assert lines[0].split(",")[:2] == ["date", "flow"]
        assert len(lines) == 6  # header plus five rows
        assert lines[1].startswith("1975-01-01")

    def test_csv_writes_nan_as_empty(self, client, opened):
        fid = opened["pandas_table.h5"]
        response = client.get(
            f"/api/files/{fid}/node/export?path=/monthly&format=csv&start=5&stop=7"
        )
        # rows 5-9 of `flow` are NaN in the fixture
        assert ",," in response.text.splitlines()[1]

    def test_xlsx_download(self, client, opened):
        fid = opened["pandas_table.h5"]
        response = client.get(
            f"/api/files/{fid}/node/export?path=/monthly&format=xlsx&start=0&stop=20"
        )
        assert response.status_code == 200
        assert response.content[:2] == b"PK"  # a zip container, i.e. real xlsx

    def test_xlsx_refused_above_the_row_limit(self, client, opened, monkeypatch):
        import h5grid.export as export_module

        monkeypatch.setattr(export_module, "XLSX_ROW_LIMIT", 10)
        fid = opened["pandas_table.h5"]
        response = client.get(
            f"/api/files/{fid}/node/export?path=/monthly&format=xlsx"
        )
        assert response.status_code == 413
        assert "CSV" in response.json()["detail"]

    def test_export_honours_the_time_index(self, client, opened):
        fid = opened["pywr_style.h5"]
        response = client.get(
            f"/api/files/{fid}/node/export?path=/reservoir&format=csv"
            "&start=0&stop=3&cols=0:2&use_time_index=true"
        )
        lines = response.text.strip().splitlines()
        assert lines[0] == "date,climate[0],climate[1]"
        assert lines[1].startswith("1975-01-01")

    def test_unknown_format_is_400(self, client, opened):
        fid = opened["pandas_table.h5"]
        response = client.get(
            f"/api/files/{fid}/node/export?path=/monthly&format=parquet"
        )
        assert response.status_code == 400


class TestFileChanged:
    def test_modified_file_yields_409(self, client, tmp_path):
        import h5py
        import numpy as np

        path = tmp_path / "live.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=np.arange(100, dtype="f8"))

        fid = client.post("/api/files/open", json={"path": str(path)}).json()["file_id"]
        assert client.get(f"/api/files/{fid}/tree").status_code == 200

        # Simulate a model run replacing its output while the file is open.
        # Written aside and moved into place: HDF5 refuses to truncate a file
        # this process already holds open, and atomic replace is what a careful
        # writer does anyway.
        replacement = tmp_path / "next.h5"
        with h5py.File(replacement, "w") as f:
            f.create_dataset("data", data=np.arange(200, dtype="f8"))
        replacement.replace(path)

        response = client.get(f"/api/files/{fid}/tree")
        assert response.status_code == 409
        assert response.json()["error"] == "file_changed"

    def test_reopening_after_a_change_gives_a_new_id(self, client, tmp_path):
        import h5py
        import numpy as np

        path = tmp_path / "live.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=np.arange(10, dtype="f8"))
        first = client.post("/api/files/open", json={"path": str(path)}).json()["file_id"]

        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=np.arange(20, dtype="f8"))
        second = client.post("/api/files/open", json={"path": str(path)}).json()["file_id"]

        assert first != second
        assert client.get(f"/api/files/{second}/tree").status_code == 200


class TestBrowseAndClose:
    def test_browse_lists_h5_files(self, client, fixture_files):
        body = client.get(f"/api/browse?dir={fixture_files}").json()
        names = {e["name"]: e for e in body["entries"]}
        assert names["plain.h5"]["is_h5"] is True
        assert names["plain.h5"]["is_dir"] is False
        assert body["parent"] is not None

    def test_browse_missing_directory_is_404(self, client):
        assert client.get("/api/browse?dir=/nope/nowhere").status_code == 404

    def test_browse_returns_breadcrumbs(self, client, fixture_files):
        body = client.get(f"/api/browse?dir={fixture_files}").json()
        crumbs = body["breadcrumbs"]
        assert crumbs[0]["name"] == "/"
        assert crumbs[0]["path"] == "/"
        # The last crumb is where we are, and every crumb is a real ancestor.
        assert crumbs[-1]["path"] == body["dir"]
        assert crumbs[-1]["name"] == "fixtures"
        for earlier, later in zip(crumbs, crumbs[1:]):
            assert later["path"].startswith(earlier["path"])

    def test_browse_roots_include_home_and_mounted_volumes(self, client):
        roots = client.get("/api/browse/roots").json()["roots"]
        kinds = {r["kind"] for r in roots}
        assert "home" in kinds
        assert "root" in kinds
        assert all(Path(r["path"]).is_dir() for r in roots)
        # No duplicates: the same directory must not appear twice.
        paths = [r["path"] for r in roots]
        assert len(paths) == len(set(paths))

    def test_browse_roots_reach_external_drives(self, client, tmp_path):
        # On this machine the working tree itself lives on a mounted volume, so
        # a volume entry should be an ancestor of somewhere real. Skip cleanly
        # where no volumes are mounted.
        roots = client.get("/api/browse/roots").json()["roots"]
        volumes = [r for r in roots if r["kind"] == "volume"]
        if not volumes:
            pytest.skip("no mounted volumes on this machine")
        for volume in volumes:
            assert client.get(f"/api/browse?dir={volume['path']}").status_code == 200

    def test_close(self, client, opened):
        fid = opened["plain.h5"]
        assert client.post(f"/api/files/{fid}/close").json()["closed"] is True
        assert client.get(f"/api/files/{fid}/tree").status_code == 404

    def test_health_needs_no_token(self, auth_client):
        assert auth_client.get("/api/health").status_code == 200
