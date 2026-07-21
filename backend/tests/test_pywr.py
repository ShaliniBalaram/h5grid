"""Decoding pywr's TablesRecorder structure: scenarios, axes, node metadata."""

from __future__ import annotations

import pytest

from h5grid import pywr, service


@pytest.fixture
def scenarios_file(registry, fixture_files):
    return registry.open(fixture_files / "pywr_scenarios.h5")


@pytest.fixture
def combinations_file(registry, fixture_files):
    return registry.open(fixture_files / "pywr_combinations.h5")


class TestReadingScenarios:
    def test_scenarios_read_in_axis_order(self, scenarios_file):
        layout = scenarios_file.scenario_layout()
        assert [(s.name, s.size) for s in layout.scenarios] == [
            ("climate", 3),
            ("demand", 4),
        ]

    def test_layout_cached(self, scenarios_file):
        assert scenarios_file.scenario_layout() is scenarios_file.scenario_layout()

    def test_absent_for_non_pywr_files(self, plain, pandas_table):
        assert plain.scenario_layout() is None
        assert pandas_table.scenario_layout() is None

    def test_shape_recognised(self, scenarios_file):
        layout = scenarios_file.scenario_layout()
        assert layout.describes((1096, 3, 4)) is True
        # A dataset whose trailing dims are not the scenario sizes is something
        # else and must not be labelled as scenarios.
        assert layout.describes((1096, 5, 4)) is False
        assert layout.describes((1096, 3)) is False

    def test_axis_names(self, scenarios_file):
        layout = scenarios_file.scenario_layout()
        assert layout.axis_name(1) == "climate"
        assert layout.axis_name(2) == "demand"
        assert layout.axis_name(0) is None
        assert layout.axis_name(9) is None


class TestScenarioColumnLabels:
    def test_default_view_labels_the_first_scenario(self, scenarios_file):
        payload = service.node_data(
            scenarios_file, "/reservoir", stop=2, use_time_index=True
        )
        assert payload["columns"] == [
            "date",
            "climate[0]",
            "climate[1]",
            "climate[2]",
        ]

    def test_switching_the_free_axis_relabels(self, scenarios_file):
        payload = service.node_data(
            scenarios_file, "/reservoir", stop=2, dim_slice_spec=",0,"
        )
        assert payload["columns"] == [
            "demand[0]",
            "demand[1]",
            "demand[2]",
            "demand[3]",
        ]

    def test_values_follow_the_selected_axis(self, scenarios_file):
        # The fixture scales by climate then demand, so both axes must vary and
        # the labelling must match the data actually returned.
        by_climate = service.node_data(scenarios_file, "/reservoir", stop=1)
        by_demand = service.node_data(
            scenarios_file, "/reservoir", stop=1, dim_slice_spec=",0,"
        )
        assert by_climate["rows"][0][0] == pytest.approx(by_demand["rows"][0][0])
        assert by_climate["rows"][0] != by_demand["rows"][0]

    def test_dimension_names_exposed_in_meta(self, scenarios_file):
        meta = service.node_meta(scenarios_file, "/reservoir")
        assert meta["dim_names"] == [None, "climate", "demand"]
        assert meta["shape"] == [1096, 3, 4]

    def test_single_scenario_file_still_labelled(self, pywr):
        # pywr_style.h5 has one scenario, "climate" of size 20, over a 2D array.
        payload = service.node_data(pywr, "/reservoir", stop=1, cols_spec="0:3")
        assert payload["columns"] == ["climate[0]", "climate[1]", "climate[2]"]

    def test_non_matching_dataset_keeps_positional_names(self, pywr):
        # /monthly_summary is (120, 4) — not a scenario array.
        payload = service.node_data(pywr, "/monthly_summary", stop=1)
        assert payload["columns"] == ["col_0", "col_1", "col_2", "col_3"]

    def test_non_pywr_files_unaffected(self, plain):
        payload = service.node_data(plain, "/matrix", stop=1, cols_spec="0:3")
        assert payload["columns"] == ["node_0", "node_1", "node_2"]


class TestScenarioCombinations:
    def test_combinations_read(self, combinations_file):
        layout = combinations_file.scenario_layout()
        assert layout.is_collapsed() is True
        assert layout.combinations == [(0, 0), (0, 2), (1, 1), (2, 0), (2, 3)]

    def test_collapsed_axis_labelled_by_combination(self, combinations_file):
        payload = service.node_data(
            combinations_file, "/reservoir", stop=1, use_time_index=True
        )
        assert payload["columns"] == [
            "date",
            "climate=0, demand=0",
            "climate=0, demand=2",
            "climate=1, demand=1",
            "climate=2, demand=0",
            "climate=2, demand=3",
        ]

    def test_shape_recognition_uses_combination_count(self, combinations_file):
        layout = combinations_file.scenario_layout()
        assert layout.describes((365, 5)) is True
        assert layout.describes((365, 3, 4)) is False

    def test_meta_reports_collapsed(self, combinations_file):
        meta = service.node_meta(combinations_file, "/reservoir")
        assert meta["scenarios"]["collapsed"] is True
        assert meta["scenarios"]["combination_count"] == 5


class TestNodeMetadata:
    def test_modern_attribute_names(self, scenarios_file):
        meta = service.node_meta(scenarios_file, "/reservoir")
        assert meta["pywr"] == {"attribute": "volume", "type": "Reservoir"}

    def test_legacy_attribute_names(self, scenarios_file):
        # Older pywr wrote lower-case hyphenated attributes; both must work.
        meta = service.node_meta(scenarios_file, "/legacy_node")
        assert meta["pywr"] == {"attribute": "flow", "type": "Input"}

    def test_index_parameter_recorded_as_int(self, scenarios_file):
        meta = service.node_meta(scenarios_file, "/control_curve_index")
        assert meta["pywr"]["attribute"] == "parameter-index"
        assert meta["dtype"] == "int32"

    def test_absent_on_non_pywr_nodes(self, plain):
        assert service.node_meta(plain, "/matrix")["pywr"] is None

    def test_exposed_in_the_tree(self, scenarios_file):
        tree = service.tree_payload(scenarios_file)
        nodes = {c["name"]: c for c in tree["children"]}
        assert nodes["reservoir"]["pywr"] == {
            "attribute": "volume",
            "type": "Reservoir",
        }
        assert nodes["time"]["pywr"] is None


class TestTimeIndexWithScenarios:
    def test_time_index_still_applies_to_3d_arrays(self, scenarios_file):
        payload = service.node_data(
            scenarios_file, "/reservoir", stop=3, use_time_index=True
        )
        assert payload["rows"][0][0] == "2020-01-01"
        assert payload["rows"][2][0] == "2020-01-03"

    def test_stats_work_on_a_named_scenario_column(self, scenarios_file):
        from h5grid import stats

        reader = scenarios_file.reader("/reservoir")
        result = stats.column_stats(reader, "climate[1]")
        assert result["count"] == 1096
        assert result["min"] > 0


class TestHelpers:
    def test_is_pywr_file(self, scenarios_file, plain):
        assert pywr.is_pywr_file(scenarios_file.h5()) is True
        assert pywr.is_pywr_file(plain.h5()) is False

    def test_layout_json_shape(self, scenarios_file):
        payload = scenarios_file.scenario_layout().to_json()
        assert payload["scenarios"] == [
            {"name": "climate", "size": 3},
            {"name": "demand", "size": 4},
        ]
        assert payload["collapsed"] is False
