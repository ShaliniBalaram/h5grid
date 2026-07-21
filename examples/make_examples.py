"""Generate example .h5 files to open in H5Grid.

    python examples/make_examples.py

Writes into examples/:

  example_3d_pywr.h5   A multi-scenario pywr TablesRecorder run. Node arrays are
                       (timesteps, climate, demand) — 3D, because pywr writes one
                       axis per Scenario. Carries /time, /scenarios and the
                       PYWR_ATTRIBUTE / PYWR_TYPE tags, so H5Grid names the axes
                       and the columns.

  example_3d_plain.h5  A plain h5py 3D array with no pywr metadata at all:
                       (day, hour, zone) hourly demand. Shows the generic
                       N-dimensional handling — the dimension selectors fall back
                       to "dim 1" / "dim 2" when nothing names them.

  example_4d.h5        A 4D array, to show that more than one extra dimension
                       gets its own selector.

Open one with:  h5grid examples/example_3d_pywr.h5
"""

from __future__ import annotations

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers blosc for h5py)
import numpy as np
import pandas as pd
import tables

HERE = Path(__file__).parent
RNG = np.random.default_rng(20260722)


def _seasonal(n_steps: int, start: str = "2015-01-01") -> tuple[pd.DatetimeIndex, np.ndarray]:
    """A daily date index and a seasonal signal to build data on top of."""
    dates = pd.date_range(start, periods=n_steps, freq="D")
    day_of_year = dates.dayofyear.to_numpy()
    season = 1.0 + 0.35 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    return dates, season


def _write_time_table(h5: tables.File, dates: pd.DatetimeIndex) -> None:
    class TimeRow(tables.IsDescription):
        index = tables.Int64Col(pos=0)
        year = tables.Int64Col(pos=1)
        month = tables.Int64Col(pos=2)
        day = tables.Int64Col(pos=3)

    table = h5.create_table("/", "time", TimeRow, "Timesteps")
    row = table.row
    for i, d in enumerate(dates):
        row["index"] = i
        row["year"], row["month"], row["day"] = d.year, d.month, d.day
        row.append()
    table.flush()


def make_3d_pywr(path: Path) -> None:
    """Ten years daily, 5 climate futures x 4 demand futures, per node."""
    n_steps = 3653
    climate_size, demand_size = 5, 4
    dates, season = _seasonal(n_steps)
    filters = tables.Filters(complevel=5, complib="blosc")

    # Drier to wetter across climate; lower to higher across demand.
    climate_factor = np.linspace(0.65, 1.35, climate_size).reshape(1, climate_size, 1)
    demand_factor = np.linspace(0.85, 1.30, demand_size).reshape(1, 1, demand_size)
    shape = (n_steps, climate_size, demand_size)

    with tables.open_file(str(path), mode="w", filters=filters) as h5:
        h5.root._v_attrs.PYWR_FORMAT = 1
        h5.root._v_attrs.PYWR_VERSION = 1
        h5.root._v_attrs.title = "Example multi-scenario run"

        _write_time_table(h5, dates)

        class ScenarioRow(tables.IsDescription):
            name = tables.StringCol(1024, pos=0)
            size = tables.Int64Col(pos=1)

        scenarios = h5.create_table("/", "scenarios", ScenarioRow, "Scenarios")
        row = scenarios.row
        for name, size in (("climate", climate_size), ("demand", demand_size)):
            row["name"] = name.encode()
            row["size"] = size
            row.append()
        scenarios.flush()

        base = (season * 100.0).reshape(n_steps, 1, 1)

        nodes = [
            ("reservoir_storage", "volume", "Reservoir", 1.0, 12.0),
            ("river_abstraction", "flow", "Link", 0.45, 6.0),
            ("borehole_supply", "flow", "Input", 0.20, 2.0),
            ("demand_centre", "flow", "Output", 0.60, 4.0),
            ("compensation_release", "flow", "Link", 0.15, 1.0),
        ]
        for name, attribute, node_type, scale, noise in nodes:
            array = h5.create_carray(
                "/", name, tables.Float64Atom(), shape=shape, filters=filters
            )
            values = base * scale * climate_factor * demand_factor
            values = values + RNG.normal(0, noise, shape)
            array[...] = np.clip(values, 0, None)
            array._v_attrs.PYWR_ATTRIBUTE = attribute
            array._v_attrs.PYWR_TYPE = node_type

        # A control-curve index parameter, which pywr stores as int32.
        index_array = h5.create_carray(
            "/", "drought_trigger", tables.Int32Atom(), shape=shape, filters=filters
        )
        storage = np.clip(base * climate_factor * demand_factor, 0, None)
        index_array[...] = np.digitize(
            storage, [40.0, 70.0, 100.0]
        ).astype("int32")
        index_array._v_attrs.PYWR_ATTRIBUTE = "parameter-index"
        index_array._v_attrs.PYWR_TYPE = "ControlCurveIndexParameter"


def make_3d_plain(path: Path) -> None:
    """(day, hour, zone) hourly demand — no pywr metadata anywhere."""
    n_days, n_hours, n_zones = 365, 24, 12
    _, season = _seasonal(n_days)

    # Two peaks a day, scaled by season and by zone size.
    hours = np.arange(n_hours)
    diurnal = (
        1.0
        + 0.45 * np.exp(-0.5 * ((hours - 8) / 2.0) ** 2)
        + 0.55 * np.exp(-0.5 * ((hours - 19) / 2.5) ** 2)
    ).reshape(1, n_hours, 1)
    zone_scale = np.linspace(0.5, 3.0, n_zones).reshape(1, 1, n_zones)

    values = (
        season.reshape(n_days, 1, 1) * diurnal * zone_scale * 40.0
        + RNG.normal(0, 1.5, (n_days, n_hours, n_zones))
    )

    with h5py.File(path, "w") as f:
        f.attrs["title"] = "Hourly demand by zone"
        f.attrs["description"] = (
            "Plain 3D array: dimension 0 is the day, 1 is the hour of day, "
            "2 is the supply zone. No naming metadata, so H5Grid falls back to "
            "positional dimension selectors."
        )
        dataset = f.create_dataset(
            "hourly_demand",
            data=np.clip(values, 0, None),
            chunks=(64, n_hours, n_zones),
            compression="gzip",
            compression_opts=4,
        )
        dataset.attrs["units"] = "Ml/d"
        dataset.attrs["dimensions"] = "day, hour, zone"

        # A matching date column so the day axis is interpretable.
        dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
        f.create_dataset("date", data=dates.astype("int64").to_numpy())

        # Zone names, as a plain vlen-string dataset.
        f.create_dataset(
            "zone_names",
            data=np.array([f"Zone {chr(65 + i)}" for i in range(n_zones)], dtype=object),
            dtype=h5py.string_dtype(),
        )


def make_4d(path: Path) -> None:
    """(timestep, member, lead_time, site) — two pinned dimensions to pick."""
    n_steps, n_members, n_leads, n_sites = 500, 8, 6, 4
    values = (
        RNG.gamma(2.0, 20.0, (n_steps, 1, 1, 1))
        * np.linspace(0.8, 1.2, n_members).reshape(1, n_members, 1, 1)
        * np.linspace(1.0, 0.6, n_leads).reshape(1, 1, n_leads, 1)
        * np.linspace(0.5, 2.0, n_sites).reshape(1, 1, 1, n_sites)
    )

    with h5py.File(path, "w") as f:
        f.attrs["title"] = "Ensemble inflow forecast"
        f.attrs["description"] = (
            "4D: timestep x ensemble member x lead time x site. Two dimensions "
            "beyond rows and columns, so two selectors appear above the grid."
        )
        d = f.create_dataset("forecast_inflow", data=values, compression="gzip")
        d.attrs["units"] = "m3/s"
        d.attrs["dimensions"] = "timestep, member, lead_time, site"


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    targets = [
        ("example_3d_pywr.h5", make_3d_pywr),
        ("example_3d_plain.h5", make_3d_plain),
        ("example_4d.h5", make_4d),
    ]
    for name, builder in targets:
        path = HERE / name
        print(f"writing {name} ...", end=" ", flush=True)
        builder(path)
        print(f"ok ({path.stat().st_size / 1e6:.1f} MB)")

    print(f"\nexamples in {HERE}")
    print("open one with:  h5grid examples/example_3d_pywr.h5")


if __name__ == "__main__":
    main()
