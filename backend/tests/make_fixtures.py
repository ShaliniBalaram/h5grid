"""Generate the HDF5 fixture files the test suite runs against.

Usage:
    python make_fixtures.py [--big]

Creates, in ./fixtures/ next to this script:

  plain.h5         h5py-native: 1D/2D/3D floats, compound dtype, vlen strings,
                   nested groups, attributes of every scalar type.
  pandas_fixed.h5  DataFrame.to_hdf(format='fixed') with a DatetimeIndex.
  pandas_table.h5  the same data with format='table' (row-sliceable).
  pywr_style.h5    PyTables/TablesRecorder-style output: root /time table plus
                   (timesteps, scenarios) CArrays, blosc-compressed.
  pywr_scenarios.h5     two named Scenarios, so node arrays are 3D:
                        (timesteps, climate, demand), plus PYWR_* attributes.
  pywr_combinations.h5  explicit scenario combinations, so arrays collapse to
                        (timesteps, N) and /scenario_combinations names them.
  big.h5           20M x 10 chunked floats, only with --big (gitignored).

These files intentionally cover the awkward cases: NaN and Inf, an all-NaN
column, byte strings, an int64 nanosecond-epoch column that must be decoded as
dates, a 3D array needing a slice selector, and a compound dtype.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers blosc et al. for h5py)
import numpy as np
import pandas as pd
import tables

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Fixed seed: fixtures must be byte-stable so tests can assert on values.
RNG = np.random.default_rng(20240501)

N_TIMESTEPS = 3653  # 1975-01-01 .. 1984-12-31, ten years of daily data
N_SCENARIOS = 20


def make_plain(path: Path) -> None:
    """h5py-native structures, one of every shape and dtype worth testing."""
    with h5py.File(path, "w") as f:
        f.attrs["title"] = "Plain HDF5 fixture"
        f.attrs["created_by"] = "make_fixtures.py"

        # --- 1D: renders as a single column named "value" -------------------
        d1 = f.create_dataset("vector", data=np.arange(1000, dtype="f8") * 1.5)
        d1.attrs["units"] = "Ml/d"
        d1.attrs["description"] = "A simple 1D float vector"

        # --- 2D: rows x columns, with NaN and Inf to exercise JSON coercion --
        arr2d = RNG.normal(100.0, 25.0, size=(5000, 12)).astype("f8")
        arr2d[10, 3] = np.nan
        arr2d[11, 3] = np.inf
        arr2d[12, 3] = -np.inf
        arr2d[:, 11] = np.nan  # an entirely-NaN column
        d2 = f.create_dataset("matrix", data=arr2d, chunks=(500, 12))
        d2.attrs["units"] = "m3/s"
        d2.attrs["column_names"] = np.array(
            [f"node_{i}" for i in range(12)], dtype=h5py.string_dtype()
        )

        # --- 3D: needs a slice selector for the extra dimension -------------
        arr3d = RNG.normal(50.0, 5.0, size=(365, 8, 4)).astype("f4")
        f.create_dataset("cube", data=arr3d, chunks=(365, 8, 1))

        # --- compound dtype: renders as a multi-column table -----------------
        compound_dt = np.dtype(
            [
                ("id", "i4"),
                ("name", "S16"),
                ("flow", "f8"),
                ("active", "?"),
            ]
        )
        rec = np.zeros(500, dtype=compound_dt)
        rec["id"] = np.arange(500)
        rec["name"] = [f"reservoir_{i}".encode() for i in range(500)]
        rec["flow"] = RNG.uniform(0, 500, 500)
        rec["active"] = RNG.integers(0, 2, 500).astype(bool)
        f.create_dataset("records", data=rec)

        # --- variable-length UTF-8 strings ----------------------------------
        labels = np.array(
            ["Rutland Water", "Grafham", "Pitsford", "Ravensthorpe", "Naseby"] * 20,
            dtype=object,
        )
        f.create_dataset("labels", data=labels, dtype=h5py.string_dtype())

        # --- an int64 nanosecond-epoch column that must decode as dates -----
        epoch_ns = (
            pd.date_range("1975-01-01", periods=1000, freq="D").astype("int64").to_numpy()
        )
        f.create_dataset("timestamps", data=epoch_ns)

        # --- fixed-length byte strings --------------------------------------
        f.create_dataset(
            "byte_labels", data=np.array([b"AAA", b"BBB", b"CCC"] * 10, dtype="S3")
        )

        # --- nested groups ---------------------------------------------------
        inputs = f.create_group("inputs")
        inputs.attrs["source"] = "Environment Agency"
        catchment = inputs.create_group("catchment_a")
        catchment.create_dataset("rainfall", data=RNG.gamma(2.0, 3.0, (N_TIMESTEPS, 3)))
        catchment.create_dataset("pet", data=RNG.uniform(0, 5, (N_TIMESTEPS, 3)))
        deep = inputs.create_group("nested/deeper/deepest")
        deep.create_dataset("leaf", data=np.arange(10, dtype="i8"))

        # --- attributes of every scalar type, on one node -------------------
        a = f.create_dataset("attributed", data=np.zeros(5))
        a.attrs["int8"] = np.int8(-8)
        a.attrs["int32"] = np.int32(32)
        a.attrs["int64"] = np.int64(2**40)
        a.attrs["uint16"] = np.uint16(65535)
        a.attrs["float32"] = np.float32(3.5)
        a.attrs["float64"] = np.float64(np.pi)
        a.attrs["bool_true"] = np.bool_(True)
        a.attrs["bytes"] = np.bytes_(b"raw bytes")
        a.attrs["unicode"] = "unicode éè 中文"
        a.attrs["int_array"] = np.arange(5, dtype="i4")
        a.attrs["float_array"] = np.linspace(0, 1, 4)
        a.attrs["nan_attr"] = np.float64(np.nan)
        a.attrs["inf_attr"] = np.float64(np.inf)
        a.attrs["empty_str"] = ""
        a.attrs["long_text"] = "x" * 500  # inspector must truncate this


def _sample_frame() -> pd.DataFrame:
    """The DataFrame shared by the two pandas fixtures."""
    idx = pd.date_range("1975-01-01", periods=N_TIMESTEPS, freq="D", name="date")
    df = pd.DataFrame(
        {
            "flow": RNG.gamma(2.0, 50.0, N_TIMESTEPS),
            "level": RNG.normal(25.0, 2.0, N_TIMESTEPS),
            "demand": RNG.uniform(10.0, 90.0, N_TIMESTEPS),
            "spill": np.zeros(N_TIMESTEPS),
            "count": RNG.integers(0, 100, N_TIMESTEPS).astype("int64"),
        },
        index=idx,
    )
    df.loc[df.index[5:10], "flow"] = np.nan
    return df


def make_pandas_fixed(path: Path) -> None:
    """format='fixed' — cannot be row-sliced by pandas, exercises the guard."""
    df = _sample_frame()
    df.to_hdf(path, key="monthly", format="fixed", mode="w")
    # A second key so the tree has more than one logical table.
    df.head(500).to_hdf(path, key="subset", format="fixed", mode="a")
    # A frame with a plain RangeIndex, no dates involved.
    pd.DataFrame({"a": np.arange(100), "b": np.arange(100) * 2.5}).to_hdf(
        path, key="plain_index", format="fixed", mode="a"
    )


def make_pandas_table(path: Path) -> None:
    """format='table' — supports true lazy row slicing via HDFStore.select."""
    df = _sample_frame()
    df.to_hdf(path, key="monthly", format="table", mode="w", data_columns=True)
    df.head(500).to_hdf(path, key="subset", format="table", mode="a", data_columns=True)
    # A frame nested under a group path, to check tree traversal of table nodes.
    df.head(100).to_hdf(path, key="/results/deep/nested", format="table", mode="a")


def make_pywr_style(path: Path) -> None:
    """Mimic pywr's TablesRecorder output: /time table + per-node CArrays.

    Uses PyTables with blosc compression, exactly as pywr does, so the fixture
    also proves hdf5plugin lets h5py read the compressed chunks back.
    """
    dates = pd.date_range("1975-01-01", periods=N_TIMESTEPS, freq="D")
    filters = tables.Filters(complevel=5, complib="blosc")

    with tables.open_file(str(path), mode="w", filters=filters) as h5:
        # /time: the table pywr writes so outputs can be mapped back to dates.
        class TimeRow(tables.IsDescription):
            index = tables.Int64Col(pos=0)
            year = tables.Int64Col(pos=1)
            month = tables.Int64Col(pos=2)
            day = tables.Int64Col(pos=3)

        time_table = h5.create_table("/", "time", TimeRow, "Timesteps")
        row = time_table.row
        for i, d in enumerate(dates):
            row["index"] = i
            row["year"] = d.year
            row["month"] = d.month
            row["day"] = d.day
            row.append()
        time_table.flush()

        # Per-node output arrays, shape (timesteps, scenarios).
        node_names = ["reservoir", "abstraction", "demand_centre", "river_gauge"]
        for name in node_names:
            arr = h5.create_carray(
                "/",
                name,
                tables.Float64Atom(),
                shape=(N_TIMESTEPS, N_SCENARIOS),
                filters=filters,
            )
            base = RNG.gamma(2.0, 30.0, (N_TIMESTEPS, 1))
            arr[:, :] = base + RNG.normal(0, 5.0, (N_TIMESTEPS, N_SCENARIOS))

        # A node stored under a group, and one whose row count deliberately
        # does NOT match /time (the time-index toggle must stay off for it).
        grp = h5.create_group("/", "subcatchment", "Nested nodes")
        arr = h5.create_carray(
            grp,
            "tributary",
            tables.Float64Atom(),
            shape=(N_TIMESTEPS, N_SCENARIOS),
            filters=filters,
        )
        arr[:, :] = RNG.gamma(1.5, 10.0, (N_TIMESTEPS, N_SCENARIOS))

        mismatched = h5.create_carray(
            "/", "monthly_summary", tables.Float64Atom(), shape=(120, 4), filters=filters
        )
        mismatched[:, :] = RNG.uniform(0, 100, (120, 4))

        # A scenario-description table, as pywr writes for multi-scenario runs.
        class ScenarioRow(tables.IsDescription):
            name = tables.StringCol(32, pos=0)
            size = tables.Int64Col(pos=1)

        scen = h5.create_table("/", "scenarios", ScenarioRow, "Scenarios")
        srow = scen.row
        srow["name"] = b"climate"
        srow["size"] = N_SCENARIOS
        srow.append()
        scen.flush()


def make_pywr_scenarios(path: Path) -> None:
    """A multi-scenario TablesRecorder run: arrays are (timesteps, s1, s2).

    `TablesRecorder` writes `[len(timestepper)] + model.scenarios.shape`, i.e.
    one axis per Scenario object — not a flat (timesteps, scenarios) array. A
    model with two scenarios produces a 3D array, and the `/scenarios` table is
    what gives those axes names.
    """
    n_steps = 1096  # three years of daily timesteps
    dates = pd.date_range("2020-01-01", periods=n_steps, freq="D")
    sizes = {"climate": 3, "demand": 4}
    filters = tables.Filters(complevel=5, complib="blosc")

    with tables.open_file(str(path), mode="w", filters=filters) as h5:
        h5.root._v_attrs.PYWR_FORMAT = 1
        h5.root._v_attrs.PYWR_VERSION = 1
        h5.root._v_attrs.title = "Two-scenario test model"

        class TimeRow(tables.IsDescription):
            index = tables.Int64Col(pos=0)
            year = tables.Int64Col(pos=1)
            month = tables.Int64Col(pos=2)
            day = tables.Int64Col(pos=3)

        time_table = h5.create_table("/", "time", TimeRow, "Timesteps")
        row = time_table.row
        for i, d in enumerate(dates):
            row["index"] = i
            row["year"], row["month"], row["day"] = d.year, d.month, d.day
            row.append()
        time_table.flush()

        # The table that names the scenario axes, exactly as pywr writes it.
        class ScenarioRow(tables.IsDescription):
            name = tables.StringCol(1024, pos=0)
            size = tables.Int64Col(pos=1)

        scen = h5.create_table("/", "scenarios", ScenarioRow, "Scenarios")
        srow = scen.row
        for name, size in sizes.items():
            srow["name"] = name.encode()
            srow["size"] = size
            srow.append()
        scen.flush()

        shape = (n_steps, sizes["climate"], sizes["demand"])
        nodes = [
            ("reservoir", "volume", "Reservoir"),
            ("abstraction", "flow", "Link"),
            ("demand_centre", "flow", "Output"),
        ]
        for name, attribute, node_type in nodes:
            arr = h5.create_carray(
                "/", name, tables.Float64Atom(), shape=shape, filters=filters
            )
            base = RNG.gamma(2.0, 30.0, (n_steps, 1, 1))
            climate = np.array([0.8, 1.0, 1.3]).reshape(1, 3, 1)
            demand = np.array([0.9, 1.0, 1.1, 1.25]).reshape(1, 1, 4)
            arr[...] = base * climate * demand
            arr._v_attrs.PYWR_ATTRIBUTE = attribute
            arr._v_attrs.PYWR_TYPE = node_type

        # Older pywr wrote these attributes lowercase and hyphenated; both
        # spellings turn up in files people still have on disk.
        legacy = h5.create_carray(
            "/", "legacy_node", tables.Float64Atom(), shape=shape, filters=filters
        )
        legacy[...] = RNG.uniform(0, 10, shape)
        legacy._v_attrs["pywr-attribute"] = "flow"
        legacy._v_attrs["pywr-type"] = "Input"

        # An IndexParameter recorder, which pywr stores as int32.
        idx = h5.create_carray(
            "/", "control_curve_index", tables.Int32Atom(), shape=shape, filters=filters
        )
        idx[...] = RNG.integers(0, 4, shape).astype("int32")
        idx._v_attrs.PYWR_ATTRIBUTE = "parameter-index"
        idx._v_attrs.PYWR_TYPE = "ControlCurveIndexParameter"


def make_pywr_combinations(path: Path) -> None:
    """A run using explicit scenario combinations: arrays are (timesteps, N).

    When a model sets `user_combinations`, pywr collapses the scenario axes into
    a single combination axis and writes `/scenario_combinations` mapping each
    combination back to one index per scenario. The column headers must come
    from that table, not from the raw position.
    """
    n_steps = 365
    dates = pd.date_range("2020-01-01", periods=n_steps, freq="D")
    combinations = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 3)]
    filters = tables.Filters(complevel=5, complib="blosc")

    with tables.open_file(str(path), mode="w", filters=filters) as h5:
        h5.root._v_attrs.PYWR_FORMAT = 1

        class TimeRow(tables.IsDescription):
            index = tables.Int64Col(pos=0)
            year = tables.Int64Col(pos=1)
            month = tables.Int64Col(pos=2)
            day = tables.Int64Col(pos=3)

        t = h5.create_table("/", "time", TimeRow, "Timesteps")
        row = t.row
        for i, d in enumerate(dates):
            row["index"] = i
            row["year"], row["month"], row["day"] = d.year, d.month, d.day
            row.append()
        t.flush()

        class ScenarioRow(tables.IsDescription):
            name = tables.StringCol(1024, pos=0)
            size = tables.Int64Col(pos=1)

        scen = h5.create_table("/", "scenarios", ScenarioRow, "Scenarios")
        srow = scen.row
        for name, size in (("climate", 3), ("demand", 4)):
            srow["name"] = name.encode()
            srow["size"] = size
            srow.append()
        scen.flush()

        class ComboRow(tables.IsDescription):
            climate = tables.Int64Col(pos=0)
            demand = tables.Int64Col(pos=1)

        combo = h5.create_table(
            "/", "scenario_combinations", ComboRow, "Scenario combinations"
        )
        crow = combo.row
        for climate, demand in combinations:
            crow["climate"] = climate
            crow["demand"] = demand
            crow.append()
        combo.flush()

        arr = h5.create_carray(
            "/",
            "reservoir",
            tables.Float64Atom(),
            shape=(n_steps, len(combinations)),
            filters=filters,
        )
        arr[...] = RNG.gamma(2.0, 25.0, (n_steps, len(combinations)))
        arr._v_attrs.PYWR_ATTRIBUTE = "volume"
        arr._v_attrs.PYWR_TYPE = "Reservoir"


def make_big(path: Path, rows: int = 20_000_000, cols: int = 10) -> None:
    """20M x 10 chunked floats for performance tests. Written in blocks."""
    block = 500_000
    with h5py.File(path, "w") as f:
        d = f.create_dataset(
            "big",
            shape=(rows, cols),
            dtype="f8",
            chunks=(block // 10, cols),
            compression="gzip",
            compression_opts=1,
        )
        for start in range(0, rows, block):
            stop = min(start + block, rows)
            d[start:stop, :] = RNG.normal(0, 1, (stop - start, cols))
            print(f"  big.h5: {stop:,}/{rows:,} rows", end="\r", flush=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--big", action="store_true", help="also generate big.h5 (20M rows, slow)"
    )
    args = parser.parse_args()

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        ("plain.h5", make_plain),
        ("pandas_fixed.h5", make_pandas_fixed),
        ("pandas_table.h5", make_pandas_table),
        ("pywr_style.h5", make_pywr_style),
        ("pywr_scenarios.h5", make_pywr_scenarios),
        ("pywr_combinations.h5", make_pywr_combinations),
    ]
    for name, fn in targets:
        path = FIXTURE_DIR / name
        print(f"writing {name} ...", end=" ", flush=True)
        fn(path)
        print(f"ok ({path.stat().st_size / 1e6:.1f} MB)")

    if args.big:
        path = FIXTURE_DIR / "big.h5"
        print("writing big.h5 (this takes a few minutes) ...")
        make_big(path)
        print(f"ok ({path.stat().st_size / 1e9:.2f} GB)")

    print(f"\nfixtures in {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
