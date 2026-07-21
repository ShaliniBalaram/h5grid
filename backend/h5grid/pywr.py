"""Understanding what pywr's TablesRecorder actually wrote.

`TablesRecorder` stores one array per model node, shaped
`[len(timestepper)] + model.scenarios.shape` — that is, **one axis per Scenario
object**, not a flat (timesteps, scenarios) array. A model with a climate
scenario of size 3 and a demand scenario of size 4 produces a (T, 3, 4) array.

Alongside the arrays it writes tables that name all of this, and which no other
viewer reads:

  /time                    year/month/day per timestep (see timeindex.py)
  /scenarios               name and size of each Scenario, in axis order
  /scenario_combinations   present only when the model used explicit
                           combinations, in which case the scenario axes are
                           collapsed into one and this maps each column back to
                           one index per scenario
  PYWR_ATTRIBUTE           per array: flow / volume / parameter / parameter-index
  PYWR_TYPE                per array: the node's Python class name

Without this the scenario axes are anonymous integers, which is the difference
between "col_7" and "climate[1], demand[3]".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import h5py
import numpy as np

SCENARIOS_PATH = "scenarios"
# pywr has moved this around between versions; check both.
COMBINATION_PATHS = ("scenario_combinations", "scenarios/scenario_combinations")

# Current pywr writes upper-case underscored attributes; older files (and models
# built with use_legacy_attribute_naming) use lower-case hyphenated ones.
_ATTRIBUTE_KEYS = ("PYWR_ATTRIBUTE", "pywr-attribute")
_TYPE_KEYS = ("PYWR_TYPE", "pywr-type")


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


@dataclass(frozen=True)
class Scenario:
    name: str
    size: int


@dataclass
class ScenarioLayout:
    """How a file's scenario axes map onto dataset dimensions."""

    scenarios: list[Scenario]
    combinations: list[tuple[int, ...]] | None = None

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(s.size for s in self.scenarios)

    def describes(self, shape: Sequence[int]) -> bool:
        """True if `shape` looks like a recorder array for this file.

        Dimension 0 is always time and is not checked here; the caller has
        already matched it against the length of /time.
        """
        rest = tuple(int(s) for s in shape[1:])
        if self.combinations is not None:
            return len(rest) == 1 and rest[0] == len(self.combinations)
        return rest == self.sizes

    def is_collapsed(self) -> bool:
        return self.combinations is not None

    def axis_name(self, dim: int) -> str | None:
        """Name of the scenario on dataset dimension `dim` (0 is time)."""
        if dim <= 0:
            return None
        if self.combinations is not None:
            return "scenario combination" if dim == 1 else None
        index = dim - 1
        if index < len(self.scenarios):
            return self.scenarios[index].name
        return None

    def labels_for_dim(self, dim: int, size: int) -> list[str] | None:
        """Column headers for the free dimension `dim`, or None if unknown."""
        if self.combinations is not None:
            if dim != 1 or size != len(self.combinations):
                return None
            return [self.combination_label(i) for i in range(size)]

        index = dim - 1
        if not (0 <= index < len(self.scenarios)):
            return None
        scenario = self.scenarios[index]
        if scenario.size != size:
            return None
        return [f"{scenario.name}[{i}]" for i in range(size)]

    def combination_label(self, index: int) -> str:
        if self.combinations is None or index >= len(self.combinations):
            return f"combination[{index}]"
        parts = [
            f"{scenario.name}={value}"
            for scenario, value in zip(self.scenarios, self.combinations[index])
        ]
        return ", ".join(parts) if parts else f"combination[{index}]"

    def to_json(self) -> dict[str, Any]:
        return {
            "scenarios": [{"name": s.name, "size": s.size} for s in self.scenarios],
            "collapsed": self.is_collapsed(),
            "combination_count": (
                len(self.combinations) if self.combinations is not None else None
            ),
        }


def read_scenarios(h5file: h5py.File) -> list[Scenario]:
    """Read /scenarios, in axis order. Empty list if the file has none."""
    node = h5file.get(SCENARIOS_PATH)
    if node is None or not hasattr(node, "dtype") or node.dtype.names is None:
        return []

    fields = {n.lower(): n for n in node.dtype.names}
    if "name" not in fields or "size" not in fields:
        return []

    data = node[...]
    return [
        Scenario(name=_text(row[fields["name"]]), size=int(row[fields["size"]]))
        for row in data
    ]


def read_combinations(
    h5file: h5py.File, scenarios: Sequence[Scenario]
) -> list[tuple[int, ...]] | None:
    """Read the scenario-combination table, if the model used one."""
    node = None
    for path in COMBINATION_PATHS:
        candidate = h5file.get(path)
        if candidate is not None and hasattr(candidate, "dtype"):
            node = candidate
            break
    if node is None or node.dtype.names is None:
        return None

    names = list(node.dtype.names)
    # Prefer columns matching the scenario names, in scenario order; fall back
    # to the table's own column order when the names do not line up.
    lookup = {n.lower(): n for n in names}
    ordered = [lookup.get(s.name.lower()) for s in scenarios]
    if any(column is None for column in ordered):
        ordered = names

    data = node[...]
    try:
        return [tuple(int(row[column]) for column in ordered) for row in data]
    except (KeyError, ValueError, TypeError):
        return None


def read_layout(h5file: h5py.File) -> ScenarioLayout | None:
    """Build the file's scenario layout, or None if it is not a pywr file."""
    scenarios = read_scenarios(h5file)
    if not scenarios:
        return None
    return ScenarioLayout(
        scenarios=scenarios, combinations=read_combinations(h5file, scenarios)
    )


def node_metadata(obj: Any) -> dict[str, str]:
    """PYWR_ATTRIBUTE / PYWR_TYPE for a node, under either naming convention."""
    attrs = getattr(obj, "attrs", None)
    if attrs is None:
        return {}

    out: dict[str, str] = {}
    for key in _ATTRIBUTE_KEYS:
        if key in attrs:
            out["attribute"] = _text(attrs[key])
            break
    for key in _TYPE_KEYS:
        if key in attrs:
            out["type"] = _text(attrs[key])
            break
    return out


def is_pywr_file(h5file: h5py.File) -> bool:
    if any(key.upper().startswith("PYWR_") for key in h5file.attrs):
        return True
    return bool(read_scenarios(h5file))


def scenario_index_of(layout: ScenarioLayout, shape: Sequence[int], dim: int) -> int | None:
    """Which scenario a dataset dimension corresponds to, if any."""
    if not layout.describes(shape) or layout.combinations is not None:
        return None
    index = dim - 1
    return index if 0 <= index < len(layout.scenarios) else None


def summarise(values: np.ndarray) -> dict[str, float]:
    """Small helper used by tests to sanity-check decoded slices."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {}
    return {"min": float(finite.min()), "max": float(finite.max())}
