"""Walking the file into the tree the user actually wants to see.

The important rule from the spec: a group carrying a `pandas_type` attribute is
a logical table, not a folder. Its internals (`axis0`, `block0_values`, the
`_i_table` index containers PyTables writes) are hidden unless the caller asks
for the raw structure. That single rule is most of the difference between this
and every other free viewer.

Tree building stays metadata-only: row and column counts for pandas nodes come
from the shapes of their internal arrays, never from decoding the frame. A tree
request on a 10 GB file must not read 10 GB.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import h5py

from . import pywr
from .readers import FIXED_FORMAT_SIZE_LIMIT_BYTES

KIND_GROUP = "group"
KIND_DATASET = "dataset"
KIND_PANDAS_FRAME = "pandas_frame"
KIND_PANDAS_TABLE = "pandas_table"
KIND_BROKEN_LINK = "broken_link"

_DIGITS = re.compile(r"(\d+)")


def natural_sorted(names: Iterable[str]) -> list[str]:
    """Sort so `node_2` comes before `node_10`.

    Model nodes are named with unpadded numbers far more often than padded
    ones, and plain lexicographic order interleaves them (`node_1, node_10,
    node_100, node_2`), which makes a long tree very hard to scan.
    """

    def key(name: str):
        return [
            int(part) if part.isdigit() else part.lower()
            for part in _DIGITS.split(name)
        ]

    return sorted(names, key=key)


def _broken_link_node(
    group: h5py.Group, name: str, path: str, error: Exception
) -> dict[str, Any]:
    """A node standing in for a link that cannot be resolved."""
    detail = ""
    try:
        link = group.get(name, getlink=True)
        if isinstance(link, h5py.ExternalLink):
            detail = f"external link to {link.filename}:{link.path}"
        elif isinstance(link, h5py.SoftLink):
            detail = f"soft link to {link.path}"
    except Exception:
        pass

    return {
        "name": name,
        "path": path,
        "kind": KIND_BROKEN_LINK,
        "shape": None,
        "dtype": None,
        "nrows": None,
        "ncols": None,
        "ndim": None,
        "decode_fallback": False,
        "error": detail or f"{type(error).__name__}: {error}",
        "children": [],
    }

# PyTables bookkeeping that is never interesting to a modeller.
_HIDDEN_CLASSES = {"TINDEX", "INDEX"}
_HIDDEN_PREFIXES = ("_i_",)


def _attr_str(obj: Any, name: str) -> str | None:
    value = obj.attrs.get(name)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def pandas_type_of(obj: Any) -> str | None:
    """Return the `pandas_type` of a group, if it has one."""
    if not isinstance(obj, h5py.Group):
        return None
    return _attr_str(obj, "pandas_type")


def is_hidden(name: str, obj: Any) -> bool:
    if any(name.startswith(prefix) for prefix in _HIDDEN_PREFIXES):
        return True
    return _attr_str(obj, "CLASS") in _HIDDEN_CLASSES


def _pandas_frame_dims(group: h5py.Group) -> tuple[int, int]:
    """(nrows, ncols) of a fixed-format frame, from its axis arrays."""
    nrows = ncols = 0
    if "axis1" in group and hasattr(group["axis1"], "shape"):
        nrows = int(group["axis1"].shape[0])
    if "axis0" in group and hasattr(group["axis0"], "shape"):
        ncols = int(group["axis0"].shape[0])

    # Wide frames get split across several blocks; axis0 already counts them
    # all, but fall back to summing blocks if axis0 is missing.
    if ncols == 0:
        for key in group:
            if key.endswith("_items") and hasattr(group[key], "shape"):
                ncols += int(group[key].shape[0])
    return nrows, ncols


def _pandas_table_dims(group: h5py.Group) -> tuple[int, int]:
    """(nrows, ncols) of a frame_table, from the `table` dataset's dtype."""
    table = group.get("table")
    if table is None or not hasattr(table, "shape"):
        return 0, 0
    nrows = int(table.shape[0])
    names = table.dtype.names or ()
    # Every field except the index is a value column; a multi-column block is
    # stored as a single 2D field, so count its width rather than the field.
    ncols = 0
    for name in names:
        if name == "index":
            continue
        sub = table.dtype[name]
        ncols += int(sub.shape[0]) if sub.shape else 1
    return nrows, ncols


def _dataset_node(name: str, path: str, dset: h5py.Dataset) -> dict[str, Any]:
    shape = [int(s) for s in dset.shape]
    is_compound = dset.dtype.names is not None
    if is_compound:
        ncols = len(dset.dtype.names)
    elif len(shape) >= 2:
        ncols = shape[1]
    else:
        ncols = 1

    return {
        "name": name,
        "path": path,
        "kind": KIND_DATASET,
        "shape": shape,
        "dtype": "compound" if is_compound else str(dset.dtype),
        "nrows": shape[0] if shape else 1,
        "ncols": ncols,
        "ndim": len(shape),
        "decode_fallback": False,
        # What pywr recorded here — "volume" from a Reservoir, say. Free
        # structure that no other viewer surfaces.
        "pywr": pywr.node_metadata(dset) or None,
        "children": [],
    }


def build_tree(h5file: h5py.File, *, raw: bool = False) -> dict[str, Any]:
    """Build the whole tree from the file root."""
    root = _walk(h5file, name="/", path="/", raw=raw)
    root["kind"] = KIND_GROUP
    return root


def _walk(group: h5py.Group, *, name: str, path: str, raw: bool) -> dict[str, Any]:
    node: dict[str, Any] = {
        "name": name,
        "path": path,
        "kind": KIND_GROUP,
        "shape": None,
        "dtype": None,
        "nrows": None,
        "ncols": None,
        "ndim": None,
        "decode_fallback": False,
        "children": [],
    }

    ptype = pandas_type_of(group)
    if ptype in ("frame", "frame_table") and not raw:
        return _pandas_node(group, name=name, path=path, ptype=ptype)

    for child_name in natural_sorted(group.keys()):
        try:
            child = group[child_name]
        except (KeyError, OSError) as exc:
            # A broken external or soft link. Show it as an unreadable node
            # rather than dropping it: a node that silently disappears from the
            # tree is indistinguishable from one the model never wrote.
            node["children"].append(
                _broken_link_node(group, child_name, f"{path.rstrip('/')}/{child_name}", exc)
            )
            continue

        if not raw and is_hidden(child_name, child):
            continue

        child_path = f"{path.rstrip('/')}/{child_name}"
        if isinstance(child, h5py.Group):
            node["children"].append(
                _walk(child, name=child_name, path=child_path, raw=raw)
            )
        elif isinstance(child, h5py.Dataset):
            node["children"].append(_dataset_node(child_name, child_path, child))

    return node


def _pandas_node(
    group: h5py.Group, *, name: str, path: str, ptype: str
) -> dict[str, Any]:
    if ptype == "frame_table":
        nrows, ncols = _pandas_table_dims(group)
        kind = KIND_PANDAS_TABLE
        fallback = False
    else:
        nrows, ncols = _pandas_frame_dims(group)
        kind = KIND_PANDAS_FRAME
        fallback = nrows * max(ncols, 1) * 8 > FIXED_FORMAT_SIZE_LIMIT_BYTES

    node = {
        "name": name,
        "path": path,
        "kind": kind,
        "shape": [nrows, ncols],
        "dtype": ptype,
        "nrows": nrows,
        "ncols": ncols,
        "ndim": 2,
        "decode_fallback": fallback,
        "children": [],
    }

    # Too large to decode: show it as a folder of raw blocks so the data is
    # still reachable, with the banner explaining why.
    if fallback:
        node["kind"] = KIND_GROUP
        node["children"] = _walk(group, name=name, path=path, raw=True)["children"]
    return node


def find_node_kind(h5file: h5py.File, path: str) -> str:
    """Classify a single path without walking the whole file."""
    key = path.strip("/")
    obj = h5file if key == "" else h5file[key]

    if isinstance(obj, h5py.Dataset):
        return KIND_DATASET

    ptype = pandas_type_of(obj)
    if ptype == "frame_table":
        return KIND_PANDAS_TABLE
    if ptype == "frame":
        nrows, ncols = _pandas_frame_dims(obj)
        if nrows * max(ncols, 1) * 8 > FIXED_FORMAT_SIZE_LIMIT_BYTES:
            return KIND_GROUP
        return KIND_PANDAS_FRAME
    return KIND_GROUP
