"""Open-file registry: handle caching, the mtime guard, and access serialization.

Two constraints shape this module.

The HDF5 C library is not reliably thread-safe (h5py holds a global lock, but
PyTables does not), so every read for a given file runs under that file's
asyncio lock and inside a worker thread. The lock keeps HDF5 calls serialized;
the thread keeps the event loop responsive during a slow read.

Model runs rewrite their output files while a user has them open. `file_id`
folds in the mtime, so once the file changes on disk the id no longer resolves
and the API answers 409 rather than serving data from a stale handle.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import h5py
import pandas as pd

from . import tree
from .readers import (
    NodeReader,
    PandasFixedReader,
    PandasTableReader,
    RawDatasetReader,
)

T = TypeVar("T")

IDLE_TIMEOUT_SECONDS = 600  # 10 minutes, per the spec
MAX_OPEN_FILES = 8
MAX_CACHED_FRAMES = 2  # decoded fixed-format frames, each up to the 500 MB guard

H5_SUFFIXES = {".h5", ".hdf5", ".hdf", ".he5", ".nc"}


class FileChangedError(Exception):
    """The file on disk no longer matches the handle we opened."""


class NodeNotFoundError(Exception):
    pass


@dataclass
class OpenFile:
    file_id: str
    path: Path
    mtime_ns: int
    size_bytes: int
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=time.monotonic)

    _h5: h5py.File | None = None
    _store: pd.HDFStore | None = None
    _store_failed: bool = False
    _readers: dict[str, NodeReader] = field(default_factory=dict)
    _frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    _stats: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    _time_index: pd.DatetimeIndex | None = None
    _time_index_loaded: bool = False
    _scenarios: Any = None
    _scenarios_loaded: bool = False

    # -- handles ---------------------------------------------------------

    def h5(self) -> h5py.File:
        self.last_used = time.monotonic()
        if self._h5 is None:
            self._h5 = _open_h5py(self.path)
        return self._h5

    def store(self) -> pd.HDFStore | None:
        """A PyTables handle, or None if this file is not PyTables-readable."""
        self.last_used = time.monotonic()
        if self._store is None and not self._store_failed:
            try:
                self._store = pd.HDFStore(str(self.path), mode="r")
            except Exception:
                self._store_failed = True
        return self._store

    def close(self) -> None:
        for handle in (self._store, self._h5):
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                pass
        self._store = None
        self._h5 = None
        self._readers.clear()
        self._frames.clear()
        self._time_index = None
        self._time_index_loaded = False
        self._scenarios = None
        self._scenarios_loaded = False

    def verify_unchanged(self) -> None:
        try:
            stat = self.path.stat()
        except OSError as exc:
            raise FileChangedError(str(exc)) from exc
        if stat.st_mtime_ns != self.mtime_ns or stat.st_size != self.size_bytes:
            raise FileChangedError(
                f"{self.path.name} changed on disk since it was opened."
            )

    # -- caches ----------------------------------------------------------

    def frame_cache(self) -> dict[str, pd.DataFrame]:
        if len(self._frames) > MAX_CACHED_FRAMES:
            for key in list(self._frames)[:-MAX_CACHED_FRAMES]:
                self._frames.pop(key, None)
        return self._frames

    def stats_cache(self) -> dict[tuple[str, str], dict[str, Any]]:
        return self._stats

    def time_index(self):
        """The file's `/time` table as a DatetimeIndex, or None."""
        if not self._time_index_loaded:
            self._time_index_loaded = True
            try:
                from . import timeindex

                path = timeindex.find_time_table(self.h5())
                if path is not None:
                    self._time_index = timeindex.read_time_index(self.h5(), path)
            except Exception:
                self._time_index = None
        return self._time_index

    def scenario_layout(self):
        """The file's pywr scenario layout, or None if it is not a pywr file."""
        if not self._scenarios_loaded:
            self._scenarios_loaded = True
            try:
                from . import pywr

                self._scenarios = pywr.read_layout(self.h5())
            except Exception:
                self._scenarios = None
        return self._scenarios

    # -- readers ---------------------------------------------------------

    def reader(self, node_path: str) -> NodeReader:
        """Build (and cache) the right reader for a node. Selection happens once."""
        key = "/" + node_path.strip("/")
        cached = self._readers.get(key)
        if cached is not None:
            return cached

        h5 = self.h5()
        lookup = key.strip("/")
        obj = h5 if lookup == "" else h5.get(lookup)
        if obj is None:
            raise NodeNotFoundError(f"No node at {key!r}")

        kind = tree.find_node_kind(h5, key)

        if kind == tree.KIND_DATASET:
            reader: NodeReader = RawDatasetReader(obj, key, self.scenario_layout())
        elif kind == tree.KIND_PANDAS_TABLE:
            store = self.store()
            if store is None:
                raise NodeNotFoundError(
                    f"{key!r} is a pandas table but the file could not be opened "
                    "with PyTables."
                )
            reader = PandasTableReader(store, key)
        elif kind == tree.KIND_PANDAS_FRAME:
            store = self.store()
            if store is None:
                raise NodeNotFoundError(
                    f"{key!r} is a pandas frame but the file could not be opened "
                    "with PyTables."
                )
            reader = PandasFixedReader(store, key, self.frame_cache())
        else:
            raise NodeNotFoundError(f"{key!r} is a group, not a readable table.")

        self._readers[key] = reader
        return reader

    async def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run a blocking HDF5 call: serialized per file, off the event loop."""
        async with self.lock:
            self.verify_unchanged()
            self.last_used = time.monotonic()
            return await asyncio.to_thread(fn, *args, **kwargs)


def _open_h5py(path: Path) -> h5py.File:
    """Open read-only, tolerating h5py builds without the `locking` argument."""
    try:
        return h5py.File(str(path), "r", locking=False)
    except TypeError:
        return h5py.File(str(path), "r")


class FileRegistry:
    """All files open in this session, keyed by a path+mtime hash."""

    def __init__(
        self,
        *,
        idle_timeout: float = IDLE_TIMEOUT_SECONDS,
        max_open: int = MAX_OPEN_FILES,
    ) -> None:
        self._files: dict[str, OpenFile] = {}
        self._idle_timeout = idle_timeout
        self._max_open = max_open

    def open(self, path: str | os.PathLike[str]) -> OpenFile:
        resolved = Path(path).expanduser()
        try:
            resolved = resolved.resolve(strict=True)
        except OSError as exc:
            raise FileNotFoundError(f"No such file: {path}") from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"Not a file: {resolved}")

        stat = resolved.stat()
        file_id = _make_file_id(resolved, stat.st_mtime_ns)

        existing = self._files.get(file_id)
        if existing is not None:
            existing.last_used = time.monotonic()
            return existing

        # Fail fast with a clear message rather than at first read.
        probe = _open_h5py(resolved)
        probe.close()

        entry = OpenFile(
            file_id=file_id,
            path=resolved,
            mtime_ns=stat.st_mtime_ns,
            size_bytes=stat.st_size,
        )
        self._files[file_id] = entry
        self._evict()
        return entry

    def get(self, file_id: str) -> OpenFile:
        entry = self._files.get(file_id)
        if entry is None:
            raise KeyError(file_id)
        entry.verify_unchanged()
        entry.last_used = time.monotonic()
        return entry

    def close(self, file_id: str) -> bool:
        entry = self._files.pop(file_id, None)
        if entry is None:
            return False
        entry.close()
        return True

    def close_all(self) -> None:
        for entry in list(self._files.values()):
            entry.close()
        self._files.clear()

    def sweep(self) -> None:
        """Release handles for files nobody has touched recently.

        The entry stays in the registry, so its file_id keeps working and the
        handle is reopened transparently on the next request.
        """
        now = time.monotonic()
        for entry in self._files.values():
            if entry._h5 is None and entry._store is None:
                continue
            if now - entry.last_used > self._idle_timeout and not entry.lock.locked():
                entry.close()

    def _evict(self) -> None:
        if len(self._files) <= self._max_open:
            return
        by_age = sorted(self._files.values(), key=lambda e: e.last_used)
        for entry in by_age[: len(self._files) - self._max_open]:
            entry.close()
            self._files.pop(entry.file_id, None)

    def __len__(self) -> int:
        return len(self._files)


def _make_file_id(path: Path, mtime_ns: int) -> str:
    digest = hashlib.sha256(f"{path}:{mtime_ns}".encode()).hexdigest()
    return digest[:16]


def _windows_volume_label(root: str) -> str:
    """The friendly name of a Windows drive, e.g. "Backup" for D:\\.

    Returns "" when the label cannot be read (unformatted, disconnected
    network drive, or a permission problem) — the caller falls back to the
    bare drive letter.
    """
    try:
        import ctypes

        buffer = ctypes.create_unicode_buffer(261)
        ok = ctypes.windll.kernel32.GetVolumeInformationW(  # type: ignore[attr-defined]
            ctypes.c_wchar_p(root),
            buffer,
            ctypes.sizeof(buffer) // ctypes.sizeof(ctypes.c_wchar),
            None,
            None,
            None,
            None,
            0,
        )
        return buffer.value.strip() if ok else ""
    except Exception:
        return ""


def _windows_drives() -> list[tuple[str, Path]]:
    """(display name, path) for every mounted Windows drive."""
    try:
        letters = os.listdrives()  # Python 3.12+
    except AttributeError:
        letters = [f"{chr(c)}:\\" for c in range(ord("A"), ord("Z") + 1)]

    drives: list[tuple[str, Path]] = []
    for root in letters:
        path = Path(root)
        try:
            if not path.is_dir():
                continue
        except OSError:
            # Empty optical/card readers raise rather than returning False.
            continue
        letter = root.rstrip("\\/")
        label = _windows_volume_label(root)
        drives.append((f"{label} ({letter})" if label else letter, path))
    return drives


# Where each platform mounts removable and network media. Module-level so tests
# can point it at a temporary tree.
_MOUNT_ROOTS = (
    Path("/Volumes"),      # macOS
    Path("/media"),        # Linux (Debian/Ubuntu)
    Path("/run/media"),    # Linux (Fedora/Arch)
    Path("/mnt"),          # Linux manual mounts, and WSL's Windows drives
)


def _unix_mounted_volumes() -> list[tuple[str, Path]]:
    """(display name, path) for removable/external media on macOS and Linux."""
    volumes: list[tuple[str, Path]] = []

    def scan(directory: Path) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            return
        for child in children:
            if child.name.startswith("."):
                continue
            try:
                if not child.is_dir():
                    continue
                # macOS puts the boot volume in /Volumes as a symlink to "/".
                # Listing it among the external drives is misleading, and it
                # would otherwise take the slot the "Computer" entry wants.
                if child.resolve() == Path("/"):
                    continue
            except OSError:
                continue
            volumes.append((child.name, child))

    for mount_root in _MOUNT_ROOTS:
        if not mount_root.is_dir():
            continue
        # Linux commonly nests one level deeper: /media/<user>/<label>. Descend
        # into the directory matching the current user so the drive itself is
        # offered, not the container folder.
        user_dir = mount_root / os.environ.get("USER", os.environ.get("USERNAME", ""))
        try:
            if user_dir.name and user_dir.is_dir():
                scan(user_dir)
                continue
        except OSError:
            pass
        scan(mount_root)

    return volumes


def list_roots() -> list[dict[str, Any]]:
    """Quick-access locations for the file picker.

    Drives matter here: model data usually lives on an external or network
    drive, and reaching it by walking up from the home directory is a poor way
    to get there. Everything below is read from the running machine — nothing
    is hardcoded, so the names shown are whatever that computer actually has.
    """
    roots: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(name: str, path: Path, kind: str) -> None:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            return
        key = str(resolved)
        if key in seen or not resolved.is_dir():
            return
        seen.add(key)
        roots.append({"name": name, "path": key, "kind": kind})

    add("Home", Path.home(), "home")
    add("Desktop", Path.home() / "Desktop", "folder")
    add("Documents", Path.home() / "Documents", "folder")
    try:
        add(Path.cwd().name or "Working directory", Path.cwd(), "cwd")
    except OSError:
        pass

    if os.name == "nt":
        # Windows has no single filesystem root, so the drive list *is* the
        # root list — and "/" would resolve to whichever drive we happen to be
        # on, which is meaningless. Drives cover it instead.
        for name, path in _windows_drives():
            add(name, path, "volume")
    else:
        for name, path in _unix_mounted_volumes():
            add(name, path, "volume")
        add("Computer", Path("/"), "root")

    return roots


def list_directory(directory: str | os.PathLike[str] | None) -> dict[str, Any]:
    """Directory listing for the Open-file dialog.

    A browser cannot hand a web page a filesystem path, so the picker is served
    by the backend instead.
    """
    target = Path(directory).expanduser() if directory else Path.home()
    try:
        target = target.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"No such directory: {directory}") from exc
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {target}")

    entries: list[dict[str, Any]] = []
    try:
        for child in sorted(
            target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        ):
            if child.name.startswith("."):
                continue
            try:
                is_dir = child.is_dir()
                size = child.stat().st_size if not is_dir else None
            except OSError:
                continue
            suffix = child.suffix.lower()
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "is_dir": is_dir,
                    "size_bytes": size,
                    "is_h5": (not is_dir) and suffix in H5_SUFFIXES,
                }
            )
    except PermissionError as exc:
        raise PermissionError(f"Cannot read {target}") from exc

    # Every ancestor, so the client can offer a clickable path rather than only
    # a step-up-one control.
    crumbs: list[dict[str, str]] = []
    for ancestor in reversed([target, *target.parents]):
        crumbs.append({"name": ancestor.name or "/", "path": str(ancestor)})

    return {
        "dir": str(target),
        "parent": str(target.parent) if target.parent != target else None,
        "breadcrumbs": crumbs,
        "entries": entries,
    }
