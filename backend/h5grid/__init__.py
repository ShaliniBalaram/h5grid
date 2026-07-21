"""H5Grid: a lightweight HDF5 viewer for water resource model files."""

import os

# Must be set before the HDF5 C library is loaded by h5py or PyTables, which is
# why it lives here rather than in cli.py. Without it, opening a file that a
# model run currently holds open fails outright, and opening one from a network
# share often does too. We only ever open read-only, so skipping the lock is
# safe for us; the reader may see a torn write, which the mtime guard catches.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# Registers blosc, zstd, lz4 and friends with the HDF5 library. PyTables (and
# therefore pywr) writes blosc-compressed chunks by default, and plain h5py
# cannot decompress them: reads fail with "can't open directory .../plugin".
# Importing here means any use of the package has the filters available.
import hdf5plugin  # noqa: E402,F401

__version__ = "0.1.0"
