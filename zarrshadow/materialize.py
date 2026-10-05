"""Turn a reference file system into an ordinary Zarr store.

A reference file system points at bytes in other files. materialize reads
those bytes and writes them into a Zarr store of their own, with the
chunking and compression chosen here, so the result no longer depends on
the files it came from:

    materialize("session.nwb.zarrshadow", "session.nwb.zarr")

Everything that is stored in the reference file system itself (groups,
attributes, small datasets) is copied as it is. An array whose chunks are
references is rewritten. The reading goes through RfsStore, so this works
for any source the references describe and needs none of the libraries that
read the source formats.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import zarr

from .builder import contiguous_chunk_shape
from .open_rfs import load_rfs
from .rfs_store import RfsStore, _array_path

#: Given an array's path and the array as read through the references, returns
#: keyword arguments for zarr.create_array ("chunks", "shards", "compressors",
#: "filters", "serializer") that replace the defaults, or None to copy the
#: array's chunks as they are stored.
Layout = Callable[[str, zarr.Array], "dict | None"]


def materialize(
    rfs: dict | str,
    target: Any,
    *,
    layout: Layout | None = None,
    chunk_bytes: int = 4 * 2**20,
    slab_bytes: int = 256 * 2**20,
    verify: bool = False,
    validate_sources: bool = True,
) -> dict[str, dict]:
    """Write the data a reference file system points at into a Zarr store.

    Parameters
    ----------
    rfs : dict or str
        A reference file system, or the path or URL of one.
    target : str or zarr store
        Where to write: a directory path, or a zarr store.
    layout : callable or None
        Chooses how each referenced array is stored; see Layout. By default an
        array of numbers is cut into chunks of about chunk_bytes along its
        first axis and compressed with zarr's default compressor, and any
        other array is copied as stored.
    chunk_bytes : int
        Approximate uncompressed size of a chunk in the default layout.
    slab_bytes : int
        How much of an array is held in memory at a time while copying.
    verify : bool
        Read back what was written and compare it with what was read.
    validate_sources : bool
        Check the source files against the sizes and ETags the references record.

    Returns
    -------
    dict
        For each array that was rewritten, its path and a dict with "nbytes",
        the size of its values, and "nbytes_stored", the size written.
    """
    if isinstance(rfs, str):
        rfs = load_rfs(rfs)
    source_store = RfsStore(rfs, validate_sources=validate_sources)
    source = zarr.open_group(source_store, mode="r", zarr_format=3)
    target_store = zarr.storage.LocalStore(target) if isinstance(target, str) else target

    referenced = _referenced_arrays(rfs)
    # Copy what the references file stores itself: metadata and small datasets
    copied = {}
    for key in rfs["refs"]:
        if _array_path(key) in referenced or _metadata_path(key) in referenced:
            continue
        copied[key] = source_store._get_bytes(key)
    _write_keys(target_store, copied)

    report = {}
    for path in sorted(referenced):
        array = source[path]
        default = _default_layout(array, chunk_bytes)
        chosen = default if layout is None else layout(path, array)
        if chosen is None:
            _copy_as_stored(source_store, target_store, path, array)
            continue
        if layout is not None:
            chosen = {**(default or {}), **chosen}
        written = zarr.create_array(
            target_store,
            name=path,
            shape=array.shape,
            dtype=array.dtype,
            fill_value=array.fill_value,
            attributes=dict(array.attrs),
            dimension_names=array.metadata.dimension_names,
            zarr_format=3,
            overwrite=True,
            **chosen,
        )
        _copy_values(array, written, slab_bytes, verify)
        report[path] = {"nbytes": int(array.nbytes), "nbytes_stored": int(written.nbytes_stored())}

    # A copy of every array's metadata may be kept in the root, and it must describe what is stored now
    root = rfs["refs"].get("zarr.json", {})
    if "consolidated_metadata" in (json.loads(root) if isinstance(root, str) else root):
        zarr.consolidate_metadata(target_store, zarr_format=3)
    return report


def _referenced_arrays(rfs: dict) -> set[str]:
    """The paths of the arrays with at least one chunk that is a reference to another file."""
    paths = set(rfs.get("indexes", {})) | set(rfs.get("selections", {}))
    for key, value in rfs["refs"].items():
        if isinstance(value, list):
            path = _array_path(key)
            if path is not None:
                paths.add(path)
    for entry in rfs.get("gen", []):
        head, sep, _ = entry["key"].rpartition("/c/")
        paths.add(head if sep else "")
    return paths


def _metadata_path(key: str) -> str | None:
    """The path of the group or array a zarr.json key describes, or None for any other key."""
    if key == "zarr.json":
        return ""
    return key[: -len("/zarr.json")] if key.endswith("/zarr.json") else None


def _default_layout(array: zarr.Array, chunk_bytes: int) -> dict | None:
    """Chunks along the first axis for an array of numbers; None, to copy as stored, for anything else."""
    if array.dtype.kind not in "iufb" or array.ndim == 0:
        return None
    chunks = contiguous_chunk_shape(array.shape, array.dtype.itemsize, chunk_bytes)
    return {"chunks": tuple(max(c, 1) for c in chunks)}


def _copy_values(source: zarr.Array, written: zarr.Array, slab_bytes: int, verify: bool) -> None:
    """Copy an array in slabs along its first axis that line up with the chunks being written."""
    if source.size == 0:
        return
    row_bytes = max(1, source.dtype.itemsize * int(np.prod(source.shape[1:])))
    chunk_rows = written.shards[0] if written.shards is not None else written.chunks[0]
    rows = max(chunk_rows, slab_bytes // row_bytes // chunk_rows * chunk_rows)
    for start in range(0, source.shape[0], rows):
        region = slice(start, min(start + rows, source.shape[0]))
        values = source[region]
        written[region] = values
        if verify and not np.array_equal(written[region], values, equal_nan=values.dtype.kind == "f"):
            raise RuntimeError(f"{written.path!r} rows {region.start}:{region.stop} read back different from what was written")


def _copy_as_stored(source_store: RfsStore, target_store: Any, path: str, array: zarr.Array) -> None:
    """Copy an array's metadata and every chunk it has, without decoding them."""
    keys = {f"{path}/zarr.json" if path else "zarr.json": None}
    prefix = f"{path}/c" if path else "c"
    for coords in np.ndindex(*array.cdata_shape):
        keys[prefix + "".join(f"/{c}" for c in coords)] = None
    out = {}
    for key in keys:
        data = source_store._get_bytes(key)
        if data is not None:
            out[key] = data
    _write_keys(target_store, out)


def _write_keys(store: Any, values: dict[str, bytes]) -> None:
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    async def write() -> None:
        prototype = default_buffer_prototype()
        for key, data in values.items():
            await store.set(key, prototype.buffer.from_bytes(data))

    sync(write())
