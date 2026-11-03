"""Statistics of blocks of an array, stored next to it, that readers use to skip data.

A reader that wants the part of a sorted timestamps array between two times
would otherwise read the whole array. With the minimum and maximum of every
block of, say, 10,000 values, it reads that small array, bisects it to find
the one or two blocks holding the times, and reads only those. Tabular formats
call these per-block statistics zone maps; Parquet stores them for every row
group. The convention is described in docs/block-statistics.md.

The statistics of an array named "<name>" are Zarr arrays next to it, in a
".stats" group in the same parent group, at ".stats/<name>/<type>". A copy of
the parent group keeps them. The array lists them in its "block_statistics"
attribute, with paths relative to its parent group:

    "block_statistics": {
        "sorted": "ascending",
        "statistics": [
            {"type": "minmax", "block_shape": [10000], "path": ".stats/timestamps/minmax"}
        ]
    }

A block is block_shape elements of the array, counted from its origin; the
last block along a dimension may be shorter. Along each dimension the block
size must divide the chunk size or be a multiple of it, so that no block is
split unevenly between chunks. A size of 1 keeps that dimension (a value per
channel) and the size of the whole dimension reduces over it.

Only "minmax" is defined so far. Its array has the shape of the block grid
plus a last dimension of 2, holding each block's minimum and maximum, with the
array's own data type. NaN is ignored, and a block holding only NaN has NaN
for both.

add_block_statistics adds statistics to a reference file system, inline, so
that they travel with it into a directory, a kerchunk JSON file or an
Icechunk repo. write_block_statistics adds them to an array in a writable
Zarr store. find_range uses them to find a range of a sorted array.
"""

from __future__ import annotations

import copy
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import zarr

from .builder import chunk_key, inline_value, metadata_key, zarr_data_type

STATISTICS_ATTRIBUTE = "block_statistics"
STATS_GROUP = ".stats"
TYPES = ("minmax",)
SORTED_VALUES = ("ascending",)

# Statistics arrays are split into chunks of about this many bytes along their first axis.
_STATS_CHUNK_BYTES = 2**20
_STATS_CODECS = [
    {"name": "bytes", "configuration": {"endian": "little"}},
    {"name": "zstd", "configuration": {"level": 3, "checksum": False}},
]


# -- Blocks --


def resolve_block_shape(
    shape: Sequence[int],
    chunk_shape: Sequence[int],
    block_shape: Sequence[int | None] | None = None,
    granularity: int | Sequence[int] | None = None,
) -> list[int]:
    """The block shape, in elements, for an array of shape and chunk_shape.

    Give block_shape in elements, where None means the whole dimension, or
    granularity, a number of chunks per block for every dimension or one
    number for all of them. With neither, a block is a chunk.
    """
    shape = [int(s) for s in shape]
    chunk_shape = [int(c) for c in chunk_shape]
    if block_shape is not None and granularity is not None:
        raise ValueError("Give block_shape or granularity, not both")
    if granularity is not None:
        if isinstance(granularity, int):
            granularity = [granularity] * len(shape)
        if len(granularity) != len(shape):
            raise ValueError(f"granularity has {len(granularity)} dimensions; the array has {len(shape)}")
        block_shape = [int(g) * c for g, c in zip(granularity, chunk_shape)]
    elif block_shape is None:
        block_shape = chunk_shape
    if len(block_shape) != len(shape):
        raise ValueError(f"block_shape has {len(block_shape)} dimensions; the array has {len(shape)}")
    resolved = []
    for dim, (size, chunk, block) in enumerate(zip(shape, chunk_shape, block_shape)):
        block = max(size, 1) if block is None else int(block)
        if block < 1:
            raise ValueError(f"block_shape[{dim}] is {block}; it must be at least 1")
        whole = block >= size
        if not whole and block % chunk and chunk % block:
            raise ValueError(
                f"block_shape[{dim}] = {block} neither divides the chunk size {chunk} nor is a multiple of it"
            )
        resolved.append(block)
    return resolved


def block_grid(shape: Sequence[int], block_shape: Sequence[int]) -> list[int]:
    """The number of blocks along each dimension."""
    return [max(1, math.ceil(int(s) / int(b))) for s, b in zip(shape, block_shape)]


# -- Computing --


def compute_minmax(array: Any, block_shape: Sequence[int], *, read_bytes: int = 64 * 2**20) -> np.ndarray:
    """The minimum and maximum of every block, as an array of shape (*block_grid, 2).

    array is a zarr Array (or anything with shape, dtype, chunks and numpy
    indexing). It is read in pieces of whole chunks of about read_bytes, so
    memory stays near that. NaN is ignored; a block of only NaN gives NaN.
    """
    shape = [int(s) for s in array.shape]
    chunk_shape = [int(c) for c in array.chunks]
    block_shape = [int(b) for b in block_shape]
    dtype = np.dtype(array.dtype)
    if dtype.kind not in "iufb":
        raise TypeError(f"minmax needs a numeric array, not {dtype}")
    grid = block_grid(shape, block_shape)
    is_float = dtype.kind == "f"
    if is_float:
        low = np.full(grid, np.inf, dtype=dtype)
        high = np.full(grid, -np.inf, dtype=dtype)
        seen = np.zeros(grid, dtype=bool)
    else:
        info = np.iinfo(dtype) if dtype.kind in "iu" else None
        low = np.full(grid, info.max if info else True, dtype=dtype)
        high = np.full(grid, info.min if info else False, dtype=dtype)
    if not shape:
        value = np.asarray(array[...])
        return np.array([value, value], dtype=dtype)
    if 0 in shape:
        return np.stack([low, high], axis=-1)

    # Each chunk is cut into pieces of min(block, chunk) along every dimension;
    # every piece lies in exactly one block. Several chunks along the first
    # dimension are read at once, so that a remote store fetches them concurrently.
    piece = [min(b, c) for b, c in zip(block_shape, chunk_shape)]
    chunk_bytes = dtype.itemsize * int(np.prod(chunk_shape, dtype=np.int64))
    read_shape = [chunk_shape[0] * max(1, read_bytes // max(chunk_bytes, 1))] + chunk_shape[1:]
    read_grid = [math.ceil(s / r) for s, r in zip(shape, read_shape)]
    for coords in np.ndindex(*read_grid):
        start = [i * r for i, r in zip(coords, read_shape)]
        stop = [min(s0 + r, s) for s0, r, s in zip(start, read_shape, shape)]
        data = np.asarray(array[tuple(slice(a, b) for a, b in zip(start, stop))])
        n_pieces = [math.ceil((b - a) / p) for a, b, p in zip(start, stop, piece)]
        # Repeat edge values to fill the last piece; a repeated value changes neither minimum nor maximum
        pad = [(0, n * p - (b - a)) for n, p, a, b in zip(n_pieces, piece, start, stop)]
        if any(after for _, after in pad):
            data = np.pad(data, pad, mode="edge")
        split = [x for n, p in zip(n_pieces, piece) for x in (n, p)]
        pieces = data.reshape(split)
        axes = tuple(range(1, 2 * len(shape), 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            piece_low = np.nanmin(pieces, axis=axes) if is_float else pieces.min(axis=axes)
            piece_high = np.nanmax(pieces, axis=axes) if is_float else pieces.max(axis=axes)
        # The block of every piece
        index = np.ix_(*[(a + np.arange(n) * p) // b for a, n, p, b in zip(start, n_pieces, piece, block_shape)])
        if is_float:
            valid = ~np.isnan(piece_low)
            np.logical_or.at(seen, index, valid)
            np.fmin.at(low, index, piece_low)
            np.fmax.at(high, index, piece_high)
        else:
            np.minimum.at(low, index, piece_low)
            np.maximum.at(high, index, piece_high)
    if is_float:
        low[~seen] = np.nan
        high[~seen] = np.nan
    return np.stack([low, high], axis=-1)


_COMPUTE = {"minmax": compute_minmax}


def compute_block_statistics(
    array: Any,
    statistics: Sequence[str] = ("minmax",),
    *,
    block_shape: Sequence[int | None] | None = None,
    granularity: int | Sequence[int] | None = None,
) -> tuple[list[int], dict[str, np.ndarray]]:
    """Compute statistics of the blocks of array, without storing them.

    Returns the resolved block shape and each statistic's array.
    """
    resolved = resolve_block_shape(array.shape, array.chunks, block_shape, granularity)
    values = {}
    for name in statistics:
        if name not in _COMPUTE:
            raise ValueError(f"Unknown statistic {name!r}; known: {', '.join(TYPES)}")
        values[name] = _COMPUTE[name](array, resolved)
    return resolved, values


def _split(array_path: str) -> tuple[str, str]:
    """The parent group's path and the array's name."""
    array_path = array_path.strip("/")
    if not array_path:
        raise ValueError("Block statistics need an array inside a group, not one at the root of a store")
    parent, _, name = array_path.rpartition("/")
    return parent, name


def _stats_path(array_name: str, statistic: str) -> str:
    """Where a statistic of the array array_name is, relative to the array's parent group."""
    return f"{STATS_GROUP}/{array_name}/{statistic}"


def _join(parent: str, path: str) -> str:
    return f"{parent}/{path}" if parent else path


def _stats_chunk_shape(shape: Sequence[int], itemsize: int) -> list[int]:
    shape = [int(s) for s in shape]
    row_bytes = itemsize * int(np.prod(shape[1:], dtype=np.int64))
    rows = max(1, _STATS_CHUNK_BYTES // max(row_bytes, 1))
    return [min(shape[0], rows) if shape[0] else 1] + [max(s, 1) for s in shape[1:]]


def _statistic_entry(name: str, block_shape: list[int], path: str) -> dict:
    return {"type": name, "block_shape": block_shape, "path": path}


def _check_sorted(sorted_: str | None) -> None:
    if sorted_ is not None and sorted_ not in SORTED_VALUES:
        raise ValueError(f"sorted must be one of {SORTED_VALUES} or None, not {sorted_!r}")


def _merge_attribute(existing: dict | None, entries: list[dict], sorted_: str | None) -> dict:
    attribute = copy.deepcopy(existing) if existing else {"statistics": []}
    kept = [e for e in attribute.get("statistics", []) if e["path"] not in {n["path"] for n in entries}]
    attribute["statistics"] = kept + entries
    if sorted_ is not None:
        attribute["sorted"] = sorted_
    return attribute


# -- Writing into a reference file system --


def add_block_statistics(
    rfs: dict,
    arrays: Mapping[str, Mapping[str, Any]],
    *,
    validate_sources: bool = True,
) -> dict:
    """Return a copy of rfs with statistics of some of its arrays added inline.

    arrays maps each array's path to its options: "statistics" (default
    ["minmax"]), "block_shape" or "granularity" (see resolve_block_shape),
    and "sorted" ("ascending", or None when the array isn't known to be
    sorted). The array's data is read through the references, so this reads
    every chunk of those arrays once.

    The statistics are inline chunks in the references, so write_rfs and
    rfs_to_icechunk carry them like any other small array. Each statistics
    array records the sources (size and ETag of every referenced file) it
    was computed from.

        rfs = add_block_statistics(rfs, {
            "acquisition/lick_times/timestamps": {"block_shape": [10_000], "sorted": "ascending"},
        })
    """
    from .rfs_store import RfsStore

    result = copy.deepcopy(rfs)
    refs = result["refs"]
    store = RfsStore(rfs, validate_sources=validate_sources)
    root = zarr.open_group(store, mode="r", zarr_format=3)
    created_groups: set[str] = set()
    for array_path, options in arrays.items():
        array_path = array_path.strip("/")
        options = dict(options)
        unknown = set(options) - {"statistics", "block_shape", "granularity", "sorted"}
        if unknown:
            raise ValueError(f"Unknown options for {array_path}: {sorted(unknown)}")
        sorted_ = options.get("sorted")
        _check_sorted(sorted_)
        array = root[array_path]
        if not isinstance(array, zarr.Array):
            raise ValueError(f"{array_path} is not an array")
        resolved, values = compute_block_statistics(
            array,
            options.get("statistics", ("minmax",)),
            block_shape=options.get("block_shape"),
            granularity=options.get("granularity"),
        )
        parent, array_name = _split(array_path)
        entries = []
        for name, value in values.items():
            path = _stats_path(array_name, name)
            full = _join(parent, path)
            _add_groups(refs, full, created_groups)
            _add_inline_array(refs, full, value, {
                "type": name,
                "block_shape": resolved,
                "array": array_name,
                "sources": copy.deepcopy(rfs.get("sources", {})),
            })
            entries.append(_statistic_entry(name, resolved, path))
        meta_key = metadata_key(array_path)
        meta = json.loads(refs[meta_key])
        attributes = meta.setdefault("attributes", {})
        attributes[STATISTICS_ATTRIBUTE] = _merge_attribute(attributes.get(STATISTICS_ATTRIBUTE), entries, sorted_)
        refs[meta_key] = json.dumps(meta, separators=(",", ":"))
    return result


def _add_groups(refs: dict, path: str, created: set[str]) -> None:
    parts = path.split("/")[:-1]
    for i in range(1, len(parts) + 1):
        group = "/".join(parts[:i])
        key = metadata_key(group)
        if group in created or key in refs:
            continue
        refs[key] = json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}, separators=(",", ":"))
        created.add(group)


def _add_inline_array(refs: dict, path: str, value: np.ndarray, attributes: dict) -> None:
    value = np.ascontiguousarray(value)
    little = value.astype(value.dtype.newbyteorder("<"), copy=False)
    chunks = _stats_chunk_shape(value.shape, value.dtype.itemsize)
    meta = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(value.shape),
        "data_type": zarr_data_type(value.dtype),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunks}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": False if value.dtype.kind == "b" else 0,
        "codecs": _STATS_CODECS,
        "attributes": attributes,
        "storage_transformers": [],
    }
    for key in [k for k in refs if k.startswith(f"{path}/")]:
        del refs[key]
    refs[metadata_key(path)] = json.dumps(meta, separators=(",", ":"))
    import numcodecs

    zstd = numcodecs.Zstd(level=3)
    grid = [math.ceil(s / c) for s, c in zip(value.shape, chunks)]
    for coords in np.ndindex(*grid):
        region = tuple(slice(i * c, (i + 1) * c) for i, c in zip(coords, chunks))
        block = little[region]
        # Edge chunks are stored whole, padded, as Zarr expects
        full = np.zeros(chunks, dtype=little.dtype)
        full[tuple(slice(0, s) for s in block.shape)] = block
        refs[chunk_key(path, coords)] = inline_value(zstd.encode(full.tobytes()))


# -- Writing into a Zarr store --


def write_block_statistics(
    array: zarr.Array,
    statistics: Sequence[str] = ("minmax",),
    *,
    block_shape: Sequence[int | None] | None = None,
    granularity: int | Sequence[int] | None = None,
    sorted: str | None = None,  # noqa: A002 - the attribute's own name
) -> dict:
    """Compute statistics of array and write them next to it, in its writable store.

    Returns the array's new "block_statistics" attribute.
    """
    _check_sorted(sorted)
    parent_path, array_name = _split(array.path)
    parent = zarr.open_group(array.store, path=parent_path, mode="r+", zarr_format=3)
    resolved, values = compute_block_statistics(
        array, statistics, block_shape=block_shape, granularity=granularity
    )
    entries = []
    for name, value in values.items():
        path = _stats_path(array_name, name)
        stats = parent.create_array(
            path,
            shape=value.shape,
            dtype=value.dtype,
            chunks=_stats_chunk_shape(value.shape, value.dtype.itemsize),
            compressors=[zarr.codecs.ZstdCodec(level=3)],
            fill_value=False if value.dtype.kind == "b" else 0,
            attributes={"type": name, "block_shape": resolved, "array": array_name},
            overwrite=True,
        )
        stats[...] = value
        entries.append(_statistic_entry(name, resolved, path))
    attribute = _merge_attribute(array.attrs.get(STATISTICS_ATTRIBUTE), entries, sorted)
    array.attrs[STATISTICS_ATTRIBUTE] = attribute
    return attribute


# -- Reading --


def load_block_statistics(array: zarr.Array, name: str = "minmax"):
    """The block shape and values of one of array's statistics, or None if it has none."""
    attribute = array.attrs.get(STATISTICS_ATTRIBUTE)
    if not attribute:
        return None
    for entry in attribute.get("statistics", []):
        if entry["type"] == name:
            parent_path, _ = _split(array.path)
            parent = zarr.open_group(array.store, path=parent_path, mode="r", zarr_format=3)
            return [int(b) for b in entry["block_shape"]], np.asarray(parent[entry["path"]][...])
    return None


def find_range(
    array: zarr.Array,
    low: Any,
    high: Any,
) -> slice:
    """The slice of a sorted one-dimensional array holding the values from low to high.

    Equal to slice(np.searchsorted(a, low, "left"), np.searchsorted(a, high,
    "right")), but with the array's minmax statistics it reads only those
    and the one or two blocks where the range starts and ends. An array
    without them, or not marked sorted, is read whole.
    """
    if len(array.shape) != 1:
        raise ValueError("find_range needs a one-dimensional array")
    attribute = array.attrs.get(STATISTICS_ATTRIBUTE) or {}
    loaded = load_block_statistics(array, "minmax") if attribute.get("sorted") == "ascending" else None
    if loaded is None:
        data = np.asarray(array[...])
        return slice(int(np.searchsorted(data, low, "left")), int(np.searchsorted(data, high, "right")))
    (block,), minmax = loaded
    maxima = minmax[:, 1]
    n = int(array.shape[0])

    def position(value: Any, side: str) -> int:
        # Every block before b is entirely before value; the position is in block b
        b = int(np.searchsorted(maxima, value, "left" if side == "left" else "right"))
        if b >= len(maxima):
            return n
        start = b * block
        data = np.asarray(array[start:min(start + block, n)])
        return start + int(np.searchsorted(data, value, side))

    start = position(low, "left")
    stop = position(high, "right")
    return slice(start, max(start, stop))
