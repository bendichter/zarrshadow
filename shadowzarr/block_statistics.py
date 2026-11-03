"""Statistics of blocks of an array, stored next to it, that readers use to skip data.

A reader that wants the part of a sorted timestamps array between two times
would otherwise read the whole array. With the minimum and maximum of every
block of, say, 10,000 values, it reads that small array, bisects it to find
the one or two blocks holding the times, and reads only those. Tabular formats
call these per-block statistics zone maps; Parquet stores them for every row
group. The convention is described in docs/block-statistics.md.

The statistics of an array named "<name>" are Zarr arrays next to it, in a
".stats" group in the same parent group, at ".stats/<name>/<type>/<shape>",
where <shape> is the block shape with its sizes joined by "x". A statistic can
have several block shapes, one array each. A copy of the parent group keeps them. The array lists them in its "block_statistics"
attribute, with paths relative to its parent group:

    "block_statistics": {
        "sorted": "ascending",
        "statistics": [
            {"type": "minmax", "block_shape": [10000], "path": ".stats/timestamps/minmax/10000"}
        ]
    }

A block is block_shape elements of the array, counted from its origin; the
last block along a dimension may be shorter. Along each dimension the block
size must divide the chunk size or be a multiple of it, so that no block is
split unevenly between chunks. A size of 1 keeps that dimension (a value per
channel) and the size of the whole dimension reduces over it.

Three statistics are defined. "minmax" has the shape of the block grid plus
a last dimension of 2, holding each block's minimum and maximum, with the
array's own data type; NaN is ignored. "missing_count" counts each block's
missing values: NaN for floats by default, or values named in "missing".
"moments" holds each block's count of valid values with their sum and sum of
squares (int64, exact, for integers of up to 16 bits) or mean and M2 (float64),
from which means and variances of any set of blocks follow.

A statistic can have several block shapes. When each coarser block is a
whole number of finer ones, they are levels: coarser levels are rolled up
from finer ones without reading the data again, and readers start at the
coarsest and read only what they need of the finer ones.

add_block_statistics adds statistics to a reference file system, inline, so
that they travel with it into a directory, a kerchunk JSON file or an
Icechunk repo. write_block_statistics adds them to an array in a writable
Zarr store. searchsorted and find_range use them to search a sorted array,
minmax_range to find the minimum and maximum over a range, and moments_range
its count, mean and variance.
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
TYPES = ("minmax", "missing_count", "moments")
SORTED_VALUES = ("ascending",)

# Statistics arrays are split into chunks of about this many bytes along their first axis.
_STATS_CHUNK_BYTES = 2**20
# Readers skip coarser levels when reading this much of a finer one is enough.
DEFAULT_MAX_BYTES = 2**20
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


def resolve_missing(dtype: Any, missing: Sequence[Any] | None = None) -> list[Any]:
    """The values that count as missing in an array of dtype.

    By default NaN for floating-point arrays and nothing for others. A list
    names them explicitly, for example [-32768] for a sentinel, or the
    array's fill value when it marks data that was never written.
    """
    dtype = np.dtype(dtype)
    if missing is None:
        return [float("nan")] if dtype.kind == "f" else []
    return [float("nan") if (isinstance(m, str) and m == "NaN") or (isinstance(m, float) and np.isnan(m)) else m
            for m in missing]


def _missing_json(missing: Sequence[Any]) -> list[Any]:
    """missing as JSON can hold it, with NaN as the string "NaN"."""
    return ["NaN" if isinstance(m, float) and np.isnan(m) else (m.item() if hasattr(m, "item") else m) for m in missing]


def _valid(data: np.ndarray, missing: Sequence[Any]) -> np.ndarray:
    """Where data holds a value that isn't missing."""
    valid = np.ones(data.shape, dtype=bool)
    for m in missing:
        if isinstance(m, float) and np.isnan(m):
            if data.dtype.kind == "f":
                valid &= ~np.isnan(data)
        else:
            valid &= data != m
    return valid


def moments_dtype(dtype: Any) -> np.dtype:
    """int64 (count, sum, sum_sq) for integers of up to 16 bits, which is exact; float64 (count, mean, M2) otherwise."""
    dtype = np.dtype(dtype)
    return np.dtype("int64") if dtype.kind in "iub" and dtype.itemsize <= 2 else np.dtype("float64")


def _pieces(array: Any, block_shape: Sequence[int], read_bytes: int):
    """Yield (data, valid, index) for every piece of every chunk, read a few chunks at a time.

    data and valid have shape (n0, p0, n1, p1, ...): pieces of
    min(block, chunk) elements, with the short last piece padded and marked
    not valid. index (from np.ix_) gives each piece's block.
    """
    shape = [int(s) for s in array.shape]
    chunk_shape = [int(c) for c in array.chunks]
    block_shape = [int(b) for b in block_shape]
    piece = [min(b, c) for b, c in zip(block_shape, chunk_shape)]
    chunk_bytes = np.dtype(array.dtype).itemsize * int(np.prod(chunk_shape, dtype=np.int64))
    read_shape = [chunk_shape[0] * max(1, read_bytes // max(chunk_bytes, 1))] + chunk_shape[1:]
    read_grid = [math.ceil(s / r) for s, r in zip(shape, read_shape)]
    for coords in np.ndindex(*read_grid):
        start = [i * r for i, r in zip(coords, read_shape)]
        stop = [min(s0 + r, s) for s0, r, s in zip(start, read_shape, shape)]
        data = np.asarray(array[tuple(slice(a, b) for a, b in zip(start, stop))])
        n_pieces = [math.ceil((b - a) / p) for a, b, p in zip(start, stop, piece)]
        pad = [(0, n * p - (b - a)) for n, p, a, b in zip(n_pieces, piece, start, stop)]
        inside = np.ones(data.shape, dtype=bool)
        if any(after for _, after in pad):
            data = np.pad(data, pad)
            inside = np.pad(inside, pad)
        split = [x for n, p in zip(n_pieces, piece) for x in (n, p)]
        index = np.ix_(*[(a + np.arange(n) * p) // b for a, n, p, b in zip(start, n_pieces, piece, block_shape)])
        yield data.reshape(split), inside.reshape(split), index


class _MinMaxAccumulator:
    def __init__(self, grid, dtype, missing):
        self.dtype = np.dtype(dtype)
        if self.dtype.kind not in "iufb":
            raise TypeError(f"minmax needs a numeric array, not {self.dtype}")
        self.is_float = self.dtype.kind == "f"
        if self.is_float:
            self.low = np.full(grid, np.inf, dtype=self.dtype)
            self.high = np.full(grid, -np.inf, dtype=self.dtype)
            self.seen = np.zeros(grid, dtype=bool)
            self.empty_low = self.empty_high = np.nan
        else:
            info = np.iinfo(self.dtype) if self.dtype.kind in "iu" else None
            self.empty_low = info.max if info else True
            self.empty_high = info.min if info else False
            self.low = np.full(grid, self.empty_low, dtype=self.dtype)
            self.high = np.full(grid, self.empty_high, dtype=self.dtype)

    def update(self, data, inside, index, axes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if self.is_float:
                values = np.where(inside, data, np.nan)
                piece_low, piece_high = np.nanmin(values, axis=axes), np.nanmax(values, axis=axes)
                np.logical_or.at(self.seen, index, ~np.isnan(piece_low))
                np.fmin.at(self.low, index, piece_low)
                np.fmax.at(self.high, index, piece_high)
            else:
                np.minimum.at(self.low, index, np.where(inside, data, self.empty_low).min(axis=axes))
                np.maximum.at(self.high, index, np.where(inside, data, self.empty_high).max(axis=axes))

    def result(self):
        if self.is_float:
            self.low[~self.seen] = np.nan
            self.high[~self.seen] = np.nan
        return np.stack([self.low, self.high], axis=-1)


class _MissingCountAccumulator:
    def __init__(self, grid, dtype, missing):
        self.missing = missing
        self.count = np.zeros(grid, dtype=np.int64)

    def update(self, data, inside, index, axes):
        np.add.at(self.count, index, (inside & ~_valid(data, self.missing)).sum(axis=axes))

    def result(self):
        return self.count


class _MomentsAccumulator:
    def __init__(self, grid, dtype, missing):
        self.missing = missing
        self.ndim = len(grid)
        self.exact = moments_dtype(dtype).kind == "i"
        self.count = np.zeros(grid, dtype=np.int64)
        if self.exact:
            self.total = np.zeros(grid, dtype=np.int64)
            self.squares = np.zeros(grid, dtype=np.int64)
        else:
            self.shift = np.full(grid, np.nan)
            self.total = np.zeros(grid)
            self.squares = np.zeros(grid)

    def update(self, data, inside, index, axes):
        keep = inside & _valid(data, self.missing)
        piece_count = keep.sum(axis=axes)
        np.add.at(self.count, index, piece_count)
        if self.exact:
            values = np.where(keep, data, 0).astype(np.int64)
            np.add.at(self.total, index, values.sum(axis=axes))
            np.add.at(self.squares, index, (values * values).sum(axis=axes))
            return
        values = np.where(keep, data, 0).astype(np.float64)
        # Each block's shift is the mean of the first piece with values that reaches it,
        # so that the sums below are of small deviations and keep their precision
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            piece_mean = values.sum(axis=axes) / piece_count
        unset = np.isnan(self.shift[index]) & ~np.isnan(piece_mean)
        if unset.any():
            self.shift[index] = np.where(unset, piece_mean, self.shift[index])
        expand = tuple(slice(None) if i % 2 == 0 else None for i in range(2 * self.ndim))
        centred = np.where(keep, values - np.nan_to_num(self.shift[index])[expand], 0.0)
        np.add.at(self.total, index, centred.sum(axis=axes))
        np.add.at(self.squares, index, (centred * centred).sum(axis=axes))

    def result(self):
        if self.exact:
            return np.stack([self.count, self.total, self.squares], axis=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            has = self.count > 0
            mean = np.where(has, np.nan_to_num(self.shift) + self.total / np.maximum(self.count, 1), np.nan)
            m2 = np.where(has, self.squares - self.total * self.total / np.maximum(self.count, 1), np.nan)
        return np.stack([self.count.astype(np.float64), mean, np.where(has, np.maximum(m2, 0.0), np.nan)], axis=-1)


_ACCUMULATORS = {"minmax": _MinMaxAccumulator, "missing_count": _MissingCountAccumulator, "moments": _MomentsAccumulator}


class _AsOneDimensional:
    """A zero-dimensional array seen as one element along one dimension."""

    def __init__(self, array):
        self._value = np.asarray(array[...]).reshape(1)
        self.shape, self.chunks, self.dtype = (1,), (1,), self._value.dtype

    def __getitem__(self, key):
        return self._value[key]


def compute_statistics(
    array: Any,
    block_shape: Sequence[int],
    statistics: Sequence[str] = ("minmax",),
    missing: Sequence[Any] | None = None,
    *,
    read_bytes: int = 64 * 2**20,
) -> dict[str, np.ndarray]:
    """Compute several statistics of the blocks of array in one pass over its data.

    block_shape must already be resolved (see resolve_block_shape). The array
    is read in pieces of whole chunks of about read_bytes, so memory stays
    near that.
    """
    for name in statistics:
        if name not in _ACCUMULATORS:
            raise ValueError(f"Unknown statistic {name!r}; known: {', '.join(TYPES)}")
    missing = resolve_missing(array.dtype, missing)
    scalar = len(array.shape) == 0
    if scalar:
        array, block_shape = _AsOneDimensional(array), [1]
    grid = block_grid(array.shape, block_shape)
    accumulators = {name: _ACCUMULATORS[name](grid, array.dtype, missing) for name in statistics}
    if 0 not in [int(s) for s in array.shape]:
        axes = tuple(range(1, 2 * len(grid), 2))
        for data, inside, index in _pieces(array, block_shape, read_bytes):
            for accumulator in accumulators.values():
                accumulator.update(data, inside, index, axes)
    results = {name: accumulator.result() for name, accumulator in accumulators.items()}
    return {name: value[0] for name, value in results.items()} if scalar else results


def compute_minmax(array: Any, block_shape: Sequence[int], *, read_bytes: int = 64 * 2**20) -> np.ndarray:
    """The minimum and maximum of every block, shape (*block_grid, 2), in the array's data type.

    NaN is ignored; a block of only NaN gives NaN.
    """
    return compute_statistics(array, block_shape, ["minmax"], read_bytes=read_bytes)["minmax"]


def compute_missing_count(
    array: Any, block_shape: Sequence[int], missing: Sequence[Any] | None = None, *, read_bytes: int = 64 * 2**20
) -> np.ndarray:
    """The number of missing values (see resolve_missing) in every block, int64 of the block grid's shape."""
    return compute_statistics(array, block_shape, ["missing_count"], missing, read_bytes=read_bytes)["missing_count"]


def compute_moments(
    array: Any, block_shape: Sequence[int], missing: Sequence[Any] | None = None, *, read_bytes: int = 64 * 2**20
) -> np.ndarray:
    """The count of valid values and their moments in every block, shape (*block_grid, 3).

    For integers of up to 16 bits, int64 (count, sum, sum of squares), exact.
    Otherwise float64 (count, mean, M2), M2 being the sum of squared
    deviations from the block's mean, accumulated about a shift taken from
    the block's own data so that precision isn't lost. Missing values
    (see resolve_missing) are left out.
    """
    return compute_statistics(array, block_shape, ["moments"], missing, read_bytes=read_bytes)["moments"]


def compute_block_statistics(
    array: Any,
    statistics: Sequence[str] = ("minmax",),
    *,
    block_shape: Sequence[int | None] | None = None,
    granularity: int | Sequence[int] | None = None,
    missing: Sequence[Any] | None = None,
) -> tuple[list[int], dict[str, np.ndarray]]:
    """Compute statistics of the blocks of array, without storing them.

    missing names the values missing_count counts and moments leave out (see
    resolve_missing). Returns the resolved block shape and each statistic's array.
    """
    resolved = resolve_block_shape(array.shape, array.chunks, block_shape, granularity)
    return resolved, compute_statistics(array, resolved, statistics, missing)


def _nests(fine: Sequence[int], coarse: Sequence[int], shape: Sequence[int]) -> bool:
    """Whether every coarse block is a whole number of fine blocks along each dimension."""
    return all(c % f == 0 or c >= s for f, c, s in zip(fine, coarse, shape))


def rollup_minmax(
    values: np.ndarray, fine_block_shape: Sequence[int], coarse_block_shape: Sequence[int], shape: Sequence[int]
) -> np.ndarray:
    """The minmax of coarse blocks, from the minmax of fine blocks of an array of shape.

    Each coarse block must be a whole number of fine blocks along every
    dimension, or cover the whole dimension. No data is read, and the result
    equals compute_minmax at the coarse block shape.
    """
    fine = [int(b) for b in fine_block_shape]
    coarse = [int(b) for b in coarse_block_shape]
    shape = [int(s) for s in shape]
    if not _nests(fine, coarse, shape):
        raise ValueError(f"Blocks of {coarse} are not whole numbers of blocks of {fine}")
    fine_grid = list(values.shape[:-1])
    factor = [n if c >= s else c // f for n, f, c, s in zip(fine_grid, fine, coarse, shape)]
    groups = [math.ceil(n / k) for n, k in zip(fine_grid, factor)]
    # Repeat edge blocks to fill the last group; a repeated value changes neither minimum nor maximum
    pad = [(0, g * k - n) for g, k, n in zip(groups, factor, fine_grid)] + [(0, 0)]
    if any(after for _, after in pad):
        values = np.pad(values, pad, mode="edge")
    split = [x for g, k in zip(groups, factor) for x in (g, k)] + [2]
    pieces = values.reshape(split)
    axes = tuple(range(1, 2 * len(fine_grid), 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if values.dtype.kind == "f":
            low, high = np.nanmin(pieces[..., 0], axis=axes), np.nanmax(pieces[..., 1], axis=axes)
        else:
            low, high = pieces[..., 0].min(axis=axes), pieces[..., 1].max(axis=axes)
    return np.stack([low, high], axis=-1)


def _group(values: np.ndarray, fine: Sequence[int], coarse: Sequence[int], shape: Sequence[int], trailing: int):
    """Fine blocks grouped into coarse ones: shape (g0, k0, g1, k1, ..., *trailing dims), padded with zeros."""
    fine = [int(b) for b in fine]
    coarse = [int(b) for b in coarse]
    shape = [int(s) for s in shape]
    if not _nests(fine, coarse, shape):
        raise ValueError(f"Blocks of {coarse} are not whole numbers of blocks of {fine}")
    fine_grid = list(values.shape[: values.ndim - trailing])
    factor = [n if c >= s else c // f for n, f, c, s in zip(fine_grid, fine, coarse, shape)]
    groups = [math.ceil(n / k) for n, k in zip(fine_grid, factor)]
    pad = [(0, g * k - n) for g, k, n in zip(groups, factor, fine_grid)] + [(0, 0)] * trailing
    if any(after for _, after in pad):
        values = np.pad(values, pad)
    split = [x for g, k in zip(groups, factor) for x in (g, k)] + list(values.shape[values.ndim - trailing:])
    return values.reshape(split), tuple(range(1, 2 * len(fine_grid), 2))


def rollup_missing_count(values, fine_block_shape, coarse_block_shape, shape) -> np.ndarray:
    """The missing_count of coarse blocks, from that of fine blocks."""
    grouped, axes = _group(values, fine_block_shape, coarse_block_shape, shape, 0)
    return grouped.sum(axis=axes)


def _merge_moments(parts: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """Merge moments along axes. parts has a last dimension of 3."""
    if parts.dtype.kind == "i":
        return parts.sum(axis=axes)
    count, mean, m2 = parts[..., 0], parts[..., 1], parts[..., 2]
    total = count.sum(axis=axes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        weighted = np.where(count > 0, count * np.nan_to_num(mean), 0.0)
        merged_mean = weighted.sum(axis=axes) / total
        expanded = np.expand_dims(merged_mean, axes)
        spread = np.where(count > 0, np.nan_to_num(m2) + count * (np.nan_to_num(mean) - expanded) ** 2, 0.0)
        merged_m2 = spread.sum(axis=axes)
    merged_mean = np.where(total > 0, merged_mean, np.nan)
    merged_m2 = np.where(total > 0, merged_m2, np.nan)
    return np.stack([total, merged_mean, merged_m2], axis=-1)


def rollup_moments(values, fine_block_shape, coarse_block_shape, shape) -> np.ndarray:
    """The moments of coarse blocks, from those of fine blocks."""
    grouped, axes = _group(values, fine_block_shape, coarse_block_shape, shape, 1)
    return _merge_moments(grouped, axes)


_ROLLUP = {"minmax": rollup_minmax, "missing_count": rollup_missing_count, "moments": rollup_moments}


def compute_block_statistics_levels(
    array: Any,
    statistics: Sequence[str] = ("minmax",),
    block_shapes: Sequence[Sequence[int | None]] = (),
    missing: Sequence[Any] | None = None,
) -> list[tuple[list[int], dict[str, np.ndarray]]]:
    """Compute statistics at several block shapes, reading the data as few times as possible.

    Block shapes are computed from the smallest up. A block shape that is a
    whole number of an already computed one is rolled up from it without
    reading the data; any other is computed from the data. Returns the
    resolved block shape and values of each, in the order given.
    """
    resolved = [resolve_block_shape(array.shape, array.chunks, b) for b in block_shapes]
    order = sorted(range(len(resolved)), key=lambda i: int(np.prod(resolved[i], dtype=np.int64)))
    for name in statistics:
        if name not in _ACCUMULATORS:
            raise ValueError(f"Unknown statistic {name!r}; known: {', '.join(TYPES)}")
    done: dict[int, dict[str, np.ndarray]] = {}
    for i in order:
        source = next(
            (j for j in reversed([j for j in order if j in done]) if _nests(resolved[j], resolved[i], array.shape)),
            None,
        )
        if source is None:
            # Every statistic of this level in one pass over the data
            done[i] = compute_statistics(array, resolved[i], statistics, missing)
        elif resolved[source] == resolved[i]:
            done[i] = done[source]
        else:
            done[i] = {
                name: _ROLLUP[name](done[source][name], resolved[source], resolved[i], array.shape)
                for name in statistics
            }
    return [(resolved[i], done[i]) for i in range(len(resolved))]


def _split(array_path: str) -> tuple[str, str]:
    """The parent group's path and the array's name."""
    array_path = array_path.strip("/")
    if not array_path:
        raise ValueError("Block statistics need an array inside a group, not one at the root of a store")
    parent, _, name = array_path.rpartition("/")
    return parent, name


def _stats_path(array_name: str, statistic: str, block_shape: Sequence[int]) -> str:
    """Where a statistic of the array array_name is, relative to the array's parent group.

    Each type is a group holding one array per block shape, named by its
    sizes joined with "x", so that the levels of one statistic sit together.
    """
    return f"{STATS_GROUP}/{array_name}/{statistic}/{'x'.join(str(int(b)) for b in block_shape) or 'scalar'}"


def _join(parent: str, path: str) -> str:
    return f"{parent}/{path}" if parent else path


def _stats_chunk_shape(shape: Sequence[int], itemsize: int) -> list[int]:
    shape = [int(s) for s in shape]
    row_bytes = itemsize * int(np.prod(shape[1:], dtype=np.int64))
    rows = max(1, _STATS_CHUNK_BYTES // max(row_bytes, 1))
    return [min(shape[0], rows) if shape[0] else 1] + [max(s, 1) for s in shape[1:]]


def _statistic_entry(name: str, block_shape: list[int], path: str, missing: list[Any] | None = None) -> dict:
    entry = {"type": name, "block_shape": block_shape, "path": path}
    if name in ("missing_count", "moments") and missing is not None:
        entry["missing"] = _missing_json(missing)
    return entry


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
    arrays: Mapping[str, Mapping[str, Any] | Sequence[Mapping[str, Any]]],
    *,
    validate_sources: bool = True,
) -> dict:
    """Return a copy of rfs with statistics of some of its arrays added inline.

    arrays maps each array's path to its options: "statistics" (default
    ["minmax"]), "block_shape" or "granularity" (see resolve_block_shape),
    and "sorted" ("ascending", or None when the array isn't known to be
    sorted), and "missing" (the values missing_count counts and moments
    leave out; see resolve_missing). A list of option sets adds several
    levels, one per block shape.
    The array's data is read through the references once; a level whose
    blocks are whole numbers of a finer level's is rolled up from it.

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
    for array_path, given in arrays.items():
        array_path = array_path.strip("/")
        option_sets = [dict(given)] if isinstance(given, Mapping) else [dict(o) for o in given]
        sorted_ = None
        for options in option_sets:
            unknown = set(options) - {"statistics", "block_shape", "granularity", "sorted", "missing"}
            if unknown:
                raise ValueError(f"Unknown options for {array_path}: {sorted(unknown)}")
            _check_sorted(options.get("sorted"))
            sorted_ = options.get("sorted") or sorted_
        array = root[array_path]
        if not isinstance(array, zarr.Array):
            raise ValueError(f"{array_path} is not an array")
        # Each option set is one level; levels that share statistics are computed together
        levels = []
        by_statistics: dict[tuple, list[int]] = {}
        for i, options in enumerate(option_sets):
            shape = resolve_block_shape(array.shape, array.chunks, options.get("block_shape"), options.get("granularity"))
            missing = resolve_missing(array.dtype, options.get("missing"))
            key = (tuple(options.get("statistics", ("minmax",))), json.dumps(_missing_json(missing)))
            by_statistics.setdefault(key, []).append(i)
            levels.append(shape)
        parent, array_name = _split(array_path)
        entries = []
        for (statistics, missing_key), indexes in by_statistics.items():
            missing = resolve_missing(array.dtype, json.loads(missing_key))
            computed = compute_block_statistics_levels(array, statistics, [levels[i] for i in indexes], missing)
            for resolved, values in computed:
                for name, value in values.items():
                    path = _stats_path(array_name, name, resolved)
                    full = _join(parent, path)
                    _add_groups(refs, full, created_groups)
                    entry = _statistic_entry(name, resolved, path, missing)
                    _add_inline_array(refs, full, value, {
                        **{k: v for k, v in entry.items() if k != "path"},
                        "array": array_name,
                        "sources": copy.deepcopy(rfs.get("sources", {})),
                    })
                    entries.append(entry)
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
    levels: Sequence[Sequence[int | None]] | None = None,
    missing: Sequence[Any] | None = None,
    sorted: str | None = None,  # noqa: A002 - the attribute's own name
) -> dict:
    """Compute statistics of array and write them next to it, in its writable store.

    levels, a list of block shapes, writes several levels at once: the data
    is read once, and coarser levels are rolled up from finer ones. Returns
    the array's new "block_statistics" attribute.
    """
    _check_sorted(sorted)
    parent_path, array_name = _split(array.path)
    parent = zarr.open_group(array.store, path=parent_path, mode="r+", zarr_format=3)
    if levels is not None:
        if block_shape is not None or granularity is not None:
            raise ValueError("Give levels, or block_shape or granularity, not both")
        computed = compute_block_statistics_levels(array, statistics, levels, missing)
    else:
        computed = [compute_block_statistics(
            array, statistics, block_shape=block_shape, granularity=granularity, missing=missing
        )]
    resolved_missing = resolve_missing(array.dtype, missing)
    entries = []
    for resolved, values in computed:
        entries += _write_level(parent, array_name, resolved, values, resolved_missing)
    attribute = _merge_attribute(array.attrs.get(STATISTICS_ATTRIBUTE), entries, sorted)
    array.attrs[STATISTICS_ATTRIBUTE] = attribute
    return attribute


def _write_level(parent: zarr.Group, array_name: str, resolved: list[int], values: dict, missing) -> list[dict]:
    entries = []
    for name, value in values.items():
        entry = _statistic_entry(name, resolved, _stats_path(array_name, name, resolved), missing)
        path = _stats_path(array_name, name, resolved)
        stats = parent.create_array(
            path,
            shape=value.shape,
            dtype=value.dtype,
            chunks=_stats_chunk_shape(value.shape, value.dtype.itemsize),
            compressors=[zarr.codecs.ZstdCodec(level=3)],
            fill_value=False if value.dtype.kind == "b" else 0,
            attributes={**{k: v for k, v in entry.items() if k != "path"}, "array": array_name},
            overwrite=True,
        )
        stats[...] = value
        entries.append(entry)
    return entries


# -- Reading --


def load_block_statistics(array: zarr.Array, name: str = "minmax", block_shape: Sequence[int] | None = None):
    """The block shape and values of one of array's statistics, or None if it has none.

    With several block shapes of the statistic, block_shape picks one; by
    default the one with the smallest blocks.
    """
    attribute = array.attrs.get(STATISTICS_ATTRIBUTE)
    if not attribute:
        return None
    entries = [e for e in attribute.get("statistics", []) if e["type"] == name]
    if block_shape is not None:
        entries = [e for e in entries if [int(b) for b in e["block_shape"]] == [int(b) for b in block_shape]]
    if not entries:
        return None
    entry = min(entries, key=lambda e: int(np.prod(e["block_shape"], dtype=np.int64)))
    parent_path, _ = _split(array.path)
    parent = zarr.open_group(array.store, path=parent_path, mode="r", zarr_format=3)
    return [int(b) for b in entry["block_shape"]], np.asarray(parent[entry["path"]][...])


class _LazyStatistics:
    """A statistics array that is opened when first read.

    Its shape and data type follow from the array it describes, so a reader
    can weigh levels without fetching their metadata.
    """

    def __init__(self, array: zarr.Array, path: str, block_shape: list[int], kind: str = "minmax"):
        self._array, self._path = array, path
        grid = tuple(block_grid(array.shape, block_shape))
        if kind == "minmax":
            self.shape, self.dtype = grid + (2,), np.dtype(array.dtype)
        elif kind == "moments":
            self.shape, self.dtype = grid + (3,), moments_dtype(array.dtype)
        else:
            self.shape, self.dtype = grid, np.dtype("int64")
        self._opened = None

    def open(self) -> zarr.Array:
        if self._opened is None:
            parent_path, _ = _split(self._array.path)
            parent = zarr.open_group(self._array.store, path=parent_path, mode="r", zarr_format=3)
            self._opened = parent[self._path]
        return self._opened

    def __getitem__(self, key):
        return self.open()[key]


def _read_all(reads: list[tuple[Any, tuple]]) -> list[np.ndarray]:
    """Read several selections at once, each (array, selection), in parallel threads."""
    if len(reads) <= 1:
        return [np.asarray(a[sel]) for a, sel in reads]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(len(reads), 16)) as pool:
        return list(pool.map(lambda read: np.asarray(read[0][read[1]]), reads))


def _levels(array: zarr.Array, kind: str, missing: Any = None) -> list[tuple[list[int], _LazyStatistics]]:
    """The array's statistics of one type as (block shape, statistics array), smallest blocks first.

    For types that leave out missing values, only entries with the given
    missing values (as JSON) count; by default those of the finest entry.
    """
    attribute = array.attrs.get(STATISTICS_ATTRIBUTE) or {}
    entries = [e for e in attribute.get("statistics", []) if e["type"] == kind]
    entries.sort(key=lambda e: int(np.prod(e["block_shape"], dtype=np.int64)))
    if kind != "minmax" and entries:
        wanted = entries[0].get("missing") if missing is None else missing
        entries = [e for e in entries if e.get("missing") == wanted]
    levels = []
    for e in entries:
        block_shape = [int(b) for b in e["block_shape"]]
        levels.append((block_shape, _LazyStatistics(array, e["path"], block_shape, kind)))
    return levels


def _minmax_levels(array: zarr.Array) -> list[tuple[list[int], _LazyStatistics]]:
    """The array's minmax statistics as (block shape, statistics array), smallest blocks first."""
    return _levels(array, "minmax")


def _nested(levels: list[tuple[list[int], Any]], shape: Sequence[int]) -> list[tuple[list[int], Any]]:
    """From the finest level up, the levels whose blocks are whole numbers of the previous one's."""
    chain: list[tuple[list[int], Any]] = []
    for level in levels:
        if not chain or (level[0] != chain[-1][0] and _nests(chain[-1][0], level[0], shape)):
            chain.append(level)
    return chain


def _runs(blocks: np.ndarray, limit: int):
    """The runs of equal values in nondecreasing blocks, as (block, start, stop), for blocks below limit."""
    if not blocks.size:
        return []
    edges = np.flatnonzero(np.diff(blocks)) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [blocks.size]])
    return [(int(blocks[a]), int(a), int(z)) for a, z in zip(starts, stops) if blocks[a] < limit]


def _refine(blocks, values, n_blocks, ratio, n_fine, read, side):
    """The position of each value one level finer, from its block at this level.

    values are sorted, and blocks[i] is the block of values[i]. Each block
    holds ratio finer entries; read(start, stop) returns the sorted keys of
    the finer entries from start to stop, which are the maxima of a finer
    level or the data itself. Neighbouring blocks are read together.
    """
    out = np.full(values.shape, n_fine, dtype=np.int64)
    runs = _runs(blocks, n_blocks)
    i = 0
    while i < len(runs):
        j = i
        while j + 1 < len(runs) and runs[j + 1][0] == runs[j][0] + 1:
            j += 1
        offset = runs[i][0] * ratio
        keys = read(offset, min((runs[j][0] + 1) * ratio, n_fine))
        for b, a, z in runs[i:j + 1]:
            lo = b * ratio - offset
            out[a:z] = b * ratio + np.searchsorted(keys[lo:lo + ratio], values[a:z], side)
        i = j + 1
    return out


def _level_bytes(level: tuple[list[int], Any], rows: int) -> int:
    """The bytes of rows entries along the first dimension of a statistics array."""
    stats = level[1]
    return rows * int(np.prod(stats.shape[1:], dtype=np.int64)) * stats.dtype.itemsize


def searchsorted(
    array: zarr.Array, values: Any, side: str = "left", *, max_bytes: int = DEFAULT_MAX_BYTES
) -> int | np.ndarray:
    """Where values would go in a sorted one-dimensional array, as np.searchsorted gives.

    With the array's minmax statistics, it bisects the coarsest level, then
    reads only the parts of each finer level and of the data that the values
    fall in: each block once, and runs of neighbouring blocks in one read.
    Levels are used when each one's blocks are whole numbers of the next
    finer one's. Search starts at the finest level of at most max_bytes,
    since reading that whole is no slower than stepping through coarser
    ones, or at the coarsest level when every level is larger. An array
    without statistics, or not marked sorted, is read whole. A scalar gives an int, an array of values an int64 array of their
    shape; the values may be in any order.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', not {side!r}")
    if len(array.shape) != 1:
        raise ValueError("searchsorted needs a one-dimensional array")
    scalar = np.ndim(values) == 0
    values = np.asarray(values)
    attribute = array.attrs.get(STATISTICS_ATTRIBUTE) or {}
    levels = _nested(_minmax_levels(array), array.shape) if attribute.get("sorted") == "ascending" else []
    small = [i for i, level in enumerate(levels) if _level_bytes(level, level[1].shape[0]) <= max_bytes]
    if small:
        levels = levels[: small[0] + 1]
    if not levels:
        result = np.searchsorted(np.asarray(array[...]), values, side).astype(np.int64)
        return int(result) if scalar else result
    n = int(array.shape[0])
    flat = values.ravel()
    # Work on the values in sorted order, so that the values of each block are
    # one run; sorting is skipped when they are sorted already
    order = None
    if flat.size > 1 and not (flat[1:] >= flat[:-1]).all():
        order = np.argsort(flat, kind="stable")
        flat = flat[order]
    # Every block before b is entirely before the value's position, which is in block b
    (block,), coarsest = levels[-1]
    maxima = np.asarray(coarsest[:, 1])
    blocks = np.searchsorted(maxima, flat, side)
    n_blocks = len(maxima)
    for (fine_block,), fine in [*reversed(levels[:-1]), ([1], None)]:
        if fine is None:
            n_fine = n
            read = lambda a, z: np.asarray(array[a:z])  # noqa: E731
        else:
            n_fine = int(fine.shape[0])
            read = lambda a, z, fine=fine: np.asarray(fine[a:z, 1])  # noqa: E731
        ratio = math.ceil(block / fine_block)
        blocks = _refine(blocks, flat, n_blocks, ratio, n_fine, read, side)
        n_blocks, block = n_fine, fine_block
    if order is not None:
        result = np.empty_like(blocks)
        result[order] = blocks
    else:
        result = blocks
    result = result.reshape(values.shape)
    return int(result) if scalar else result


def _range(array, start: int, stop: int, levels, max_bytes: int, name: str, reduce_data, reduce_level):
    """Pieces of array[start:stop] along the first dimension, each reduced, for combining.

    Uses whole blocks of the coarsest useful level inside the range, finer
    levels toward its edges, and the data only for the partial blocks at the
    two ends. reduce_data(data, reduce_axes) and reduce_level(values) turn a
    piece into its contribution. All pieces are read in parallel.
    """
    shape = [int(x) for x in array.shape]
    n = shape[0]
    start, stop = int(start), int(stop)
    if not 0 <= start < stop <= n:
        raise ValueError(f"Need 0 <= start < stop <= {n}, not {start}, {stop}")
    if levels:
        # The levels that share the finest level's blocks along the other dimensions
        others = levels[0][0][1:]
        levels = _nested([lv for lv in levels if lv[0][1:] == others], shape)
    else:
        others = [1] * (len(shape) - 1)
    for size, block in zip(shape[1:], others):
        if block != 1 and block < size:
            raise ValueError(f"{name} needs blocks of 1 or the whole size along the other dimensions")
    reduce_axes = tuple(d for d, (size, block) in enumerate(zip(shape[1:], others)) if block != 1 and size > 1)
    # The finest level whose blocks inside the range fit in max_bytes, and finer ones for the edges
    for k, level in enumerate(levels):
        if _level_bytes(level, math.ceil((stop - start) / level[0][0])) <= max_bytes:
            levels = levels[: k + 1]
            break

    # Which pieces to read depends only on start and stop, so plan them all, then read them together
    plan: list[tuple[int, int, int]] = []  # (level, or -1 for the data; first; last)

    def cover(lo: int, hi: int, k: int) -> None:
        if lo >= hi:
            return
        if k < 0:
            plan.append((-1, lo, hi))
            return
        block = levels[k][0][0]
        first = math.ceil(lo / block)
        last = math.ceil(n / block) if hi >= n else hi // block
        if first >= last:
            cover(lo, hi, k - 1)
            return
        plan.append((k, first, last))
        cover(lo, first * block, k - 1)
        cover(min(last * block, n), hi, k - 1)

    cover(start, stop, len(levels) - 1)
    reads = [(array if k < 0 else levels[k][1].open(), (slice(a, z),)) for k, a, z in plan]
    values = _read_all(reads)
    return [reduce_data(v, reduce_axes) if k < 0 else reduce_level(v) for (k, _, _), v in zip(plan, values)]


def minmax_range(
    array: zarr.Array, start: int, stop: int, *, max_bytes: int = DEFAULT_MAX_BYTES
) -> tuple[Any, Any]:
    """The exact minimum and maximum of array[start:stop] along the first dimension.

    Uses the array's minmax levels: whole blocks of the coarsest level that
    fit inside the range, then finer levels toward its edges, and the data
    itself only for the parts of the finest blocks at the two ends. Along the
    other dimensions the result has one value per block, so a block size of
    1 there gives a value per element (per channel), and the whole size one
    value. Those block sizes must be 1 or the whole dimension. NaN is ignored.
    It starts at the finest level whose blocks in the range take at most
    max_bytes, so coarser levels are read only when they save reading more
    than that. Without statistics the range is read whole.
    """

    def reduce_data(data: np.ndarray, reduce_axes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            low = np.nanmin(data, axis=0) if data.dtype.kind == "f" else data.min(axis=0)
            high = np.nanmax(data, axis=0) if data.dtype.kind == "f" else data.max(axis=0)
            if reduce_axes:
                low = (np.nanmin if data.dtype.kind == "f" else np.min)(low, axis=reduce_axes, keepdims=True)
                high = (np.nanmax if data.dtype.kind == "f" else np.max)(high, axis=reduce_axes, keepdims=True)
        return low, high

    def reduce_level(values: np.ndarray):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if values.dtype.kind == "f":
                return np.nanmin(values[..., 0], axis=0), np.nanmax(values[..., 1], axis=0)
            return values[..., 0].min(axis=0), values[..., 1].max(axis=0)

    parts = _range(array, start, stop, _minmax_levels(array), max_bytes, "minmax_range", reduce_data, reduce_level)
    low, high = parts[0]
    for part_low, part_high in parts[1:]:
        low, high = np.fmin(low, part_low), np.fmax(high, part_high)
    if array.ndim == 1:
        return low[()], high[()]
    return low, high


def moments_range(
    array: zarr.Array,
    start: int,
    stop: int,
    *,
    ddof: int = 0,
    missing: Sequence[Any] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    """The count of valid values, mean and variance of array[start:stop] along the first dimension.

    Like minmax_range, from the array's moments levels and the data only at
    the two ends of the range; per channel with blocks of [n, 1]. Missing
    values are left out: those the statistics were computed with, or missing
    (see resolve_missing) to pick statistics computed with those. Returns
    {"count", "mean", "variance"}, the variance with ddof delta degrees of
    freedom. Integer data of up to 16 bits gives the exact result of its
    integer sums.
    """
    wanted = None if missing is None else _missing_json(resolve_missing(array.dtype, missing))
    levels = _levels(array, "moments", wanted)
    if levels:
        attribute = array.attrs.get(STATISTICS_ATTRIBUTE) or {}
        entry = next(e for e in attribute["statistics"] if e["type"] == "moments" and e["path"] == levels[0][1]._path)
        used = resolve_missing(array.dtype, entry.get("missing"))
    else:
        used = resolve_missing(array.dtype, missing)
    exact = moments_dtype(array.dtype).kind == "i"

    def reduce_data(data: np.ndarray, reduce_axes):
        keep = _valid(data, used)
        axes = (0,) + tuple(a + 1 for a in reduce_axes)
        count = keep.sum(axis=axes, keepdims=True)
        if exact:
            values = np.where(keep, data, 0).astype(np.int64)
            result = np.stack([count, values.sum(axis=axes, keepdims=True),
                               (values * values).sum(axis=axes, keepdims=True)], axis=-1)
        else:
            values = np.where(keep, data, 0).astype(np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = values.sum(axis=axes, keepdims=True) / count
                m2 = np.where(keep, values - mean, 0.0)
                m2 = (m2 * m2).sum(axis=axes, keepdims=True)
            result = np.stack([count.astype(np.float64), np.where(count > 0, mean, np.nan),
                               np.where(count > 0, m2, np.nan)], axis=-1)
        return result[0]

    def reduce_level(values: np.ndarray):
        return _merge_moments(values, (0,))

    parts = _range(array, start, stop, levels, max_bytes, "moments_range", reduce_data, reduce_level)
    merged = _merge_moments(np.stack(parts), (0,))
    if exact:
        flat = merged.reshape(-1, 3)
        count = np.array([int(c) for c in flat[:, 0]], dtype=np.int64)
        mean = np.empty(len(flat))
        variance = np.empty(len(flat))
        for i, (c, total, squares) in enumerate(flat.tolist()):
            mean[i] = total / c if c else np.nan
            variance[i] = (squares * c - total * total) / (c * (c - ddof)) if c > ddof else np.nan
        shape = merged.shape[:-1]
        count, mean, variance = count.reshape(shape), mean.reshape(shape), variance.reshape(shape)
    else:
        count = merged[..., 0].astype(np.int64)
        mean = merged[..., 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            variance = np.where(count > ddof, merged[..., 2] / (count - ddof), np.nan)
    if array.ndim == 1:
        return {"count": int(count[()]), "mean": float(mean[()]), "variance": float(variance[()])}
    return {"count": count, "mean": mean, "variance": variance}


def find_range(
    array: zarr.Array,
    low: Any,
    high: Any,
) -> slice:
    """The slice of a sorted one-dimensional array holding the values from low to high.

    Equal to slice(searchsorted(a, low, "left"), searchsorted(a, high,
    "right")): with the array's minmax statistics it reads only those and
    the one or two blocks where the range starts and ends.
    """
    start = searchsorted(array, low, "left")
    stop = searchsorted(array, high, "right")
    return slice(start, max(start, stop))
