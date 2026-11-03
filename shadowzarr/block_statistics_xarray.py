"""Xarray objects that use block statistics without changing how they're used.

open_timeseries opens an NWB TimeSeries as a DataArray. Its data is a
StatsArray, a lazy array that Xarray wraps like any other: selecting along
time reads nothing, and the reductions Xarray calls on it, min, max, mean,
var and std over time, are answered from the array's minmax and moments
statistics (see shadowzarr.block_statistics), reading the data only at the two
ends of the range. Anything else reads the data as usual. The time
coordinate has a BlockStatisticsIndex, so selecting by time bisects the
timestamps' statistics instead of loading the timestamps.

    da = open_timeseries(root["acquisition/lfp"])
    window = da.sel(time=slice(1200.0, 3000.0))   # reads the timestamps' statistics and two blocks
    window.max("time"), window.std("time")        # per channel, from the data's statistics
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import math

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from numpy.lib.mixins import NDArrayOperatorsMixin
from xarray.core.indexing import IndexSelResult
from xarray.indexes import RangeIndex

from .block_statistics import (
    STATISTICS_ATTRIBUTE,
    minmax_range,
    moments_range,
    searchsorted,
)

_MINMAX = {np.min: "min", np.amin: "min", np.max: "max", np.amax: "max",
           np.nanmin: "min", np.nanmax: "max"}
_MOMENTS = {np.mean: "mean", np.nanmean: "mean", np.var: "var", np.nanvar: "var",
            np.std: "std", np.nanstd: "std"}
_SKIPS_NAN = {np.nanmin, np.nanmax, np.nanmean, np.nanvar, np.nanstd}


def _materialize(value: Any) -> Any:
    if isinstance(value, StatsArray):
        return np.asarray(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_materialize(v) for v in value)
    return value


class _ReadOnce:
    """The array, with each selection read at most once, for answering several reductions."""

    def __init__(self, array: zarr.Array):
        self._array = array
        self._read: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._array, name)

    def __getitem__(self, key: Any) -> Any:
        token = repr(key)
        if token not in self._read:
            self._read[token] = np.asarray(self._array[key])
        return self._read[token]


class StatsArray(NDArrayOperatorsMixin):
    """A lazy view of a range of a Zarr array along its first dimension.

    Slicing along the first dimension gives another view without reading.
    numpy's min, max, mean, var and std (and their nan forms) over the first
    dimension use the array's block statistics; every other operation reads
    the view, so results are always what numpy would give.
    """

    def __init__(self, array: zarr.Array, start: int = 0, stop: int | None = None):
        if array.ndim < 1:
            raise ValueError("StatsArray needs an array of at least one dimension")
        self.base = array
        n = int(array.shape[0])
        self.start = int(start)
        self.stop = n if stop is None else int(stop)
        self.dtype = np.dtype(array.dtype)

    # -- Array attributes --

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.stop - self.start, *[int(s) for s in self.base.shape[1:]])

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    def __len__(self) -> int:
        return self.shape[0]

    def __copy__(self) -> StatsArray:
        return self

    def __deepcopy__(self, memo: dict) -> StatsArray:
        # A read-only view: copies can share it, and the store behind it can't be copied
        return self

    def __repr__(self) -> str:
        return f"StatsArray({self.base.path or 'array'}[{self.start}:{self.stop}], shape={self.shape}, dtype={self.dtype})"

    # -- Indexing --

    def __getitem__(self, key: Any) -> Any:
        key = key if isinstance(key, tuple) else (key,)
        if any(k is Ellipsis for k in key):
            i = next(i for i, k in enumerate(key) if k is Ellipsis)
            key = key[:i] + (slice(None),) * (self.ndim - len(key) + 1) + key[i + 1:]
        key = key + (slice(None),) * (self.ndim - len(key))
        first, rest = key[0], key[1:]
        whole_rest = all(isinstance(k, slice) and k == slice(None) for k in rest)
        if isinstance(first, slice) and first.step in (None, 1) and whole_rest:
            a, z, _ = first.indices(len(self))
            return StatsArray(self.base, self.start + a, self.start + max(a, z))
        # Anything else reads only what it selects
        if isinstance(first, slice):
            a, z, step = first.indices(len(self))
            first = slice(self.start + a, self.start + z, step) if step > 0 else slice(
                self.start + a, None if z < 0 else self.start + z, step)
        elif isinstance(first, (int, np.integer)):
            first = self.start + (int(first) % len(self))
        else:
            first = np.asarray(first)
            if first.dtype == bool:
                first = np.flatnonzero(first)
            first = self.start + np.where(first < 0, first + len(self), first)
        full = (first, *rest)
        if all(isinstance(k, (slice, int, np.integer)) for k in full):
            return np.asarray(self.base[full])
        return np.asarray(self.base.oindex[full])

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        data = np.asarray(self.base[self.start:self.stop])
        return data if dtype is None else data.astype(dtype)

    def astype(self, dtype: Any, copy: bool = True) -> Any:
        if np.dtype(dtype) == self.dtype and not copy:
            return self
        return np.asarray(self).astype(dtype)

    # -- numpy --

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return getattr(ufunc, method)(*_materialize(inputs), **_materialize(kwargs))

    def __array_function__(self, func, types, args, kwargs):
        if (func in _MINMAX or func in _MOMENTS) and args and args[0] is self:
            result = self._reduce(func, args[1:], kwargs)
            if result is not NotImplemented:
                return result
        return func(*_materialize(args), **_materialize(kwargs))

    def _reduce(self, func, args, kwargs):
        kwargs = dict(kwargs)
        axis = args[0] if args else kwargs.pop("axis", None)
        keepdims = bool(kwargs.pop("keepdims", False))
        ddof = int(kwargs.pop("ddof", 0))
        for name in ("dtype", "out", "where", "initial"):
            if kwargs.pop(name, None) is not None:
                return NotImplemented
        if kwargs or len(args) > 1 or len(self) == 0:
            return NotImplemented
        if axis is None and self.ndim == 1:
            axis = 0
        if isinstance(axis, tuple):
            if len(axis) != 1:
                return NotImplemented
            axis = axis[0]
        if axis not in (0, -self.ndim):
            return NotImplemented
        try:
            if func in _MINMAX:
                result = self._minmax(func)
            else:
                result = self._moments(func, ddof)
        except ValueError:
            # Statistics whose blocks don't suit a range of the first dimension
            return NotImplemented
        if result is NotImplemented:
            return result
        if keepdims:
            result = np.expand_dims(result, 0)
        return result

    def _minmax(self, func):
        skips_nan = func in _SKIPS_NAN or self.dtype.kind != "f"
        if not self._has("minmax") or (not skips_nan and not self._has("moments")):
            return NotImplemented
        # Both reductions below read the same edges; read them once
        base = self.base if skips_nan else _ReadOnce(self.base)
        low, high = minmax_range(base, self.start, self.stop)
        result = np.asarray(low if _MINMAX[func] == "min" else high)
        if not skips_nan:
            # numpy gives NaN where a value is NaN; the counts tell where
            count = moments_range(base, self.start, self.stop)["count"]
            result = np.where(np.asarray(count) < len(self), np.nan, result).astype(self.dtype)
        return result[()] if result.ndim == 0 else result

    def _moments(self, func, ddof):
        if not self._has("moments"):
            return NotImplemented
        stats = moments_range(self.base, self.start, self.stop, ddof=ddof)
        count = np.asarray(stats["count"])
        name = _MOMENTS[func]
        value = np.asarray(stats["mean"] if name == "mean" else stats["variance"], dtype=np.float64)
        if name == "std":
            value = np.sqrt(value)
        if func not in _SKIPS_NAN and self.dtype.kind == "f":
            value = np.where(count < len(self), np.nan, value)
        out_dtype = self.dtype if self.dtype.kind == "f" else np.dtype("float64")
        value = value.astype(out_dtype)
        return value[()] if value.ndim == 0 else value

    def _has(self, kind: str) -> bool:
        """Whether the array has statistics of kind that keep every element of the other dimensions.

        numpy reduces over the first dimension only, so statistics that also
        combine values across other dimensions can't answer it.
        """
        entries = (self.base.attrs.get(STATISTICS_ATTRIBUTE) or {}).get("statistics", [])
        others = [int(s) for s in self.base.shape[1:]]
        return any(
            e["type"] == kind and all(b == 1 or s <= 1 for b, s in zip(e["block_shape"][1:], others))
            for e in entries
        )


class BlockStatisticsIndex(xr.Index):
    """An Xarray index for a sorted coordinate held in a StatsArray.

    Selection bisects the coordinate's minmax statistics, reading only those
    and the blocks the labels fall in. Supports slices, single labels and
    arrays of labels, exactly or with method="nearest".
    """

    def __init__(self, array: StatsArray, dim: Hashable, name: Hashable):
        self.array, self.dim, self.name = array, dim, name

    @classmethod
    def from_variables(cls, variables: Mapping[Any, xr.Variable], *, options: Mapping[str, Any]):
        if len(variables) != 1:
            raise ValueError("BlockStatisticsIndex indexes one coordinate")
        (name, variable), = variables.items()
        if variable.ndim != 1:
            raise ValueError("BlockStatisticsIndex needs a one-dimensional coordinate")
        data = variable._data
        if not isinstance(data, StatsArray):
            raise TypeError("BlockStatisticsIndex needs a coordinate whose data is a StatsArray")
        return cls(data, variable.dims[0], name)

    def create_variables(self, variables: Mapping[Any, xr.Variable] | None = None) -> dict[Hashable, xr.Variable]:
        attrs = variables[self.name].attrs if variables and self.name in variables else None
        return {self.name: xr.Variable((self.dim,), self.array, attrs=attrs)}

    def _positions(self, values: Any, side: str) -> np.ndarray:
        """Positions in the view, from positions in the whole sorted array."""
        found = np.asarray(searchsorted(self.array.base, values, side))
        return np.clip(found, self.array.start, self.array.stop) - self.array.start

    def sel(self, labels: dict[Any, Any], method: str | None = None, tolerance: Any = None) -> IndexSelResult:
        label = labels[self.name]
        if isinstance(label, slice):
            if label.step is not None:
                raise ValueError("BlockStatisticsIndex selects slices without a step")
            start = 0 if label.start is None else int(self._positions(label.start, "left"))
            stop = len(self.array) if label.stop is None else int(self._positions(label.stop, "right"))
            return IndexSelResult({self.dim: slice(start, max(start, stop))})
        values = np.asarray(label.values if isinstance(label, (xr.DataArray, xr.Variable)) else label)
        positions = self._positions(values, "left")
        if method is None:
            inside = positions < len(self.array)
            found = np.zeros(values.shape, dtype=bool)
            if inside.any():
                found[inside] = self.array[positions[inside]] == values[inside]
            if not found.all():
                raise KeyError(f"not all values found in index {self.name!r}")
        elif method == "nearest":
            right = np.minimum(positions, len(self.array) - 1)
            left = np.maximum(positions - 1, 0)
            right_value = self.array[right.ravel()].reshape(right.shape)
            left_value = self.array[left.ravel()].reshape(left.shape)
            positions = np.where(np.abs(values - left_value) <= np.abs(right_value - values), left, right)
            if tolerance is not None:
                chosen = np.where(positions == left, left_value, right_value)
                if (np.abs(chosen - values) > tolerance).any():
                    raise KeyError(f"not all values found in index {self.name!r} within tolerance {tolerance}")
        else:
            raise ValueError(f"BlockStatisticsIndex supports method=None or 'nearest', not {method!r}")
        if positions.ndim == 0:
            return IndexSelResult({self.dim: int(positions)})
        if isinstance(label, (xr.DataArray, xr.Variable)):
            positions = xr.Variable(label.dims, positions)
        return IndexSelResult({self.dim: positions})

    def isel(self, indexers: Mapping[Any, Any]) -> BlockStatisticsIndex | None:
        indexer = indexers[self.dim]
        if isinstance(indexer, slice) and indexer.step in (None, 1):
            return type(self)(self.array[indexer], self.dim, self.name)
        return None

    def equals(self, other: Any, *, exclude: Any = None) -> bool:
        return (isinstance(other, BlockStatisticsIndex) and other.array.base is self.array.base
                and (other.array.start, other.array.stop) == (self.array.start, self.array.stop))

    def to_pandas_index(self) -> pd.Index:
        return pd.Index(np.asarray(self.array), name=self.dim)

    def _repr_inline_(self, max_width: int) -> str:
        return f"BlockStatisticsIndex ({len(self.array):,} values)"


class RegularTimeIndex(RangeIndex):
    """An Xarray RangeIndex whose slices select the samples inside them, like BlockStatisticsIndex.

    RangeIndex selects slices only with method="nearest", rounding their
    ends; here a plain slice from a to b gives the samples whose times t
    have a <= t <= b.
    """

    def sel(self, labels, method=None, tolerance=None):
        label = labels[self.dim]
        if method is None and isinstance(label, slice) and label.step is None:
            step, first = float(self.step), float(self.start)
            eps = 1e-9 * abs(step)
            lo = 0 if label.start is None else math.ceil((label.start - first) / step - eps)
            hi = self.size if label.stop is None else math.floor((label.stop - first) / step + eps) + 1
            lo, hi = min(max(lo, 0), self.size), min(max(hi, 0), self.size)
            return IndexSelResult({self.dim: slice(lo, max(lo, hi))})
        return super().sel(labels, method="nearest" if method is None else method, tolerance=tolerance)

    def isel(self, indexers):
        result = super().isel(indexers)
        if isinstance(result, RangeIndex) and not isinstance(result, RegularTimeIndex):
            result.__class__ = RegularTimeIndex
        return result


def time_coordinate(timestamps: zarr.Array, dim: str = "time") -> xr.Coordinates:
    """Coordinates for dim from a sorted timestamps array, indexed by its block statistics."""
    variable = xr.Variable((dim,), StatsArray(timestamps))
    index = BlockStatisticsIndex.from_variables({dim: variable}, options={})
    return xr.Coordinates({dim: variable}, indexes={dim: index})


def open_timeseries(group: zarr.Group, *, dims: tuple[str, ...] | None = None, name: str | None = None) -> xr.DataArray:
    """An NWB TimeSeries group as a DataArray whose data and time use block statistics.

    The time coordinate comes from the "timestamps" array, with a
    BlockStatisticsIndex, or from "starting_time" and its "rate", with an
    Xarray RangeIndex. Other dimensions are named dim_1, dim_2 and so on
    unless dims names them all.
    """
    data = group["data"]
    dims = tuple(dims) if dims is not None else ("time", *[f"dim_{i}" for i in range(1, data.ndim)])
    if len(dims) != data.ndim:
        raise ValueError(f"dims has {len(dims)} names; the data has {data.ndim} dimensions")
    if "timestamps" in group:
        coords = time_coordinate(group["timestamps"], dims[0])
    else:
        start = float(np.asarray(group["starting_time"][...]))
        rate = float(group["starting_time"].attrs["rate"])
        n = int(data.shape[0])
        index = RegularTimeIndex.linspace(start, start + (n - 1) / rate, n, coord_name=dims[0], dim=dims[0])
        coords = xr.Coordinates.from_xindex(index)
    attrs = {k: v for k, v in group.attrs.items() if isinstance(v, (str, int, float))}
    unit = data.attrs.get("unit")
    if unit:
        attrs["unit"] = unit
    return xr.DataArray(StatsArray(data), dims=dims, coords=coords, name=name or group.basename, attrs=attrs)
