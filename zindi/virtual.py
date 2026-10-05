"""Arrays stored in other files, described without reading them.

A VirtualArray knows the shape, data type, and chunking of an array and where
its chunks are in other files. It holds no data. It can be sliced and stacked
like an array, and the result is another VirtualArray that points at the same
bytes:

    raw = VirtualArray.contiguous("run_g0_t0.imec0.ap.bin", shape=[n_samples, 385], dtype="int16")
    neural = raw[:, :384]       # drop the sync channel
    sync = raw[:, 384]
    first_minute = neural[: 60 * 30_000]

    channels = [VirtualArray.contiguous(path, offset=16384, shape=[n], dtype="int16") for path in files]
    signal = stack(channels, axis=1)        # one file per channel -> (time, channel)

    movie = VirtualArray.from_rfs(generate_rfs_tiff("movie.tif"), "0")     # (frame, row, column)
    movie = movie.transpose(0, 2, 1)                                       # (frame, column, row)

add_to writes one into an RfsBuilder, and zindi.nwb puts them in an NWB file.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .builder import (
    RfsBuilder,
    bytes_codecs,
    chunk_key,
    columns_selection,
    contiguous_chunk_shape,
    zarr_data_type,
)

# Maps a chunk's coordinates in an array to its coordinates in the array being
# written, which differ when the array is part of a stack. Coordinates may be
# None for the axis a series of chunks runs along.
_Place = Callable[[list], list]


class VirtualArray:
    """An array whose chunks are byte ranges in other files.

    Make one with VirtualArray.contiguous, for an uncompressed block of a
    file, or VirtualArray.from_rfs, for an array of a reference file system
    that a generator built. Combine several with stack.
    """

    shape: tuple[int, ...]
    data_type: Any
    chunk_shape: tuple[int, ...]
    codecs: list[dict]
    fill_value: Any = 0
    #: What the source says about the array, such as a sampling rate. Not written by add_to unless passed to it.
    attributes: dict
    dimension_names: Sequence[str | None] | None = None

    # -- Construction --

    @staticmethod
    def contiguous(
        url: str,
        *,
        shape: Sequence[int],
        dtype: Any,
        offset: int = 0,
        file_size: int | None = None,
        chunk_bytes: int | None = 4 * 2**20,
        attributes: dict | None = None,
    ) -> VirtualArray:
        """An uncompressed C-ordered array stored in one piece.

        Parameters
        ----------
        url : str
            The file's URL or local path.
        shape : sequence of int
            The array's shape in the file, with every column the file holds.
        dtype : numpy dtype
            The values' type, with their byte order.
        offset : int
            Where the array starts in the file.
        file_size : int or None
            The file's size. With it, a short last chunk is read at full
            length when the file extends that far.
        chunk_bytes : int or None
            Approximate size of a chunk along the first axis. None makes the
            whole array one chunk.
        """
        return _Contiguous(url, int(offset), shape, np.dtype(dtype), file_size, chunk_bytes, attributes)

    @staticmethod
    def from_rfs(rfs: dict, path: str) -> VirtualArray:
        """The array at path in a reference file system, such as one generate_rfs returned."""
        return _Referenced(rfs, path)

    # -- Description --

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        """The numpy dtype of the values, in native byte order."""
        if not isinstance(self.data_type, str):
            raise TypeError(f"Data type {self.data_type!r} has no simple numpy dtype")
        return np.dtype(self.data_type)

    def layout(self) -> dict:
        """The parts of the array's zarr.json that describe how its chunks are stored."""
        return {
            "shape": list(self.shape),
            "data_type": self.data_type,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(self.chunk_shape)}},
            "codecs": self.codecs,
            "fill_value": self.fill_value,
        }

    def __repr__(self) -> str:
        return f"<VirtualArray shape={self.shape} data_type={self.data_type!r} chunk_shape={self.chunk_shape}>"

    # -- Use --

    def __getitem__(self, key: Any) -> VirtualArray:
        raise NotImplementedError(
            "Only arrays made with VirtualArray.contiguous can be sliced: slicing a chunked, "
            "compressed, stacked, or transposed array would need its data"
        )

    def transpose(self, *axes: int) -> VirtualArray:
        """Put the axes in another order, like numpy.transpose.

        The chunks stay as the file has them. Where they hold more than one
        element along two or more axes whose order changes, the array gets
        the Zarr transpose codec, which tells a reader the order the values
        are stored in. Otherwise the bytes are the same in either order and
        only the chunk coordinates change, as for a file that stores one
        channel after another read as time by channel with one-column chunks.

        zarr-python reads a whole chunk at a time from an array with the
        transpose codec, so this suits arrays with small chunks, such as one
        frame of a movie.
        """
        if len(axes) == 1 and not isinstance(axes[0], int):
            axes = tuple(axes[0])
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        axes = tuple(int(a) % self.ndim if -self.ndim <= int(a) < self.ndim else int(a) for a in axes)
        if sorted(axes) != list(range(self.ndim)):
            raise ValueError(f"axes {axes} is not an ordering of the {self.ndim} axes")
        array = self
        if isinstance(self, _Transposed):  # reorder the array underneath once
            array, axes = self._array, tuple(self._axes[a] for a in axes)
        if axes == tuple(range(self.ndim)):
            return array
        return _Transposed(array, axes)

    def add_to(
        self,
        builder: RfsBuilder,
        path: str,
        *,
        attributes: dict | None = None,
        dimension_names: Sequence[str | None] | None = None,
        metadata: bool = True,
    ) -> None:
        """Write the array into a builder at path.

        With metadata False only the chunk locations and selection are
        written, for an array whose zarr.json is already in the builder.
        """
        if metadata:
            builder.add_array(
                path,
                shape=self.shape,
                data_type=self.data_type,
                chunk_shape=self.chunk_shape,
                codecs=self.codecs,
                fill_value=self.fill_value,
                attributes=attributes,
                dimension_names=dimension_names,
            )
        selection = self._selection()
        if selection is not None:
            builder.add_selection(path, **selection)
        self._add_chunks(builder, path, lambda coords: coords)

    def placeholder(self) -> Any:
        """What to give pynwb as a dataset's data; see zindi.nwb."""
        from .nwb import placeholder

        return placeholder(self)

    # -- For subclasses --

    def _selection(self) -> dict | None:
        return None

    def _add_chunks(self, builder: RfsBuilder, path: str, place: _Place) -> None:
        raise NotImplementedError


class _Contiguous(VirtualArray):
    """An uncompressed block of one file."""

    def __init__(
        self,
        url: str,
        offset: int,
        shape: Sequence[int],
        dtype: np.dtype,
        file_size: int | None,
        chunk_bytes: int | None,
        attributes: dict | None,
        columns: np.ndarray | None = None,
        row_shape: tuple[int, ...] | None = None,
    ) -> None:
        shape = tuple(int(n) for n in shape)
        if not shape:
            raise ValueError("A contiguous array needs at least one dimension")
        self._url, self._offset, self._dtype = url, offset, dtype
        self._file_size, self._chunk_bytes = file_size, chunk_bytes
        # The shape of one row in the file, and which of its values, in what
        # arrangement, a row of this array holds (None: all of them, as stored)
        self._row_shape = tuple(shape[1:]) if row_shape is None else row_shape
        self._columns = columns
        self.shape = shape
        self.data_type = zarr_data_type(dtype)
        self.codecs = bytes_codecs(dtype)
        self.chunk_shape = tuple(max(c, 1) for c in contiguous_chunk_shape(shape, dtype.itemsize, chunk_bytes))
        self.attributes = dict(attributes or {})

    @property
    def _row_bytes(self) -> int:
        return int(np.prod(self._row_shape)) * self._dtype.itemsize

    def __getitem__(self, key: Any) -> VirtualArray:
        key = key if isinstance(key, tuple) else (key,)
        if any(k is Ellipsis or k is None for k in key):
            raise IndexError("Index a VirtualArray with slices, integers, and lists of integers")
        if len(key) > self.ndim:
            raise IndexError(f"Too many indices for an array with {self.ndim} dimensions")
        rows = key[0] if key else slice(None)
        if not isinstance(rows, slice):
            raise IndexError(
                "The first axis of a contiguous array takes a slice: its rows are what the file stores in order"
            )
        start, stop, step = rows.indices(self.shape[0])
        if step != 1:
            raise IndexError("The first axis of a contiguous array cannot be sliced with a step")
        n_rows = max(0, stop - start)

        columns = self._columns
        if len(key) > 1:
            if columns is None:
                columns = np.arange(int(np.prod(self._row_shape))).reshape(self._row_shape)
            columns = columns[tuple(key[1:])]
            if columns.size == 0:
                raise IndexError("The selection is empty")
            if columns.shape == self._row_shape and np.array_equal(columns.ravel(), np.arange(columns.size)):
                columns = None  # every value of the row, as stored
        return _Contiguous(
            self._url,
            self._offset + start * self._row_bytes,
            (n_rows, *(self._row_shape if columns is None else columns.shape)),
            self._dtype,
            self._file_size,
            self._chunk_bytes,
            self.attributes,
            columns=columns,
            row_shape=self._row_shape,
        )

    def _selection(self) -> dict | None:
        if self._columns is None:
            return None
        record_size, keep = columns_selection(
            int(np.prod(self._row_shape)), self._dtype.itemsize, self._columns.ravel().tolist()
        )
        return {"record_size": record_size, "keep": keep}

    def _add_chunks(self, builder: RfsBuilder, path: str, place: _Place) -> None:
        if self.shape[0] == 0:
            return
        builder.add_contiguous_chunks(
            path,
            url=self._url,
            start=self._offset,
            shape=self.shape,
            chunk_shape=self.chunk_shape,
            itemsize=self._dtype.itemsize,
            file_size=self._file_size,
            row_bytes=self._row_bytes,
            coords=place([None] + [0] * (self.ndim - 1)),
        )


class _Referenced(VirtualArray):
    """An array of a reference file system, with the chunks its generator found."""

    def __init__(self, rfs: dict, path: str) -> None:
        refs = rfs["refs"]
        meta_key = f"{path}/zarr.json" if path else "zarr.json"
        if meta_key not in refs:
            raise KeyError(f"No array at {path!r}")
        meta = refs[meta_key]
        meta = json.loads(meta) if isinstance(meta, str) else meta
        if meta.get("node_type") != "array":
            raise ValueError(f"{path!r} is a group, not an array")
        if meta["chunk_grid"]["name"] != "regular":
            raise NotImplementedError(f"Unsupported chunk grid {meta['chunk_grid']['name']!r}")
        self.shape = tuple(meta["shape"])
        self.data_type = meta["data_type"]
        self.chunk_shape = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
        self.codecs = meta["codecs"]
        self.fill_value = meta.get("fill_value", 0)
        self.attributes = dict(meta.get("attributes", {}))
        self.dimension_names = meta.get("dimension_names")

        templates = rfs.get("templates", {})

        def expand(url: str) -> str:
            for name, value in templates.items():
                url = url.replace("{{" + name + "}}", value)
            return url

        prefix = f"{path}/c" if path else "c"
        self._refs: dict[tuple[int, ...], Any] = {}
        for key, value in refs.items():
            if key == prefix or key.startswith(prefix + "/"):
                parts = key[len(prefix) + 1 :].split("/") if len(key) > len(prefix) else []
                if not all(part.isdigit() for part in parts):
                    continue
                if isinstance(value, list):
                    value = [expand(value[0]), *value[1:]]
                self._refs[tuple(int(part) for part in parts)] = value
        self._gens = [
            {**entry, "url": expand(entry["url"]), "key": entry["key"][len(prefix) + 1 :].split("/")}
            for entry in rfs.get("gen", [])
            if entry["key"].startswith(prefix + "/")
        ]
        self._index = rfs.get("indexes", {}).get(path)
        self._chosen = rfs.get("selections", {}).get(path)

    def _selection(self) -> dict | None:
        return self._chosen

    def _add_chunks(self, builder: RfsBuilder, path: str, place: _Place) -> None:
        for coords, value in self._refs.items():
            builder.refs[chunk_key(path, place(list(coords)))] = value
        for entry in self._gens:
            # Each part of a gen key is a number or a placeholder, and place only moves them
            builder.gen.append({**entry, "key": f"{path}/c/" + "/".join(str(part) for part in place(entry["key"]))})
        if self._index is not None:
            if place(list(range(self.ndim))) != list(range(self.ndim)):
                raise NotImplementedError("An array with a chunk index cannot be stacked yet")
            builder.indexes[path] = self._index


class _Stacked(VirtualArray):
    """Arrays joined along a new axis."""

    def __init__(self, arrays: Sequence[VirtualArray], axis: int) -> None:
        first = arrays[0]
        self._arrays, self._axis = list(arrays), axis
        self.shape = (*first.shape[:axis], len(arrays), *first.shape[axis:])
        self.chunk_shape = (*first.chunk_shape[:axis], 1, *first.chunk_shape[axis:])
        self.data_type, self.codecs, self.fill_value = first.data_type, first.codecs, first.fill_value
        self.attributes = {}

    def _selection(self) -> dict | None:
        return self._arrays[0]._selection()

    def _add_chunks(self, builder: RfsBuilder, path: str, place: _Place) -> None:
        axis = self._axis
        for j, array in enumerate(self._arrays):
            array._add_chunks(builder, path, lambda coords, j=j: place([*coords[:axis], j, *coords[axis:]]))


class _Transposed(VirtualArray):
    """An array with its axes in another order."""

    def __init__(self, array: VirtualArray, axes: tuple[int, ...]) -> None:
        self._array, self._axes = array, axes
        self.shape = tuple(array.shape[a] for a in axes)
        self.chunk_shape = tuple(array.chunk_shape[a] for a in axes)
        self.data_type, self.fill_value = array.data_type, array.fill_value
        self.attributes = dict(array.attributes)
        # A chunk's bytes depend only on the order of the axes it extends along
        extended = [a for a in axes if array.chunk_shape[a] > 1]
        if extended == sorted(extended):
            self.codecs = array.codecs
        else:
            # The codec's order takes this array's axes to the stored ones
            stored = [axes.index(a) for a in range(len(axes))]
            self.codecs = [{"name": "transpose", "configuration": {"order": stored}}, *array.codecs]

    def _selection(self) -> dict | None:
        return self._array._selection()

    def _add_chunks(self, builder: RfsBuilder, path: str, place: _Place) -> None:
        axes = self._axes
        self._array._add_chunks(builder, path, lambda coords: place([coords[a] for a in axes]))


def stack(arrays: Sequence[VirtualArray], axis: int = 0) -> VirtualArray:
    """Join arrays along a new axis, like numpy.stack.

    The arrays must have the same shape, data type, chunking, codecs, and
    selection. Each keeps its chunks, which get one more coordinate, so files
    that each hold one channel, frame, or plane become one array.
    """
    arrays = list(arrays)
    if not arrays:
        raise ValueError("Need at least one array to stack")
    first = arrays[0]
    if not -first.ndim - 1 <= axis <= first.ndim:
        raise ValueError(f"axis {axis} is out of range for stacking arrays with {first.ndim} dimensions")
    axis = axis % (first.ndim + 1)
    for i, array in enumerate(arrays[1:], start=1):
        for name in ("shape", "data_type", "chunk_shape", "codecs", "fill_value"):
            if getattr(array, name) != getattr(first, name):
                raise ValueError(
                    f"Cannot stack: array {i} has {name} {getattr(array, name)!r}, "
                    f"and array 0 has {getattr(first, name)!r}"
                )
        if array._selection() != first._selection():
            raise ValueError(f"Cannot stack: array {i} and array 0 select different bytes of each record")
    return _Stacked(arrays, axis)
