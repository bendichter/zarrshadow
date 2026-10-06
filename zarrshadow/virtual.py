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

add_to writes one into an RfsBuilder, and zarrshadow.nwb puts them in an NWB file.
"""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .builder import (
    RfsBuilder,
    bytes_codecs,
    chunk_key,
    contiguous_chunk_shape,
    zarr_data_type,
)
from .chunk_index import ChunkIndex
from .gen import Generator

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
    def records(
        url: str,
        *,
        shape: Sequence[int],
        dtype: Any,
        record_size: int,
        skip: int = 0,
        value_offsets: Sequence[int] | None = None,
        offset: int = 0,
        file_size: int | None = None,
        chunk_bytes: int | None = 4 * 2**20,
        attributes: dict | None = None,
    ) -> VirtualArray:
        """An uncompressed array whose rows are stored in records of fixed size, with other bytes around them.

        Each row of the array, along its first axis, is in one record of
        record_size bytes: skip bytes of something else, then the row's
        values in C order, then whatever fills the rest of the record. A
        recording stored in packets, each with a header before its samples,
        is like this. When the values of a row are not one after another in
        the record, value_offsets says where each one starts. The result can
        be sliced like a contiguous array.

        Parameters
        ----------
        url : str
            The file's URL or local path.
        shape : sequence of int
            The array's shape. The first axis counts records.
        dtype : numpy dtype
            The values' type, with their byte order.
        record_size : int
            The size of a record in the file, in bytes.
        skip : int
            How many bytes of a record come before the row's values.
        value_offsets : sequence of int or None
            Where each value of a row starts within the record, in the row's
            C order, in place of skip.
        offset : int
            Where the first record starts in the file.
        file_size, chunk_bytes
            As for contiguous.
        """
        dtype = np.dtype(dtype)
        shape = tuple(int(n) for n in shape)
        row_values = int(np.prod(shape[1:]))
        if value_offsets is not None:
            offsets = np.asarray(value_offsets, dtype=np.int64)
            if offsets.size != row_values or offsets.min() < 0 or offsets.max() + dtype.itemsize > record_size:
                raise ValueError(f"value_offsets must give {row_values} places within a record of {record_size} bytes")
            offsets = offsets.reshape(shape[1:])
        else:
            if skip < 0 or skip + row_values * dtype.itemsize > record_size:
                raise ValueError(
                    f"A row of {row_values * dtype.itemsize} bytes after {skip} does not fit in a record of {record_size}"
                )
            offsets = (int(skip) + np.arange(row_values) * dtype.itemsize).reshape(shape[1:])
        in_order = np.array_equal(offsets.ravel(), np.arange(row_values) * dtype.itemsize)
        whole_record = in_order and row_values * dtype.itemsize == record_size
        return _Contiguous(
            url, int(offset), shape, dtype, file_size, chunk_bytes, attributes,
            record_bytes=int(record_size), element_offsets=None if whole_record else offsets,
        )

    @staticmethod
    def from_memmap(
        array: np.ndarray,
        *,
        url: str | None = None,
        chunk_bytes: int | None = 4 * 2**20,
        attributes: dict | None = None,
    ) -> VirtualArray:
        """The part of a file that a numpy.memmap, or a view of one, shows.

        Readers that map a file into memory, as many of NEO's do, say with
        that array where the data is: the file, the offset, the type, and
        the shape. The view may take a field of a structured memmap, which
        is how samples stored in packets are read; the result then skips
        the rest of each packet.

        Parameters
        ----------
        array : numpy.ndarray
            A numpy.memmap or a view of one. Its rows must follow one another
            at a fixed distance in the file, each in C order.
        url : str or None
            The URL the references should point to. By default the local
            path of the mapped file.
        """
        path, start, root = memmap_location(array)
        if array.ndim == 0:
            raise ValueError("The array needs at least one dimension")
        itemsize = array.dtype.itemsize
        row_bytes = int(np.prod(array.shape[1:])) * itemsize
        expected = tuple(itemsize * int(np.prod(array.shape[k + 1 :])) for k in range(1, array.ndim))
        stride = array.strides[0] if array.shape[0] > 1 else row_bytes
        if array.dtype.hasobject or array.strides[1:] != expected or stride < row_bytes:
            raise NotImplementedError("The view does not take whole rows in C order at a fixed distance in the file")
        kwargs: dict[str, Any] = {
            "shape": array.shape,
            "dtype": array.dtype,
            "file_size": os.path.getsize(path),
            "chunk_bytes": chunk_bytes,
            "attributes": attributes,
        }
        if stride == row_bytes:
            return VirtualArray.contiguous(url or path, offset=start, **kwargs)
        # rows inside larger records: a field of a structured memmap whose items are the records
        skip = (start - int(root.offset)) % stride if root.dtype.itemsize == stride else 0
        return VirtualArray.records(url or path, record_size=stride, skip=skip, offset=start - skip, **kwargs)

    @staticmethod
    def from_chunks(
        chunks: dict[tuple[int, ...], tuple[str, int, int]],
        *,
        shape: Sequence[int],
        chunk_shape: Sequence[int],
        dtype: Any,
        attributes: dict | None = None,
    ) -> VirtualArray:
        """An uncompressed array whose chunks are given one by one.

        Parameters
        ----------
        chunks : dict
            For each chunk, its coordinates in the chunk grid and where it is:
            (URL or local path, offset, length). Chunks left out read as zeros.
        shape, chunk_shape : sequence of int
            The array's shape and the shape of one chunk. Each chunk holds its
            values in C order.
        dtype : numpy dtype
            The values' type, with their byte order.

        The pages of a TIFF stack, each stored in one piece, are chunks of one
        page each: shape (pages, rows, columns) with chunk_shape (1, rows, columns).
        """
        return _Listed(chunks, shape, chunk_shape, np.dtype(dtype), attributes)

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
        """Select whole chunks.

        Along an axis whose chunks are one element long, any slice or list of
        indices selects chunks, as for the pages of a TIFF stack. Along other
        axes a slice must start and stop on chunk boundaries. Anything else
        would need the data; only arrays made with VirtualArray.contiguous
        can be sliced freely.
        """
        key = key if isinstance(key, tuple) else (key,)
        if len(key) > self.ndim:
            raise IndexError(f"Too many indices for an array with {self.ndim} dimensions")
        picks: list[np.ndarray | None] = []
        shape = []
        for axis, (length, chunk) in enumerate(zip(self.shape, self.chunk_shape)):
            k = key[axis] if axis < len(key) else slice(None)
            n_chunks = -(-length // chunk)
            if isinstance(k, slice) and k == slice(None):
                picks.append(None)
                shape.append(length)
                continue
            if isinstance(k, slice):
                start, stop, step = k.indices(length)
                indices = np.arange(start, stop, step)
            elif isinstance(k, (list, np.ndarray)) and np.asarray(k).ndim == 1 and np.asarray(k).dtype.kind in "iu":
                indices = np.asarray(k) % length if length else np.asarray(k)
            else:
                raise NotImplementedError(_NOT_SLICEABLE + "; index with slices or lists of integers")
            if chunk == 1:
                chosen = indices
                new_length = len(indices)
            else:
                # whole chunks only: a run of elements that starts and stops on chunk boundaries
                run = len(indices) and np.array_equal(indices, np.arange(indices[0], indices[-1] + 1))
                if not run or indices[0] % chunk or ((indices[-1] + 1) % chunk and indices[-1] + 1 != length):
                    raise NotImplementedError(_NOT_SLICEABLE)
                chosen = np.arange(indices[0] // chunk, -(-(indices[-1] + 1) // chunk))
                new_length = len(indices)
            if len(chosen) == 0:
                raise IndexError("The selection is empty")
            picks.append(None if np.array_equal(chosen, np.arange(n_chunks)) else chosen)
            shape.append(new_length)
        if all(p is None for p in picks):
            return self
        return _Taken(self, picks, tuple(shape))

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
        """What to give pynwb as a dataset's data; see zarrshadow.nwb."""
        from .nwb import placeholder

        return placeholder(self)

    # -- For subclasses --

    def _selection(self) -> dict | None:
        return None

    def _add_chunks(self, builder: RfsBuilder, path: str, place: _Place) -> None:
        raise NotImplementedError


def memmap_location(array: np.ndarray) -> tuple[str, int, np.memmap]:
    """The file a numpy.memmap, or a view of one, maps, where the view's first value is in it, and the memmap."""
    root = array
    while isinstance(getattr(root, "base", None), np.ndarray):
        root = root.base
    if not isinstance(root, np.memmap) or root.filename is None:
        raise TypeError("Expected a numpy.memmap or a view of one")
    delta = array.__array_interface__["data"][0] - root.__array_interface__["data"][0]
    return str(root.filename), int(root.offset) + int(delta), root


class _Contiguous(VirtualArray):
    """Rows stored one after another in one file, uncompressed, each in a record of fixed size."""

    def __init__(
        self,
        url: str,
        offset: int,
        shape: Sequence[int],
        dtype: np.dtype,
        file_size: int | None,
        chunk_bytes: int | None,
        attributes: dict | None,
        record_bytes: int | None = None,
        element_offsets: np.ndarray | None = None,
    ) -> None:
        shape = tuple(int(n) for n in shape)
        if not shape:
            raise ValueError("A contiguous array needs at least one dimension")
        self._url, self._offset, self._dtype = url, offset, dtype
        self._file_size, self._chunk_bytes = file_size, chunk_bytes
        # A row of the array is read from one record of the file. element_offsets
        # gives, for each value of a row, where it starts within the record; None
        # means the record holds the row's values and nothing else, in order.
        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        self._record_bytes = row_bytes if record_bytes is None else int(record_bytes)
        self._element_offsets = element_offsets
        self.shape = shape
        self.data_type = zarr_data_type(dtype)
        self.codecs = bytes_codecs(dtype)
        self.chunk_shape = tuple(max(c, 1) for c in contiguous_chunk_shape(shape, dtype.itemsize, chunk_bytes))
        self.attributes = dict(attributes or {})

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

        offsets = self._element_offsets
        row_shape = self.shape[1:]
        if len(key) > 1:
            if offsets is None:
                offsets = (np.arange(int(np.prod(row_shape))) * self._dtype.itemsize).reshape(row_shape)
            offsets = offsets[tuple(key[1:])]
            if offsets.size == 0:
                raise IndexError("The selection is empty")
            row_shape = offsets.shape
            whole_record = offsets.size * self._dtype.itemsize == self._record_bytes
            if whole_record and np.array_equal(offsets.ravel(), np.arange(offsets.size) * self._dtype.itemsize):
                offsets = None  # every value of the record, as stored
        return _Contiguous(
            self._url,
            self._offset + start * self._record_bytes,
            (n_rows, *row_shape),
            self._dtype,
            self._file_size,
            self._chunk_bytes,
            self.attributes,
            record_bytes=self._record_bytes,
            element_offsets=offsets,
        )

    def _selection(self) -> dict | None:
        if self._element_offsets is None:
            return None
        itemsize = self._dtype.itemsize
        keep: list[list[int]] = []
        for start in self._element_offsets.ravel().tolist():
            if keep and keep[-1][1] == start:
                keep[-1][1] = start + itemsize
            else:
                keep.append([start, start + itemsize])
        return {"record_size": self._record_bytes, "keep": keep}

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
            row_bytes=self._record_bytes,
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


_NOT_SLICEABLE = (
    "Only arrays made with VirtualArray.contiguous can be sliced inside their chunks: slicing a chunked, "
    "compressed, stacked, or transposed array there would need its data"
)


def _chunk_refs(array: VirtualArray) -> dict[tuple[int, ...], Any]:
    """Every chunk of an array by its coordinates: [url, offset, length], or the text of a chunk stored inline."""
    builder = RfsBuilder()
    array._add_chunks(builder, "a", lambda coords: coords)
    refs: dict[tuple[int, ...], Any] = {}

    def coords_of(key: str) -> tuple[int, ...]:
        return tuple(int(part) for part in key[len("a/c/") :].split("/")) if key != "a/c" else ()

    for key, value in builder.refs.items():
        refs[coords_of(key)] = value
    for entry in builder.gen:
        for key, ref in Generator(entry, {}).items():
            refs[coords_of(key)] = ref
    for entry in builder.indexes.values():
        for coords, offset, nbytes in ChunkIndex(entry["url"], entry["index"]).iter_chunks():
            refs[tuple(int(c) for c in coords)] = [entry["url"], int(offset), int(nbytes)]
    return refs


class _Taken(VirtualArray):
    """Some of the chunks of an array, in a chosen order."""

    def __init__(self, array: VirtualArray, picks: list[np.ndarray | None], shape: tuple[int, ...]) -> None:
        self._array, self._picks = array, picks
        self.shape = shape
        self.chunk_shape = array.chunk_shape
        self.data_type, self.codecs, self.fill_value = array.data_type, array.codecs, array.fill_value
        self.attributes = dict(array.attributes)
        self.dimension_names = array.dimension_names

    def _selection(self) -> dict | None:
        return self._array._selection()

    def _add_chunks(self, builder: RfsBuilder, path: str, place: _Place) -> None:
        _add_listed_chunks(builder, path, place, self._picked_chunks(), self.shape, self.chunk_shape)

    def _picked_chunks(self) -> dict[tuple[int, ...], Any]:
        # where each of the array's chunks goes, along each axis that chunks were picked on
        positions: list[dict[int, list[int]] | None] = []
        for pick in self._picks:
            if pick is None:
                positions.append(None)
                continue
            where: dict[int, list[int]] = {}
            for new, old in enumerate(pick.tolist()):
                where.setdefault(old, []).append(new)
            positions.append(where)
        chunks: dict[tuple[int, ...], Any] = {}
        for old, ref in _chunk_refs(self._array).items():
            options = [[c] if where is None else where.get(c, []) for c, where in zip(old, positions)]
            for new in itertools.product(*options):
                chunks[tuple(new)] = ref
        return chunks


def _add_listed_chunks(
    builder: RfsBuilder,
    path: str,
    place: _Place,
    chunks: dict[tuple[int, ...], Any],
    shape: Sequence[int],
    chunk_shape: Sequence[int],
) -> None:
    """Write chunks given one by one, each [url, offset, length] or the text of a chunk stored inline."""
    grid = [-(-length // chunk) for length, chunk in zip(shape, chunk_shape)]
    urls = {ref[0] for ref in chunks.values() if isinstance(ref, list)}
    in_place = place(list(range(len(grid)))) == list(range(len(grid)))
    if in_place and len(urls) == 1 and all(isinstance(ref, list) and len(ref) == 3 for ref in chunks.values()):
        # one file: let the builder store the chunks in its most compact form
        builder.add_chunks(path, grid, urls.pop(), {c: (ref[1], ref[2]) for c, ref in chunks.items()})
        return
    for coords, ref in chunks.items():
        builder.refs[chunk_key(path, place(list(coords)))] = ref


class _Listed(VirtualArray):
    """An uncompressed array whose chunks were given one by one."""

    def __init__(
        self,
        chunks: dict[tuple[int, ...], tuple[str, int, int]],
        shape: Sequence[int],
        chunk_shape: Sequence[int],
        dtype: np.dtype,
        attributes: dict | None,
    ) -> None:
        self.shape = tuple(int(n) for n in shape)
        self.chunk_shape = tuple(int(n) for n in chunk_shape)
        if len(self.shape) != len(self.chunk_shape) or any(c < 1 for c in self.chunk_shape):
            raise ValueError(f"chunk_shape {self.chunk_shape} does not fit an array of shape {self.shape}")
        grid = [-(-length // chunk) for length, chunk in zip(self.shape, self.chunk_shape)]
        self._chunks: dict[tuple[int, ...], Any] = {}
        for coords, (url, offset, length) in chunks.items():
            coords = tuple(int(c) for c in coords)
            if len(coords) != len(grid) or any(not 0 <= c < n for c, n in zip(coords, grid)):
                raise ValueError(f"Chunk {coords} is outside the chunk grid {tuple(grid)}")
            self._chunks[coords] = [url, int(offset), int(length)]
        self.data_type = zarr_data_type(dtype)
        self.codecs = bytes_codecs(dtype)
        self.attributes = dict(attributes or {})

    def _add_chunks(self, builder: RfsBuilder, path: str, place: _Place) -> None:
        _add_listed_chunks(builder, path, place, self._chunks, self.shape, self.chunk_shape)


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
