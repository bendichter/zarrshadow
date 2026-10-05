"""Build and write reference file systems, independent of the source format.

A reference file system (RFS) describes a Zarr v3 hierarchy whose metadata
and small datasets are stored in it and whose chunks are byte ranges in other
files. A generator for a file format walks its source and calls RfsBuilder:

    builder = RfsBuilder()
    builder.add_group("")
    builder.add_array("data", shape=[10_000, 16], data_type="int16", chunk_shape=[1000, 16])
    builder.add_strided_chunks("data", ndim=2, url=url, start=12, stride=32_000, length=32_000, count=10)
    rfs = builder.build()
    write_rfs(rfs, "data.zindi")

Chunk locations can be given one at a time (add_chunk), as a whole index
array for arrays with many chunks (add_index), as an arithmetic series for
evenly spaced chunks (add_strided_chunks, stored as a kerchunk "gen" entry),
or inline (add_inline_chunk). zindi.hdf5 is the generator for HDF5 files.

When an array is only part of what the file stores in each row, such as some
of the channels of an interleaved recording, add_selection says which bytes
of every row belong to it.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import zarr
from zarr.codecs import BloscCodec

from .chunk_index import MISSING, ChunkIndex, index_block_shape
from .gen import Generator
from .sources import describe_source

DEFAULT_CODECS = [{"name": "bytes", "configuration": {"endian": "little"}}]


def chunk_key(path: str, coords: Sequence[int]) -> str:
    """Zarr v3 key of the chunk at coords; a zero-dimensional array's only chunk is "c"."""
    prefix = f"{path}/c" if path else "c"
    return prefix + "".join(f"/{int(c)}" for c in coords)


def metadata_key(path: str) -> str:
    return f"{path}/zarr.json" if path else "zarr.json"


_ZARR_DATA_TYPES = {
    ("f", 2): "float16", ("f", 4): "float32", ("f", 8): "float64",
    ("i", 1): "int8", ("i", 2): "int16", ("i", 4): "int32", ("i", 8): "int64",
    ("u", 1): "uint8", ("u", 2): "uint16", ("u", 4): "uint32", ("u", 8): "uint64",
    ("b", 1): "bool",
}


def zarr_data_type(dtype: np.dtype) -> str:
    """The zarr v3 data_type name for a numeric numpy dtype."""
    dtype = np.dtype(dtype)
    result = _ZARR_DATA_TYPES.get((dtype.kind, dtype.itemsize))
    if result is None:
        raise ValueError(f"Unsupported dtype for zarr v3: {dtype}")
    return result


def bytes_codecs(dtype: np.dtype) -> list[dict]:
    """Codecs for uncompressed values in dtype's byte order."""
    endian = "big" if np.dtype(dtype).byteorder == ">" else "little"
    return [{"name": "bytes", "configuration": {"endian": endian}}]


def contiguous_chunk_shape(shape: Sequence[int], itemsize: int, target_bytes: int | None) -> list[int]:
    """Chunk shape for a C-ordered array stored in one piece: whole, or slabs along axis 0.

    Slabs hold about target_bytes. A slab height that divides the first axis
    is preferred, so that no slab is short.
    """
    shape_list = [int(s) for s in shape]
    if target_bytes is None or not shape_list:
        return shape_list
    row_bytes = itemsize * int(np.prod(shape_list[1:]))
    if row_bytes == 0 or row_bytes * shape_list[0] <= target_bytes:
        return shape_list
    rows = max(1, target_bytes // row_bytes)
    for candidate in range(rows, rows // 2, -1):
        if shape_list[0] % candidate == 0:
            return [candidate] + shape_list[1:]
    return [rows] + shape_list[1:]


def columns_selection(n_columns: int, itemsize: int, columns: slice | Sequence[int]) -> tuple[int, list[list[int]]]:
    """The record size and byte ranges that keep some columns of interleaved data.

    The file stores n_columns values of itemsize bytes for every row. columns
    is a slice or a list of column numbers, in the order the array has them.
    Pass the result to RfsBuilder.add_selection.
    """
    if isinstance(columns, slice):
        columns = range(*columns.indices(n_columns))
    keep: list[list[int]] = []
    for column in columns:
        column = int(column)
        if not 0 <= column < n_columns:
            raise ValueError(f"Column {column} is not one of the {n_columns} columns")
        start = column * itemsize
        if keep and keep[-1][1] == start:
            keep[-1][1] = start + itemsize
        else:
            keep.append([start, start + itemsize])
    if not keep:
        raise ValueError("No columns selected")
    return n_columns * itemsize, keep


class RfsBuilder:
    """Accumulates Zarr metadata and chunk locations for one reference file system."""

    def __init__(self) -> None:
        self.refs: dict[str, Any] = {}
        self.indexes: dict[str, dict] = {}
        self.gen: list[dict] = []
        self.selections: dict[str, dict] = {}

    # -- Metadata --

    def set_metadata(self, path: str, meta: dict) -> None:
        """Store a group's or array's zarr.json as given."""
        self.refs[metadata_key(path)] = json.dumps(meta, separators=(",", ":"))

    def add_group(self, path: str, attributes: dict | None = None) -> None:
        """Add a group; path "" is the root."""
        self.set_metadata(path, {"zarr_format": 3, "node_type": "group", "attributes": attributes or {}})

    def add_array(
        self,
        path: str,
        *,
        shape: Sequence[int],
        data_type: str | dict,
        chunk_shape: Sequence[int],
        codecs: list[dict] | None = None,
        fill_value: Any = 0,
        attributes: dict | None = None,
        dimension_names: Sequence[str | None] | None = None,
    ) -> dict:
        """Add an array with a regular chunk grid and return its zarr.json.

        codecs defaults to little-endian bytes with no compression, the layout
        of raw binary data on disk.
        """
        meta = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [int(s) for s in shape],
            "data_type": data_type,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [int(c) for c in chunk_shape]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": fill_value,
            "codecs": codecs if codecs is not None else DEFAULT_CODECS,
            "attributes": attributes or {},
            "storage_transformers": [],
        }
        if dimension_names is not None:
            meta["dimension_names"] = list(dimension_names)
        self.set_metadata(path, meta)
        return meta

    # -- Chunk locations --

    def add_chunk(self, path: str, coords: Sequence[int], url: str, offset: int, length: int) -> None:
        """One chunk: length bytes at offset in the file at url."""
        self.refs[chunk_key(path, coords)] = [url, int(offset), int(length)]

    def add_inline_chunk(self, path: str, coords: Sequence[int], data: bytes) -> None:
        """A chunk stored in the RFS itself: printable text as is, anything else as base64."""
        text = None
        if not data.startswith(b"base64:") and all(32 <= b < 127 or b in (9, 10, 13) for b in data):
            text = data.decode("ascii")
        self.refs[chunk_key(path, coords)] = text if text is not None else "base64:" + base64.b64encode(data).decode("ascii")

    def add_index(self, path: str, url: str, index: np.ndarray) -> None:
        """All chunks of an array as a uint64 array of shape (*chunk_grid, 2) holding (offset, nbytes).

        Chunks that were never written hold zindi.chunk_index.MISSING. Use this
        for arrays with many chunks: write_rfs stores the index as a Zarr array
        that readers load one index chunk at a time.
        """
        self.indexes[path] = {"url": url, "index": np.asarray(index, dtype=np.uint64)}

    def add_gen(self, key: str, url: str, offset: str, length: str, dimensions: dict) -> None:
        """A kerchunk gen entry; see zindi.gen for what the templates may contain."""
        self.gen.append({"key": key, "url": url, "offset": offset, "length": length, "dimensions": dimensions})

    def add_strided_chunks(
        self,
        path: str,
        *,
        ndim: int,
        url: str,
        start: int,
        stride: int,
        length: int,
        count: int,
    ) -> None:
        """Chunks 0..count-1 along the first axis, chunk i at start + i * stride.

        The other chunk coordinates are 0, so the chunk shape must span every
        other axis. This is the layout of a raw binary recording, or of a
        contiguous HDF5 dataset split into slabs.
        """
        if count <= 0:
            return
        rest = "/0" * (ndim - 1)
        self.add_gen(
            key=f"{path}/c/{{{{i}}}}{rest}",
            url=url,
            offset=f"{{{{{int(start)} + i * {int(stride)}}}}}",
            length=str(int(length)),
            dimensions={"i": {"stop": int(count)}},
        )

    def add_selection(self, path: str, record: int, keep: Sequence[Sequence[int]]) -> None:
        """Say which bytes of the file belong to an array, when its chunks are stored with other bytes.

        Every chunk reference of the array is read as consecutive records of
        record bytes. From each record the byte ranges in keep, given as
        [start, stop) pairs within the record, are taken and joined in the
        order listed; the rest is dropped. The chunk a reader decodes is the
        result for all of the records the reference covers.

        For a file holding 385 interleaved int16 channels, of which an array
        holds the first 384, a record is one sample of every channel:
        record=770 and keep=[[0, 768]]. For samples stored in packets with a
        14-byte header each, keep=[[14, record]]. Listing the ranges in
        another order reorders the columns. columns_selection builds the
        arguments for a choice of columns.

        The references of an array with a selection give the bytes in the
        file, so their lengths are multiples of record. A selection applies to
        uncompressed data.
        """
        record = int(record)
        spans = [[int(a), int(b)] for a, b in keep]
        if record <= 0 or not spans:
            raise ValueError("A selection needs a positive record size and at least one byte range")
        for a, b in spans:
            if not 0 <= a < b <= record:
                raise ValueError(f"Byte range [{a}, {b}) is not within a record of {record} bytes")
        self.selections[path] = {"record": record, "keep": spans}

    def add_contiguous_chunks(
        self,
        path: str,
        *,
        url: str,
        start: int,
        shape: Sequence[int],
        chunk_shape: Sequence[int],
        itemsize: int,
        file_size: int | None = None,
        row_bytes: int | None = None,
    ) -> None:
        """Chunks of an uncompressed C-ordered array stored in one piece at start.

        chunk_shape comes from contiguous_chunk_shape: the whole array, or slabs
        along the first axis spanning every other axis. The slabs become one gen
        entry. Every chunk a reader decodes must be full size, so a short last
        slab is read at full length when file_size shows the file extends that
        far (zarr discards the part of an edge chunk past the end of the array);
        otherwise it is a short ref, which RfsStore pads.

        row_bytes is the size in the file of one step along the first axis,
        when that is more than the array's own row because the array has a
        selection (see add_selection).
        """
        shape = [int(s) for s in shape]
        if row_bytes is None:
            row_bytes = int(np.prod(shape[1:])) * itemsize
        total = shape[0] * row_bytes if shape else itemsize
        origin = [0] * len(shape)
        if not shape or chunk_shape[0] >= shape[0]:
            self.add_chunk(path, origin, url, start, total)
            return
        slab = int(chunk_shape[0]) * row_bytes
        n_slabs = -(-shape[0] // int(chunk_shape[0]))
        last = start + (n_slabs - 1) * slab
        full_last = shape[0] % int(chunk_shape[0]) == 0 or (file_size is not None and last + slab <= file_size)
        self.add_strided_chunks(
            path, ndim=len(shape), url=url, start=start, stride=slab, length=slab,
            count=n_slabs if full_last else n_slabs - 1,
        )
        if not full_last:
            self.add_chunk(path, [n_slabs - 1] + origin[1:], url, last, start + total - last)

    def add_chunks(
        self,
        path: str,
        grid_shape: Sequence[int],
        url: str,
        chunks: dict[tuple[int, ...], tuple[int, int]],
        *,
        index_threshold: int | None = 1000,
    ) -> str:
        """All of an array's chunks in one file, stored in the most compact form.

        chunks maps chunk coordinates to (offset, length); missing coordinates
        are chunks that were never written. Chunks that are evenly spaced along
        the first axis, with every other axis a single chunk, become one gen
        entry; more than index_threshold chunks become an index array; anything
        else becomes one ref per chunk. Returns "strided", "index", or "refs".
        """
        grid = [int(g) for g in grid_shape]
        n = len(chunks)
        if n > 1 and n == grid[0] and all(g == 1 for g in grid[1:]):
            rest = (0,) * (len(grid) - 1)
            spans = [chunks.get((i, *rest)) for i in range(n)]
            if all(spans):
                offsets = [s[0] for s in spans]
                lengths = {s[1] for s in spans}
                strides = {b - a for a, b in zip(offsets, offsets[1:])}
                if len(lengths) == 1 and len(strides) == 1:
                    self.add_strided_chunks(
                        path, ndim=len(grid), url=url, start=offsets[0],
                        stride=strides.pop(), length=lengths.pop(), count=n,
                    )
                    return "strided"
        if index_threshold is not None and n > index_threshold:
            index = np.full((*grid, 2), MISSING, dtype=np.uint64)
            for coords, (offset, length) in chunks.items():
                index[coords] = (offset, length)
            self.add_index(path, url, index)
            return "index"
        for coords, (offset, length) in chunks.items():
            self.add_chunk(path, coords, url, offset, length)
        return "refs"

    # -- Result --

    def build(self, *, record_sources: bool = True) -> dict:
        """Return the RFS dict.

        With record_sources, the size and ETag of every referenced file are
        recorded under "sources" so readers can detect a file that has changed.
        URLs used many times are replaced by templates.
        """
        extended = self.indexes or self.gen or self.selections
        rfs: dict[str, Any] = {"refs": self.refs, "version": 2 if extended else 1}
        if self.indexes:
            rfs["indexes"] = self.indexes
        if self.gen:
            rfs["gen"] = self.gen
        if self.selections:
            rfs["selections"] = self.selections
        if record_sources:
            rfs["sources"] = _describe_sources(rfs)
        _apply_templates(rfs)
        return rfs


def write_rfs(rfs: dict, output_path: str) -> None:
    """Write a reference file system to disk.

    A path ending in ".json" writes a single version 1 reference file in which
    every chunk of an indexed array is listed in refs, readable by any kerchunk
    reader. Selections cannot be expressed in version 1, so a file that has
    them stays version 2. Any other path writes a directory holding refs.json and, for each
    chunk index, a zarr v3 array under index/<array path>. The "indexes" entry
    in refs.json gives that path, relative to refs.json, and the file is marked
    version 2 so that readers without index support refuse it. Opening the
    directory reads only refs.json; index chunks are read when needed.
    """
    if output_path.endswith(".json"):
        with open(output_path, "w") as f:
            json.dump(to_version1(rfs), f, indent=2, sort_keys=True)
        return

    refs_json = os.path.join(output_path, "refs.json")
    if os.path.isdir(output_path) and os.listdir(output_path) and not os.path.exists(refs_json):
        raise FileExistsError(f"{output_path} exists and is not a zindi RFS directory")
    index_dir = os.path.join(output_path, "index")
    if os.path.isdir(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(output_path, exist_ok=True)

    indexes = rfs.get("indexes", {})
    header = {k: v for k, v in rfs.items() if k != "indexes"}
    if indexes:
        header["version"] = 2
        header["indexes"] = {p: {"url": e["url"], "index": f"index/{p}"} for p, e in indexes.items()}
    with open(refs_json, "w") as f:
        json.dump(header, f, indent=2, sort_keys=True)

    store = zarr.storage.LocalStore(index_dir)
    for path, entry in indexes.items():
        data = np.asarray(ChunkIndex(entry["url"], entry["index"]).array[...])
        arr = zarr.create_array(
            store,
            name=path,
            shape=data.shape,
            dtype="uint64",
            chunks=(*index_block_shape(data.shape[:-1]), 2),
            fill_value=int(MISSING),
            compressors=BloscCodec(cname="zstd", clevel=5, shuffle="shuffle", typesize=8),
            zarr_format=3,
        )
        arr[...] = data


def to_version1(rfs: dict) -> dict:
    """Return a version 1 copy of rfs with every indexed or generated chunk listed in refs.

    fsspec skips "gen" entries unless asked not to, so they are expanded too.
    Selections have no version 1 form: a copy that has them keeps them and is
    marked version 2, so that kerchunk readers refuse it.
    """
    indexes = rfs.get("indexes", {})
    gens = rfs.get("gen", [])
    if not indexes and not gens:
        return rfs
    templates = rfs.get("templates", {})

    def expand(url: str) -> str:
        for key, value in templates.items():
            url = url.replace("{{" + key + "}}", value)
        return url

    refs = {
        key: [expand(val[0]), val[1], val[2]] if isinstance(val, list) and len(val) == 3 else val
        for key, val in rfs["refs"].items()
    }
    for path, entry in indexes.items():
        for coords, offset, nbytes in ChunkIndex(entry["url"], entry["index"]).iter_chunks():
            refs[chunk_key(path, coords)] = [entry["url"], offset, nbytes]
    for entry in gens:
        for key, ref in Generator(entry, templates).items():
            refs[key] = ref
    out = {k: v for k, v in rfs.items() if k not in ("indexes", "gen", "templates", "refs")}
    out["version"] = 2 if out.get("selections") else 1
    out["refs"] = refs
    _apply_templates(out)
    return out


def _describe_sources(rfs: dict) -> dict:
    """Size and ETag of every file the refs, indexes, and gen entries point into."""
    urls = {val[0] for val in rfs["refs"].values() if isinstance(val, list) and len(val) == 3}
    urls |= {entry["url"] for entry in rfs.get("indexes", {}).values()}
    urls |= {entry["url"] for entry in rfs.get("gen", [])}
    sources = {}
    for url in sorted(urls):
        try:
            sources[url] = describe_source(url)
        except Exception as e:
            warnings.warn(f"Could not describe source {url}: {e}")
    return sources


def _apply_templates(rfs: dict) -> None:
    """Replace URLs used by five or more refs with template placeholders."""
    refs = rfs["refs"]
    url_counts: dict[str, int] = {}
    for val in refs.values():
        if isinstance(val, list) and len(val) == 3:
            url_counts[val[0]] = url_counts.get(val[0], 0) + 1

    templates = {f"u{i}": url for i, url in enumerate(u for u, n in url_counts.items() if n >= 5)}
    if not templates:
        return
    url_to_template = {url: key for key, url in templates.items()}
    for val in refs.values():
        if isinstance(val, list) and len(val) == 3 and val[0] in url_to_template:
            val[0] = "{{" + url_to_template[val[0]] + "}}"
    rfs["templates"] = templates
