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


class RfsBuilder:
    """Accumulates Zarr metadata and chunk locations for one reference file system."""

    def __init__(self) -> None:
        self.refs: dict[str, Any] = {}
        self.indexes: dict[str, dict] = {}
        self.gen: list[dict] = []

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

    # -- Result --

    def build(self, *, record_sources: bool = True) -> dict:
        """Return the RFS dict.

        With record_sources, the size and ETag of every referenced file are
        recorded under "sources" so readers can detect a file that has changed.
        URLs used many times are replaced by templates.
        """
        rfs: dict[str, Any] = {"refs": self.refs, "version": 2 if (self.indexes or self.gen) else 1}
        if self.indexes:
            rfs["indexes"] = self.indexes
        if self.gen:
            rfs["gen"] = self.gen
        if record_sources:
            rfs["sources"] = _describe_sources(rfs)
        _apply_templates(rfs)
        return rfs


def write_rfs(rfs: dict, output_path: str) -> None:
    """Write a reference file system to disk.

    A path ending in ".json" writes a single version 1 reference file in which
    every chunk of an indexed array is listed in refs, readable by any kerchunk
    reader. Any other path writes a directory holding refs.json and, for each
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
    out["version"] = 1
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
