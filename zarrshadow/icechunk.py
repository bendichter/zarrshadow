"""Convert between reference file systems and Icechunk repositories.

Icechunk (https://icechunk.io) is a transactional store for Zarr v3. Besides
the chunks it stores itself, it holds virtual chunks, which are byte ranges of
other files, as the references of a reference file system are. rfs_to_icechunk
writes a reference file system into a repository, and icechunk_to_rfs writes
what a repository holds as a reference file system:

    import icechunk
    from zarrshadow.icechunk import icechunk_to_rfs, rfs_to_icechunk

    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer("file:///data/", icechunk.local_filesystem_store("/data"))
    )
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage("/repos/session"),
        config=config,
        authorize_virtual_chunk_access={"file:///data/": icechunk.credentials.LocalFileSystemAccess},
    )
    session = repo.writable_session("main")
    rfs_to_icechunk("session.nwb.zarrshadow", session)
    session.commit("Add the session")

    rfs = icechunk_to_rfs(repo.readonly_session("main"))

A repository reads virtual chunks only from the places its virtual chunk
containers name, so the repository needs a container for every file the
references point into.

Requires icechunk, and virtualizarr to read a repository
(pip install zarrshadow[icechunk]).
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from .builder import RfsBuilder, chunk_key, inline_value
from .chunk_index import MISSING, ChunkIndex
from .gen import Generator
from .open_rfs import load_rfs
from .rfs_store import RfsStore, _array_path

# The prefix icechunk_to_rfs gives the chunks a repository stores itself, until it has read them
_NATIVE = "icechunk-native:"
# How many chunks are copied into a repository at a time
_COPY_BATCH = 16


def rfs_to_icechunk(
    rfs: dict | str,
    session: Any,
    *,
    url_for: Callable[[str], str] | None = None,
    copy_selections: bool = False,
    validate_sources: bool = True,
) -> None:
    """Write a reference file system into an Icechunk repository.

    Groups, arrays, and attributes are written as they are. A reference to a
    byte range of a file becomes a virtual chunk, which still points at the
    file. Chunks stored in the references themselves are written into the
    repository. The session is left uncommitted.

    Icechunk has no counterpart for two things a reference file system can
    hold, so the chunks they concern are read from the source files and written
    into the repository:

    - a last chunk that is shorter than a full chunk, which RfsStore pads when
      it reads it. This is at most one chunk for each run of contiguous data.
    - an array with a selection, whose bytes are interleaved with others in the
      file. Every chunk of it would be copied, so this is refused unless
      copy_selections is set.

    Two things in the metadata change. Icechunk refuses NaN and Infinity as
    bare words, which is how zarr-python writes such attributes, so they are
    written as the strings "NaN", "Infinity", and "-Infinity" and read back as
    strings. Consolidated metadata, which hdmf-zarr writes in the root group,
    is left out: a repository already reads all of its metadata from one
    snapshot, and a second copy would go stale when the repository changes.

    Parameters
    ----------
    rfs : dict or str
        A reference file system dict, or the path or URL of one.
    session : icechunk.Session
        A writable session of the repository. Its root is written to.
    url_for : callable or None
        Maps each URL or path the references use to the location Icechunk
        should read, which must be within one of the repository's virtual
        chunk containers. By default an http(s) URL is used as it is and a
        local path becomes a file:// URL.
    copy_selections : bool
        Copy the data of arrays that have a selection into the repository.
    validate_sources : bool
        Check the source files that are read against the sizes and ETags the
        references record. Default True.

    Raises
    ------
    NotImplementedError
        If an array has a selection and copy_selections is not set.
    ValueError
        If a file is not within any of the repository's virtual chunk containers.
    """
    if isinstance(rfs, str):
        rfs = load_rfs(rfs)
    source = RfsStore(rfs, validate_sources=validate_sources)
    store = session.store
    refs = rfs["refs"]
    selections = rfs.get("selections", {})
    etags = {url: info["etag"] for url, info in rfs.get("sources", {}).items() if "etag" in info}
    locate = url_for if url_for is not None else _location

    # Metadata, with every group before what it holds
    arrays: dict[str, _Chunks] = {}
    metadata_keys = [key for key in refs if key == "zarr.json" or key.endswith("/zarr.json")]
    for key in sorted(metadata_keys, key=lambda key: (key.count("/"), key)):
        text = _metadata_for_icechunk(sync(source.get(key, default_buffer_prototype())).to_bytes().decode())
        sync(store.set(key, default_buffer_prototype().buffer.from_bytes(text.encode())))
        meta = json.loads(text)
        if meta.get("node_type") == "array":
            path = key[: -len("/zarr.json")] if "/" in key else ""
            arrays[path] = _Chunks(path, meta)

    # Where every chunk is. A key in refs comes before the same key in an index or a gen entry.
    for entry in rfs.get("gen", []):
        for key, ref in Generator(entry, rfs.get("templates", {})).items():
            if _array_path(key) in arrays:
                arrays[_array_path(key)].add(key, ref)
    for path, entry in rfs.get("indexes", {}).items():
        arrays[path].add_index(entry["url"], np.asarray(ChunkIndex(entry["url"], entry["index"]).array[...]))
    for key, ref in refs.items():
        if _array_path(key) in arrays:
            ref = [source._expand_templates(ref[0]), *ref[1:]] if isinstance(ref, list) else ref
            arrays[_array_path(key)].add(key, ref)

    for path, chunks in arrays.items():
        if path in selections:
            if not copy_selections:
                raise NotImplementedError(
                    f"{path!r} has a selection, which Icechunk cannot express. Pass copy_selections=True "
                    "to copy its data into the repository."
                )
            chunks.copy(chunks.urls != "")
        else:
            # A chunk that RfsStore would pad is not complete in the file
            full = source._get_padded_size(chunk_key(path, [0] * len(chunks.grid)), 0) if chunks.grid else None
            if full is not None:
                chunks.copy((chunks.urls != "") & (chunks.lengths < full))
        sync(_copy_chunks(source, store, chunks.copied))
        _set_virtual_chunks(store, chunks, locate, etags)


def icechunk_to_rfs(
    session: Any,
    *,
    native_chunks_prefix: str | None = None,
    url_for: Callable[[str], str] | None = None,
    index_threshold: int | None = 1000,
    record_sources: bool = True,
) -> dict:
    """Write what an Icechunk repository holds as a reference file system.

    A virtual chunk becomes a reference to the same byte range. The chunks the
    repository stores itself are read and stored in the references, unless
    native_chunks_prefix says where they can be referenced. VirtualiZarr's
    IcechunkParser reads the repository.

    Parameters
    ----------
    session : icechunk.Session
        A session of the repository, at the snapshot to convert.
    native_chunks_prefix : str or None
        The URL or path of the repository's chunks directory, such as
        "/repos/session/chunks". With it, a chunk the repository stores
        becomes a reference to its file there, which keeps the references
        small but leaves them depending on the repository, where garbage
        collection can delete a chunk that later snapshots no longer use.
        Without it, those chunks are copied into the references, which suits
        repositories whose own chunks are small, such as coordinates and strings.
    url_for : callable or None
        Maps the location of each referenced file to the URL or path the
        references should use. By default a file:// URL becomes a local path
        and anything else is kept. RfsStore reads local paths and http(s)
        URLs, so locations of other kinds, such as s3://, need mapping.
    index_threshold : int or None
        An array with more chunks than this, all in one file, gets a chunk
        index in place of one ref per chunk.
    record_sources : bool
        Record the size and, for remote URLs, the ETag of each referenced file.

    Returns
    -------
    dict
        A reference file system dict; see zarrshadow.builder.
    """
    from obspec_utils.registry import ObjectStoreRegistry
    from virtualizarr.parsers import IcechunkParser

    from .virtualizarr import _add_manifest_store

    native = native_chunks_prefix is None
    parsed = IcechunkParser().parse_session(
        session, ObjectStoreRegistry(), native_chunks_prefix=_NATIVE if native else native_chunks_prefix
    )
    locate = url_for if url_for is not None else _path
    builder = RfsBuilder()
    _add_manifest_store(
        builder,
        parsed,
        index_threshold=index_threshold,
        url_for=lambda location: location if location.startswith(_NATIVE) else locate(location),
    )
    if native:
        keys = [key for key, ref in builder.refs.items() if isinstance(ref, list) and ref[0].startswith(_NATIVE)]
        for key, data in zip(keys, sync(_read_chunks(session.store, keys))):
            builder.refs[key] = inline_value(data)
    return builder.build(record_sources=record_sources)


class _Chunks:
    """Where the chunks of one array are, in arrays over its chunk grid flattened in C order.

    A chunk with no URL is not referenced: it was never written, or it is one of
    those in copied, which are read through RfsStore and written into the repository.
    """

    def __init__(self, path: str, meta: dict) -> None:
        if meta["chunk_grid"]["name"] != "regular":
            raise NotImplementedError(f"Unsupported chunk grid {meta['chunk_grid']['name']!r} at {path!r}")
        chunk_shape = meta["chunk_grid"]["configuration"]["chunk_shape"]
        self.path = path
        self.grid = tuple(-(-int(s) // int(c)) for s, c in zip(meta["shape"], chunk_shape))
        n = int(np.prod(self.grid))
        self.urls = np.full(n, "", dtype=object)
        self.offsets = np.zeros(n, dtype=np.uint64)
        self.lengths = np.zeros(n, dtype=np.uint64)
        self.copied: list[str] = []

    def add(self, key: str, ref: Any) -> None:
        """One chunk: [url, offset, length], or anything else for a chunk stored in the references."""
        prefix = f"{self.path}/c" if self.path else "c"
        coords = [int(part) for part in key[len(prefix) + 1 :].split("/")] if len(key) > len(prefix) else []
        i = int(np.ravel_multi_index(coords, self.grid)) if coords else 0
        if isinstance(ref, list):
            self.urls[i], self.offsets[i], self.lengths[i] = ref
        else:
            self.urls[i] = ""
            self.copied.append(key)

    def add_index(self, url: str, index: np.ndarray) -> None:
        """Every chunk of the array, from its chunk index."""
        index = index.reshape(-1, 2)
        present = index[:, 0] != MISSING
        self.urls[present] = url
        self.offsets[present], self.lengths[present] = index[present, 0], index[present, 1]

    def copy(self, which: np.ndarray) -> None:
        """Copy the referenced chunks that which marks, in place of referencing them."""
        for i in np.flatnonzero(which):
            self.copied.append(chunk_key(self.path, np.unravel_index(i, self.grid) if self.grid else ()))
        self.urls[which] = ""


def _set_virtual_chunks(
    store: Any, chunks: _Chunks, locate: Callable[[str], str], etags: dict[str, str]
) -> None:
    """Write an array's referenced chunks as virtual chunks, each with the ETag recorded for its file."""
    urls = sorted(set(chunks.urls.tolist()) - {""})
    locations = {url: locate(url) for url in urls}
    for etag in sorted({etags.get(url, "") for url in urls}):
        chosen = {url: location for url, location in locations.items() if etags.get(url, "") == etag}
        failed = store.set_virtual_refs_arr(
            array_path=chunks.path,
            chunk_grid_shape=chunks.grid,
            locations=[chosen.get(url, "") for url in chunks.urls.tolist()],
            offsets=chunks.offsets,
            lengths=chunks.lengths,
            checksum=etag or None,
        )
        if failed:
            location = chosen[chunks.urls[int(np.ravel_multi_index(failed[0], chunks.grid)) if chunks.grid else 0]]
            raise ValueError(
                f"{location} is not within any virtual chunk container of the repository. Add one with "
                "RepositoryConfig.set_virtual_chunk_container, or map the location with url_for."
            )


async def _copy_chunks(source: RfsStore, store: Any, keys: Sequence[str]) -> None:
    """Read chunks through RfsStore and write them into the repository, a few at a time."""
    prototype = default_buffer_prototype()

    async def copy(key: str) -> None:
        data = await source.get(key, prototype)
        if data is not None:
            await store.set(key, data)

    for batch in itertools.batched(keys, _COPY_BATCH):
        await asyncio.gather(*(copy(key) for key in batch))


async def _read_chunks(store: Any, keys: Sequence[str]) -> list[bytes]:
    """The chunks a repository stores at keys."""
    prototype = default_buffer_prototype()
    return [(await store.get(key, prototype)).to_bytes() for key in keys]


def _metadata_for_icechunk(text: str) -> str:
    """A zarr.json as Icechunk takes it.

    Each bare NaN, Infinity, and -Infinity becomes a string, and consolidated
    metadata is left out.
    """
    found: list[str] = []

    def constant(name: str) -> str:
        found.append(name)
        return name

    meta = json.loads(text, parse_constant=constant)
    if not found and "consolidated_metadata" not in meta:
        return text
    meta.pop("consolidated_metadata", None)
    return json.dumps(meta, separators=(",", ":"))


def _location(url: str) -> str:
    """What a reference points into as the URL Icechunk reads: a local path becomes a file:// URL."""
    return url if "://" in url else Path(os.path.abspath(url)).as_uri()


def _path(location: str) -> str:
    """The location of a virtual chunk as zarrshadow refers to files: a file:// URL becomes a local path."""
    return unquote(location[len("file://") :]) if location.startswith("file://") else location
