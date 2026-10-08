"""Zarr v3 Store backed by a reference file system (RFS).

This is a zarr v3 Store that reads data from a reference file system dict.
It handles:
- Inline data (strings, base64-encoded bytes, JSON dicts)
- Remote chunk references [url, offset, size]
- URL template expansion ({{u1}} etc.)
- DANDI URL resolution (redirects + auth)
- Retry with exponential backoff
- Chunk padding for contiguous HDF5 datasets
- Chunk indexes for arrays with many chunks (see chunk_index.py)
- kerchunk "gen" entries, evaluated when a key is requested (see gen.py)
- Selections, for arrays stored with other bytes in every record (see
  RfsBuilder.add_selection)
- Byte range requests, which fetch only the part of a chunk that is asked for

Ported from lindi's LindiReferenceFileSystemStore, adapted for zarr v3.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import requests
from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype

from .chunk_index import ChunkIndex
from .gen import Generator
from .sources import SourceChangedError, SourceChecker
from .url_resolver import resolve_url


_imagecodecs_registered = False


def _register_imagecodecs_if_needed(refs: dict) -> None:
    """Register the imagecodecs Zarr codecs that TIFF references may use."""
    global _imagecodecs_registered
    if _imagecodecs_registered:
        return
    if not any(
        isinstance(v, str) and '"imagecodecs_' in v
        for k, v in refs.items()
        if k == "zarr.json" or k.endswith("/zarr.json")
    ):
        return
    try:
        import imagecodecs.zarr
    except ImportError as e:
        raise ImportError(
            "These references use imagecodecs codecs (from a TIFF file); install imagecodecs to read them"
        ) from e
    imagecodecs.zarr.register_codecs()
    _imagecodecs_registered = True


class RfsStore(Store):
    """A read-only zarr v3 Store backed by a reference file system dict.

    Parameters
    ----------
    rfs : dict
        Reference file system dict with "refs" key, and optional "templates".
    local_cache : LocalCache or None
        Optional local cache for persisting remote chunk data on disk.
    """

    def __init__(
        self,
        rfs: dict,
        *,
        local_cache: Any = None,
        merge_gap: int = 32 * 1024,
        max_merge_size: int = 2**20,
        merge_below: int = 64 * 1024,
        validate_sources: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        rfs : dict
            Reference file system dict with "refs" key, and optional
            "templates", "gen", "indexes", "selections", and "sources".
        local_cache : LocalCache or None
            Optional local cache for persisting remote chunk data on disk.
        merge_gap : int
            Chunks of a remote file that are asked for at the same time are
            fetched in one request when the gap between them is at most this
            many bytes. Default 32 KiB.
        max_merge_size : int
            The most bytes one merged request may ask for. 0 turns merging
            off. Default 1 MiB.
        merge_below : int
            Only chunks of at most this many bytes are merged. Larger ones are
            fetched faster on their own, in parallel. Default 64 KiB.
        validate_sources : bool
            Check reads against the size and ETag recorded under "sources" and
            raise SourceChangedError if a file has changed. Default True.
        """
        super().__init__(read_only=True)
        if "refs" not in rfs:
            raise ValueError("rfs must contain a 'refs' key")
        self.rfs = rfs
        self._local_cache = local_cache
        self._merge_gap = merge_gap
        self._max_merge_size = max_merge_size
        self._merge_below = merge_below
        # Small reads of remote files that are waiting to be sent together, for each event loop
        self._pending: dict[Any, dict[str, list[tuple[int, int, str, asyncio.Future]]]] = {}
        self._executor = ThreadPoolExecutor(max_workers=32)
        self._sources = SourceChecker(rfs.get("sources", {}), enabled=validate_sources)
        _register_imagecodecs_if_needed(rfs["refs"])
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "Mozilla/5.0"
        self._indexes = {
            path: ChunkIndex(entry["url"], entry["index"])
            for path, entry in rfs.get("indexes", {}).items()
        }
        self._generators = [Generator(entry, rfs.get("templates", {})) for entry in rfs.get("gen", [])]
        self._selections = {path: Selection(**entry) for path, entry in rfs.get("selections", {}).items()}
        self._children: dict[str, set[str]] | None = None
        self._array_meta: dict[str, dict | None] = {}
        self._is_open = True

    # -- Abstract method implementations --

    def __eq__(self, value: object) -> bool:
        return isinstance(value, RfsStore) and value.rfs is self.rfs

    @property
    def supports_writes(self) -> bool:  # type: ignore[override]
        return False

    @property
    def supports_deletes(self) -> bool:  # type: ignore[override]
        return False

    @property
    def supports_listing(self) -> bool:  # type: ignore[override]
        return True

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if prototype is None:
            prototype = default_buffer_prototype()
        loop = asyncio.get_running_loop()
        if byte_range is None and self._max_merge_size > 0:
            data = await self._get_merged(key, loop)
        else:
            data = await loop.run_in_executor(self._executor, self._get_bytes, key, byte_range)
        if data is None:
            return None
        return prototype.buffer.from_bytes(data)

    async def _get_merged(self, key: str, loop: asyncio.AbstractEventLoop) -> bytes | None:
        """Read a whole value, sharing a request with other small chunks of the same remote file.

        zarr asks for the chunks of a selection at the same time, each with its
        own get. A small chunk of a remote file waits a moment for the others,
        and those that are close together in the file are fetched in one
        request. Everything else is read on its own.
        """
        if self._indexes and key.rpartition("/c/")[0] in self._indexes:
            # A chunk index may have to be read, which blocks
            ref = await loop.run_in_executor(self._executor, self._resolve, key)
        else:
            ref = self._resolve(key)
        mergeable = isinstance(ref, list) and len(ref) == 3 and 0 < ref[2] <= self._merge_below
        url = self._expand_templates(ref[0]) if mergeable else ""
        if not url.startswith(("http://", "https://")):
            return await loop.run_in_executor(self._executor, self._get_bytes, key, None)
        if self._local_cache is not None:
            cached = self._local_cache.get_remote_chunk(url=url, offset=ref[1], size=ref[2])
            if cached is not None:
                return self._finish(key, cached)
        future: asyncio.Future = loop.create_future()
        pending = self._pending.setdefault(loop, {})
        if not pending:
            loop.call_later(0.001, self._send_pending, loop)
        pending.setdefault(url, []).append((ref[1], ref[2], key, future))
        return await future

    def _send_pending(self, loop: asyncio.AbstractEventLoop) -> None:
        """Send the reads that are waiting, one request for each run that is close together in a file."""
        for url, reads in self._pending.pop(loop, {}).items():
            reads.sort(key=lambda read: read[0])
            runs: list[list[Any]] = []  # [start, end, reads]
            for read in reads:
                end = read[0] + read[1]
                if (
                    runs
                    and read[0] <= runs[-1][1] + self._merge_gap
                    and max(runs[-1][1], end) - runs[-1][0] <= self._max_merge_size
                ):
                    runs[-1][1] = max(runs[-1][1], end)
                    runs[-1][2].append(read)
                else:
                    runs.append([read[0], end, [read]])
            for start, end, run in runs:
                task = loop.run_in_executor(self._executor, self._read_run, url, start, end, run)
                task.add_done_callback(lambda done, run=run: self._deliver(done, run))

    def _read_run(self, url: str, start: int, end: int, run: list) -> list[bytes]:
        """Fetch [start, end) of a remote file and cut out the chunk of each read in the run."""
        raw = _read_bytes_from_url(url, start, end - start, session=self._session, checker=self._sources)
        chunks = []
        for offset, length, key, _ in run:
            data = raw[offset - start : offset - start + length]
            if self._local_cache is not None:
                from .local_cache import ChunkTooLargeError

                try:
                    self._local_cache.put_remote_chunk(url=url, offset=offset, size=length, data=data)
                except ChunkTooLargeError:
                    pass
            chunks.append(self._finish(key, data))
        return chunks

    @staticmethod
    def _deliver(done: asyncio.Future, run: list) -> None:
        """Give each read of a run its chunk, or the error the request ended with."""
        error = done.exception() if not done.cancelled() else asyncio.CancelledError()
        for i, (_, _, _, future) in enumerate(run):
            if future.done():
                continue
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(done.result()[i])

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Any,
    ) -> list[Buffer | None]:
        loop = asyncio.get_running_loop()
        # Separate remote byte-range refs (mergeable) from everything else
        items = list(key_ranges)
        results: list[Buffer | None] = [None] * len(items)
        non_remote_indices = []
        # Group remote refs by resolved URL for merging
        url_groups: dict[str, list[tuple[int, int, int, str]]] = {}  # url -> [(item_idx, offset, length, key)]

        # Resolving may read chunk index blocks, which blocks, so run it off the event loop
        resolved = await loop.run_in_executor(
            self._executor, lambda: [self._resolve(key) for key, _ in items]
        )
        for i, ((key, byte_range), ref) in enumerate(zip(items, resolved)):
            if byte_range is not None or not (isinstance(ref, list) and len(ref) == 3):
                non_remote_indices.append(i)
                continue
            url_or_path = self._expand_templates(ref[0])
            if not (url_or_path.startswith("http://") or url_or_path.startswith("https://")):
                non_remote_indices.append(i)
                continue
            url_groups.setdefault(url_or_path, []).append((i, ref[1], ref[2], key))

        # Fetch non-remote items individually (inline data, local files, etc.)
        if non_remote_indices:
            fetched = await asyncio.gather(
                *(self.get(items[i][0], prototype, items[i][1]) for i in non_remote_indices)
            )
            for idx, buf in zip(non_remote_indices, fetched):
                results[idx] = buf

        # For each URL, merge nearby ranges and fetch
        if url_groups:
            fetch_tasks = []
            for url, refs_for_url in url_groups.items():
                fetch_tasks.append(
                    loop.run_in_executor(
                        self._executor, self._fetch_merged_ranges, url, refs_for_url, prototype
                    )
                )
            fetched_groups = await asyncio.gather(*fetch_tasks)
            for group_results in fetched_groups:
                for item_idx, buf in group_results:
                    results[item_idx] = buf

        return results

    def _fetch_merged_ranges(
        self,
        url: str,
        refs: list[tuple[int, int, int, str]],  # (item_idx, offset, length, key)
        prototype: BufferPrototype,
    ) -> list[tuple[int, Buffer | None]]:
        """Fetch byte ranges from a single URL, merging nearby ranges."""
        # Sort by offset
        sorted_refs = sorted(refs, key=lambda r: r[1])

        # Build merged ranges, respecting merge_gap and max_merge_size
        merged: list[tuple[int, int, list[tuple[int, int, int, str]]]] = []  # (start, end, refs)
        for ref in sorted_refs:
            item_idx, offset, length, key = ref
            end = offset + length
            if merged:
                new_end = max(merged[-1][1], end)
                gap_ok = offset <= merged[-1][1] + self._merge_gap
                size_ok = new_end - merged[-1][0] <= self._max_merge_size
                if gap_ok and size_ok:
                    merged[-1] = (merged[-1][0], new_end, merged[-1][2] + [ref])
                    continue
            merged.append((offset, end, [ref]))

        # Fetch each merged range and split
        results: list[tuple[int, Buffer | None]] = []
        for start, end, group_refs in merged:
            # Check cache for individual chunks first
            uncached: list[tuple[int, int, int, str]] = []
            for item_idx, offset, length, key in group_refs:
                cached_data = None
                if self._local_cache is not None:
                    cached_data = self._local_cache.get_remote_chunk(
                        url=url, offset=offset, size=length
                    )
                if cached_data is not None:
                    results.append((item_idx, prototype.buffer.from_bytes(self._finish(key, cached_data))))
                else:
                    uncached.append((item_idx, offset, length, key))

            if not uncached:
                continue

            # Re-compute merged range for uncached items only
            uncached_start = min(r[1] for r in uncached)
            uncached_end = max(r[1] + r[2] for r in uncached)

            # Fetch the merged range
            raw = _read_bytes_from_url(
                url,
                uncached_start,
                uncached_end - uncached_start,
                session=self._session,
                checker=self._sources,
            )

            # Split and deliver individual chunks
            for item_idx, offset, length, key in uncached:
                chunk_data = raw[offset - uncached_start:offset - uncached_start + length]

                # Cache individual chunks
                if self._local_cache is not None:
                    from .local_cache import ChunkTooLargeError
                    try:
                        self._local_cache.put_remote_chunk(
                            url=url, offset=offset, size=length, data=chunk_data
                        )
                    except ChunkTooLargeError:
                        pass

                results.append((item_idx, prototype.buffer.from_bytes(self._finish(key, chunk_data))))

        return results

    async def exists(self, key: str) -> bool:
        if key in self.rfs["refs"]:
            return True
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._resolve, key) is not None

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("RfsStore is read-only")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("RfsStore is read-only")

    async def list(self) -> AsyncIterator[str]:
        for key in self.rfs["refs"]:
            yield key
        for path, index in self._indexes.items():
            for coords, _, _ in index.iter_chunks():
                yield f"{path}/c/" + "/".join(map(str, coords))
        for generator in self._generators:
            for key, _ in generator.items():
                yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        prefix = prefix.strip("/")
        indexed = self._indexed_chunk_prefix(prefix)
        if indexed is not None:
            # Inside an indexed array's chunk directory: list chunk coordinates
            path, parts = indexed
            depth = len(parts)
            seen: set[str] = set()
            for coords, _, _ in self._indexes[path].iter_chunks():
                if tuple(map(str, coords[:depth])) == parts and depth < len(coords):
                    name = str(coords[depth])
                    if name not in seen:
                        seen.add(name)
                        yield name
            return
        names = set(self._get_children().get(prefix, ()))
        for generator in self._generators:
            # Enumerate generated keys only when listing inside a generated directory
            if (prefix + "/").startswith(generator.static_prefix) and "/" in generator.static_prefix:
                for key, _ in generator.items():
                    if key.startswith(prefix + "/"):
                        names.add(key[len(prefix) + 1 :].split("/")[0])
        for name in sorted(names):
            yield name

    def _get_children(self) -> dict[str, set[str]]:
        """Map each directory prefix to its immediate children, built once."""
        if self._children is None:
            children: dict[str, set[str]] = {}
            keys = list(self.rfs["refs"]) + [f"{path}/c" for path in self._indexes]
            keys += [g.static_prefix.rsplit("/", 1)[0] for g in self._generators if "/" in g.static_prefix]
            for key in keys:
                parts = key.split("/")
                for i in range(len(parts)):
                    children.setdefault("/".join(parts[:i]), set()).add(parts[i])
            self._children = children
        return self._children

    def _indexed_chunk_prefix(self, prefix: str) -> tuple[str, tuple[str, ...]] | None:
        """If prefix is <indexed array>/c[/...], return (array path, coordinate parts)."""
        for path in self._indexes:
            chunk_dir = f"{path}/c"
            if prefix == chunk_dir or prefix.startswith(chunk_dir + "/"):
                rest = prefix[len(chunk_dir) + 1:]
                return path, tuple(rest.split("/")) if rest else ()
        return None

    # -- Core data resolution --

    def _resolve(self, key: str) -> Any:
        """Return the ref for key: inline str/dict, [url, offset, size], or None.

        Keys are looked up in refs, then in the chunk index of their array,
        then in the gen entries.
        """
        ref = self.rfs["refs"].get(key)
        if ref is not None:
            return ref
        if self._indexes:
            path, sep, coords_str = key.rpartition("/c/")
            index = self._indexes.get(path) if sep else None
            if index is not None:
                try:
                    coords = tuple(int(c) for c in coords_str.split("/"))
                except ValueError:
                    return None
                hit = index.lookup(coords)
                return None if hit is None else [index.url, hit[0], hit[1]]
        for generator in self._generators:
            ref = generator.lookup(key)
            if ref is not None:
                return ref
        return None

    def _expand_templates(self, url_or_path: str) -> str:
        if "{{" in url_or_path and "}}" in url_or_path and "templates" in self.rfs:
            for tkey, tval in self.rfs["templates"].items():
                url_or_path = url_or_path.replace("{{" + tkey + "}}", tval)
        return url_or_path

    def _get_bytes(self, key: str, byte_range: ByteRequest | None = None) -> bytes | None:
        """Resolve a key to bytes, handling all reference types.

        With byte_range, only that part of the value is returned, and for a
        reference into a file only the bytes it needs are read.
        """
        x = self._resolve(key)
        if x is None:
            return None

        if isinstance(x, str):
            if x.startswith("base64:"):
                data = base64.b64decode(x[len("base64:"):])
            else:
                data = x.encode("utf-8")
        elif isinstance(x, dict):
            data = json.dumps(x).encode("utf-8")
        elif isinstance(x, list):
            if len(x) != 3:
                raise ValueError(f"Reference list for {key} must have 3 elements")
            url_or_path, offset, length = self._expand_templates(x[0]), x[1], x[2]
            if byte_range is not None:
                return self._read_part(key, url_or_path, offset, length, byte_range)
            return self._finish(key, self._read_source(url_or_path, offset, length))
        else:
            raise ValueError(f"Unexpected reference type for {key}: {type(x)}")
        return data if byte_range is None else _apply_byte_range(data, byte_range)

    def _read_source(self, url_or_path: str, offset: int, length: int) -> bytes:
        """Read a whole reference from its file, through the local cache for remote files."""
        is_url = url_or_path.startswith("http://") or url_or_path.startswith("https://")
        if self._local_cache is not None and is_url:
            cached = self._local_cache.get_remote_chunk(url=url_or_path, offset=offset, size=length)
            if cached is not None:
                return cached

        data = _read_bytes_from_url_or_path(
            url_or_path, offset, length, session=self._session, checker=self._sources
        )

        if self._local_cache is not None and is_url:
            from .local_cache import ChunkTooLargeError

            try:
                self._local_cache.put_remote_chunk(url=url_or_path, offset=offset, size=length, data=data)
            except ChunkTooLargeError:
                pass  # chunk exceeds SQLite blob limit, skip caching
        return data

    def _finish(self, key: str, data: bytes) -> bytes:
        """Turn the bytes of a reference into the chunk a reader decodes.

        The array's selection, if it has one, is applied, and a final chunk of
        a contiguous dataset that is shorter than a full chunk is padded.
        """
        selection = self._selection_for(key)
        if selection is not None:
            data = selection.apply(data)
        padded_size = self._get_padded_size(key, len(data))
        if padded_size is not None:
            data = data + b"\0" * (padded_size - len(data))
        return data

    def _read_part(self, key: str, url_or_path: str, offset: int, length: int, byte_range: ByteRequest) -> bytes:
        """Read part of a chunk, fetching only the bytes of the file that hold it."""
        selection = self._selection_for(key)
        stored = length if selection is None else selection.selected_size(length)
        total = self._get_padded_size(key, stored) or stored
        start, stop = _range_bounds(byte_range, total)
        if stop <= start:
            return b""
        available = min(stop, stored)  # past this, the chunk is padding
        data = b""
        if start < available:
            if selection is None:
                data = _read_bytes_from_url_or_path(
                    url_or_path, offset + start, available - start, session=self._session, checker=self._sources
                )
            else:
                source_start, source_length, skip = selection.source_range(start, available)
                data = _read_bytes_from_url_or_path(
                    url_or_path, offset + source_start, source_length, session=self._session, checker=self._sources
                )
                data = selection.apply(data)[skip : skip + available - start]
        return data + b"\0" * (stop - start - len(data))

    def _selection_for(self, key: str) -> Selection | None:
        if not self._selections:
            return None
        path = _array_path(key)
        return None if path is None else self._selections.get(path)

    def _get_padded_size(self, key: str, nbytes: int) -> int | None:
        """The full size of a chunk of nbytes that needs padding (final chunk in contiguous dataset).

        Only uncompressed chunks are padded. A compressed chunk is shorter than
        its decoded size by design, and zeros appended to it break codecs that
        read to the end of their input (fletcher32, zstd).

        In zarr v3, chunk keys look like: path/c/0/1/2
        """
        parts = key.split("/")
        # Find the 'c' separator - everything after it is chunk indices
        try:
            c_idx = parts.index("c")
        except ValueError:
            return None

        if c_idx >= len(parts) - 1:
            return None

        # Check that everything after 'c' is an integer (chunk index)
        for p in parts[c_idx + 1:]:
            try:
                int(p)
            except ValueError:
                return None

        # Get the zarr.json for this array
        array_path = "/".join(parts[:c_idx])
        meta_key = f"{array_path}/zarr.json" if array_path else "zarr.json"

        if meta_key not in self._array_meta:
            meta_bytes = self._get_bytes(meta_key) if meta_key in self.rfs["refs"] else None
            meta = json.loads(meta_bytes) if meta_bytes is not None else None
            self._array_meta[meta_key] = meta if meta and meta.get("node_type") == "array" else None
        meta = self._array_meta[meta_key]
        if meta is None:
            return None
        if any(codec.get("name") != "bytes" for codec in meta.get("codecs", [])):
            return None

        chunk_shape = meta.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape")
        data_type = meta.get("data_type")
        if chunk_shape is None or data_type is None:
            return None

        if isinstance(data_type, dict):
            if data_type.get("name") in ("struct", "structured"):
                # struct lists its fields as objects; structured, its earlier name, as pairs
                fields = [
                    (f["name"], f["data_type"]) if isinstance(f, dict) else (f[0], f[1])
                    for f in data_type["configuration"]["fields"]
                ]
                dtype = np.dtype([(name, _zarr_field_type_to_numpy(field)) for name, field in fields])
            else:
                return None
        elif isinstance(data_type, str):
            dtype = np.dtype(data_type)
        else:
            return None
        if dtype.kind not in ("i", "u", "f", "V"):
            return None

        expected_size = int(np.prod(chunk_shape)) * dtype.itemsize
        if nbytes < expected_size:
            return expected_size

        return None


def _zarr_field_type_to_numpy(field_type: str | dict) -> str:
    """Convert a zarr v3 field type to a numpy dtype string."""
    if isinstance(field_type, str):
        return field_type
    if isinstance(field_type, dict):
        name = field_type.get("name")
        if name == "null_terminated_bytes":
            length = field_type["configuration"]["length_bytes"]
            return f"S{length}"
        if name == "fixed_length_utf32":
            # length_bytes is total bytes; each UTF-32 char is 4 bytes
            length = field_type["configuration"]["length_bytes"] // 4
            return f"U{length}"
    raise ValueError(f"Unsupported zarr field type: {field_type}")


def _read_bytes_from_url_or_path(
    url_or_path: str,
    offset: int,
    length: int,
    *,
    session: requests.Session | None = None,
    checker: SourceChecker | None = None,
) -> bytes:
    """Read a byte range from a URL or local file path."""
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return _read_bytes_from_url(url_or_path, offset, length, session=session, checker=checker)
    else:
        if checker is not None:
            checker.check_local(url_or_path)
        with open(url_or_path, "rb") as f:
            f.seek(offset)
            return f.read(length)


def _read_bytes_from_url(
    url: str,
    offset: int,
    length: int,
    *,
    session: requests.Session | None = None,
    checker: SourceChecker | None = None,
) -> bytes:
    """Read a byte range from a URL with retry and DANDI resolution.

    With a checker, the request carries If-Match for the recorded ETag, and a
    response showing the file changed raises SourceChangedError without retrying.
    """
    num_retries = 8
    for try_num in range(num_retries):
        try:
            resolved_url = resolve_url(url)
            range_header = f"bytes={offset}-{offset + length - 1}"
            headers = {"Range": range_header}
            if checker is not None:
                headers.update(checker.request_headers(url))
            if session is not None:
                response = session.get(resolved_url, headers=headers)
            else:
                headers["User-Agent"] = "Mozilla/5.0"
                response = requests.get(resolved_url, headers=headers)
            if checker is not None:
                checker.check_response(url, response)
            response.raise_for_status()
            return response.content
        except SourceChangedError:
            raise
        except Exception as e:
            if try_num == num_retries - 1:
                raise
            delay = 0.1 * 2**try_num
            print(f"Retry {try_num + 1}/{num_retries} for {url} in {delay:.1f}s: {e}")
            time.sleep(delay)
    raise RuntimeError(f"Failed to read from {url}")


def _range_bounds(byte_range: ByteRequest, total: int) -> tuple[int, int]:
    """The [start, stop) that a ByteRequest covers in a value of total bytes."""
    from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest

    if isinstance(byte_range, RangeByteRequest):
        start, stop = byte_range.start, total if byte_range.end is None else byte_range.end
    elif isinstance(byte_range, OffsetByteRequest):
        start, stop = byte_range.offset, total
    elif isinstance(byte_range, SuffixByteRequest):
        start, stop = total - byte_range.suffix, total
    else:
        raise TypeError(f"Unsupported byte range request: {byte_range!r}")
    return max(0, min(start, total)), max(0, min(stop, total))


def _apply_byte_range(data: bytes, byte_range: ByteRequest) -> bytes:
    """Apply a ByteRequest to raw bytes."""
    start, stop = _range_bounds(byte_range, len(data))
    return data[start:stop]


def _array_path(key: str) -> str | None:
    """The path of the array a chunk key belongs to, or None for any other key."""
    if key == "c":
        return ""
    if key.endswith("/c"):
        return key[:-2]
    head, sep, tail = key.rpartition("/c/")
    if not sep and key.startswith("c/"):
        head, sep, tail = "", "c/", key[2:]
    if sep and tail and all(part.isdigit() for part in tail.split("/")):
        return head
    return None


class Selection:
    """Which bytes of every record in the file belong to an array.

    A reference is read as consecutive records of record_size bytes. From each one,
    the [start, stop) ranges in keep are taken and joined in the order listed.
    See RfsBuilder.add_selection.
    """

    def __init__(self, record_size: int, keep: list[list[int]]) -> None:
        self.record_size = int(record_size)
        self.keep = [(int(a), int(b)) for a, b in keep]
        if self.record_size <= 0 or not self.keep or any(not 0 <= a < b <= self.record_size for a, b in self.keep):
            raise ValueError(f"Invalid selection: record_size {record_size}, keep {keep}")
        self.kept = sum(b - a for a, b in self.keep)

    def selected_size(self, source_size: int) -> int:
        """The size of what is kept from source_size bytes of the file."""
        if source_size % self.record_size:
            raise ValueError(
                f"A reference of {source_size} bytes is not a whole number of {self.record_size} byte records"
            )
        return source_size // self.record_size * self.kept

    def apply(self, data: bytes) -> bytes:
        """Keep the selected bytes of every record in data."""
        self.selected_size(len(data))
        records = np.frombuffer(data, dtype=np.uint8).reshape(-1, self.record_size)
        if len(self.keep) == 1:
            start, stop = self.keep[0]
            return records[:, start:stop].tobytes()
        return np.concatenate([records[:, a:b] for a, b in self.keep], axis=1).tobytes()

    def source_range(self, start: int, stop: int) -> tuple[int, int, int]:
        """Where bytes [start, stop) of the selected data are in the file.

        Returns the offset and length of the whole records that hold them, and
        how many selected bytes to skip at the front of what those records give.
        """
        first = start // self.kept
        last = -(-stop // self.kept)
        return first * self.record_size, (last - first) * self.record_size, start - first * self.kept
