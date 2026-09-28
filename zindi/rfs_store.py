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
from .url_resolver import resolve_url


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
        merge_gap: int = 256 * 1024,
        max_merge_size: int = 50 * 1024 * 1024,
    ) -> None:
        """
        Parameters
        ----------
        rfs : dict
            Reference file system dict with "refs" key, and optional
            "templates" and "chunk_indexes".
        local_cache : LocalCache or None
            Optional local cache for persisting remote chunk data on disk.
        merge_gap : int
            Maximum gap in bytes between two ranges before they are fetched
            separately. Ranges within this distance are merged into a single
            HTTP request. Default 256 KB.
        max_merge_size : int
            Maximum size in bytes for a single merged HTTP request. Merged
            ranges that would exceed this are split. Default 50 MB.
        """
        super().__init__(read_only=True)
        if "refs" not in rfs:
            raise ValueError("rfs must contain a 'refs' key")
        self.rfs = rfs
        self._local_cache = local_cache
        self._merge_gap = merge_gap
        self._max_merge_size = max_merge_size
        self._executor = ThreadPoolExecutor(max_workers=32)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "Mozilla/5.0"
        self._indexes = {
            path: ChunkIndex(entry["url"], entry["index"])
            for path, entry in rfs.get("chunk_indexes", {}).items()
        }
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
        data = await loop.run_in_executor(self._executor, self._get_bytes, key)
        if data is None:
            return None
        if byte_range is not None:
            data = _apply_byte_range(data, byte_range)
        return prototype.buffer.from_bytes(data)

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
                    padded_size = self._get_padded_size(key, cached_data)
                    if padded_size is not None:
                        cached_data = cached_data + b"\0" * (padded_size - len(cached_data))
                    results.append((item_idx, prototype.buffer.from_bytes(cached_data)))
                else:
                    uncached.append((item_idx, offset, length, key))

            if not uncached:
                continue

            # Re-compute merged range for uncached items only
            uncached_start = min(r[1] for r in uncached)
            uncached_end = max(r[1] + r[2] for r in uncached)

            # Fetch the merged range
            raw = _read_bytes_from_url(url, uncached_start, uncached_end - uncached_start, session=self._session)

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

                # Apply padding
                padded_size = self._get_padded_size(key, chunk_data)
                if padded_size is not None:
                    chunk_data = chunk_data + b"\0" * (padded_size - len(chunk_data))

                results.append((item_idx, prototype.buffer.from_bytes(chunk_data)))

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
        for name in sorted(self._get_children().get(prefix, ())):
            yield name

    def _get_children(self) -> dict[str, set[str]]:
        """Map each directory prefix to its immediate children, built once."""
        if self._children is None:
            children: dict[str, set[str]] = {}
            keys = list(self.rfs["refs"]) + [f"{path}/c" for path in self._indexes]
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

        Chunks of indexed arrays are looked up in their chunk index.
        """
        ref = self.rfs["refs"].get(key)
        if ref is not None or not self._indexes:
            return ref
        path, sep, coords_str = key.rpartition("/c/")
        index = self._indexes.get(path) if sep else None
        if index is None:
            return None
        try:
            coords = tuple(int(c) for c in coords_str.split("/"))
        except ValueError:
            return None
        hit = index.lookup(coords)
        if hit is None:
            return None
        return [index.url, hit[0], hit[1]]

    def _expand_templates(self, url_or_path: str) -> str:
        if "{{" in url_or_path and "}}" in url_or_path and "templates" in self.rfs:
            for tkey, tval in self.rfs["templates"].items():
                url_or_path = url_or_path.replace("{{" + tkey + "}}", tval)
        return url_or_path

    def _get_bytes(self, key: str) -> bytes | None:
        """Resolve a key to bytes, handling all reference types."""
        x = self._resolve(key)
        if x is None:
            return None

        if isinstance(x, str):
            if x.startswith("base64:"):
                return base64.b64decode(x[len("base64:"):])
            else:
                return x.encode("utf-8")
        elif isinstance(x, dict):
            return json.dumps(x).encode("utf-8")
        elif isinstance(x, list):
            if len(x) != 3:
                raise ValueError(f"Reference list for {key} must have 3 elements")
            url_or_path, offset, length = self._expand_templates(x[0]), x[1], x[2]

            is_url = url_or_path.startswith("http://") or url_or_path.startswith("https://")

            # Check local cache for remote chunks
            if self._local_cache is not None and is_url:
                cached = self._local_cache.get_remote_chunk(
                    url=url_or_path, offset=offset, size=length
                )
                if cached is not None:
                    padded_size = self._get_padded_size(key, cached)
                    if padded_size is not None:
                        cached = cached + b"\0" * (padded_size - len(cached))
                    return cached

            data = _read_bytes_from_url_or_path(url_or_path, offset, length, session=self._session)

            # Store in local cache
            if self._local_cache is not None and is_url:
                from .local_cache import ChunkTooLargeError

                try:
                    self._local_cache.put_remote_chunk(
                        url=url_or_path, offset=offset, size=length, data=data
                    )
                except ChunkTooLargeError:
                    pass  # chunk exceeds SQLite blob limit, skip caching

            # Pad if this is a final chunk in a contiguous dataset
            padded_size = self._get_padded_size(key, data)
            if padded_size is not None:
                data = data + b"\0" * (padded_size - len(data))

            return data
        else:
            raise ValueError(f"Unexpected reference type for {key}: {type(x)}")

    def _get_padded_size(self, key: str, data: bytes) -> int | None:
        """Check if a chunk needs padding (final chunk in contiguous dataset).

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

        chunk_shape = meta.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape")
        data_type = meta.get("data_type")
        if chunk_shape is None or data_type is None:
            return None

        if isinstance(data_type, dict):
            if data_type.get("name") == "structured":
                fields = data_type["configuration"]["fields"]
                dtype = np.dtype([
                    (f[0], _zarr_field_type_to_numpy(f[1])) for f in fields
                ])
            else:
                return None
        elif isinstance(data_type, str):
            dtype = np.dtype(data_type)
        else:
            return None
        if dtype.kind not in ("i", "u", "f", "V"):
            return None

        expected_size = int(np.prod(chunk_shape)) * dtype.itemsize
        if len(data) < expected_size:
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
    url_or_path: str, offset: int, length: int, *, session: requests.Session | None = None
) -> bytes:
    """Read a byte range from a URL or local file path."""
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return _read_bytes_from_url(url_or_path, offset, length, session=session)
    else:
        with open(url_or_path, "rb") as f:
            f.seek(offset)
            return f.read(length)


def _read_bytes_from_url(
    url: str, offset: int, length: int, *, session: requests.Session | None = None
) -> bytes:
    """Read a byte range from a URL with retry and DANDI resolution."""
    num_retries = 8
    for try_num in range(num_retries):
        try:
            resolved_url = resolve_url(url)
            range_header = f"bytes={offset}-{offset + length - 1}"
            headers = {"Range": range_header}
            if session is not None:
                response = session.get(resolved_url, headers=headers)
            else:
                headers["User-Agent"] = "Mozilla/5.0"
                response = requests.get(resolved_url, headers=headers)
            response.raise_for_status()
            return response.content
        except Exception as e:
            if try_num == num_retries - 1:
                raise
            delay = 0.1 * 2**try_num
            print(f"Retry {try_num + 1}/{num_retries} for {url} in {delay:.1f}s: {e}")
            time.sleep(delay)
    raise RuntimeError(f"Failed to read from {url}")


def _apply_byte_range(data: bytes, byte_range: ByteRequest) -> bytes:
    """Apply a ByteRequest to raw bytes."""
    from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest

    if isinstance(byte_range, RangeByteRequest):
        end = byte_range.end if byte_range.end is not None else len(data)
        return data[byte_range.start:end]
    elif isinstance(byte_range, OffsetByteRequest):
        return data[byte_range.offset:]
    elif isinstance(byte_range, SuffixByteRequest):
        return data[-byte_range.suffix:]
    else:
        return data
