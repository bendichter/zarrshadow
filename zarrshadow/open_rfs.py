"""Open a zarr v3 reference file system as a zarr Group.

Given an RFS dict, a path or URL to an RFS JSON file, or a path or URL to an
RFS directory, this creates an RfsStore and opens it as a zarr v3 group
hierarchy.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests
import zarr

from .rfs_store import RfsStore


def open_rfs(
    rfs: dict | str,
    *,
    local_cache: Any = None,
    merge_gap: int = 32 * 1024,
    max_merge_size: int = 2**20,
    merge_below: int = 64 * 1024,
    validate_sources: bool = True,
) -> zarr.Group:
    """Open a reference file system as a zarr v3 Group.

    Parameters
    ----------
    rfs : dict or str
        An RFS dict (with "refs" and "version" keys), or a path or URL to an
        RFS JSON file (ending in ".json") or RFS directory (as written by
        ``write_rfs``).
    local_cache : LocalCache or None
        Optional local cache for persisting remote chunk data on disk.
    merge_gap : int
        Chunks of a remote file that are asked for at the same time are
        fetched in one request when the gap between them is at most this
        many bytes. Default 32 KiB.
    max_merge_size : int
        The most bytes one merged request may ask for. 0 turns merging off.
        Default 1 MiB.
    merge_below : int
        Only chunks of at most this many bytes are merged. Default 64 KiB.
    validate_sources : bool
        Raise SourceChangedError if a file the references point into has
        changed since they were generated. Default True.

    Returns
    -------
    zarr.Group
        A read-only zarr v3 Group backed by the reference file system.
    """
    if isinstance(rfs, str):
        rfs = load_rfs(rfs)

    assert isinstance(rfs, dict)

    store = RfsStore(
        rfs,
        local_cache=local_cache,
        merge_gap=merge_gap,
        max_merge_size=max_merge_size,
        merge_below=merge_below,
        validate_sources=validate_sources,
    )
    return zarr.open_group(store, mode="r", zarr_format=3)


def load_rfs(location: str) -> dict:
    """Load an RFS dict from a JSON file or RFS directory, local or remote.

    For a directory, chunk indexes are not read here. Each "indexes" entry's
    "index" path, relative to refs.json, is replaced by a callable that opens the
    index array on first use.
    """
    is_url = location.startswith(("http://", "https://"))
    if location.endswith(".json"):
        return _read_json(location, is_url)

    base = location.rstrip("/")
    rfs = _read_json(f"{base}/refs.json", is_url)
    indexes = rfs.get("indexes")
    if indexes:
        if is_url:
            from .http_store import HttpStore

            index_store: Any = HttpStore(base)
        else:
            index_store = zarr.storage.LocalStore(base, read_only=True)
        for path, entry in indexes.items():
            index_path = _check_relative_path(entry["index"])
            entry["index"] = lambda p=index_path: zarr.open_array(index_store, path=p, mode="r")
    return rfs


def _check_relative_path(path: str) -> str:
    """Refuse index paths that would leave the directory holding refs.json."""
    parts = path.split("/")
    if path.startswith("/") or ".." in parts or "://" in path:
        raise ValueError(f"index path must be relative to refs.json: {path!r}")
    return path


def _read_json(location: str, is_url: bool) -> dict:
    if is_url:
        from .url_resolver import resolve_url

        response = requests.get(resolve_url(location))
        response.raise_for_status()
        return response.json()
    with open(location) as f:
        return json.load(f)
