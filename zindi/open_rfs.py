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
    merge_gap: int = 256 * 1024,
    max_merge_size: int = 50 * 1024 * 1024,
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
        Maximum gap in bytes between two HTTP range requests before they
        are fetched separately. Nearby ranges are merged into a single
        request. Default 256 KB.
    max_merge_size : int
        Maximum size in bytes for a single merged HTTP request. Default 50 MB.

    Returns
    -------
    zarr.Group
        A read-only zarr v3 Group backed by the reference file system.
    """
    if isinstance(rfs, str):
        rfs = load_rfs(rfs)

    assert isinstance(rfs, dict)

    store = RfsStore(
        rfs, local_cache=local_cache, merge_gap=merge_gap, max_merge_size=max_merge_size
    )
    return zarr.open_group(store, mode="r", zarr_format=3)


def load_rfs(location: str) -> dict:
    """Load an RFS dict from a JSON file or RFS directory, local or remote.

    For a directory, chunk indexes are not read here. Each "chunk_indexes"
    entry gets an "index" callable that opens the index array on first use.
    """
    is_url = location.startswith(("http://", "https://"))
    if location.endswith(".json"):
        return _read_json(location, is_url)

    base = location.rstrip("/")
    rfs = _read_json(f"{base}/refs.json", is_url)
    chunk_indexes = rfs.get("chunk_indexes")
    if chunk_indexes:
        if is_url:
            from .http_store import HttpStore

            index_store: Any = HttpStore(f"{base}/index")
        else:
            index_store = zarr.storage.LocalStore(os.path.join(base, "index"), read_only=True)
        for path, entry in chunk_indexes.items():
            entry["index"] = lambda p=path: zarr.open_array(index_store, path=p, mode="r")
    return rfs


def _read_json(location: str, is_url: bool) -> dict:
    if is_url:
        from .url_resolver import resolve_url

        response = requests.get(resolve_url(location))
        response.raise_for_status()
        return response.json()
    with open(location) as f:
        return json.load(f)
