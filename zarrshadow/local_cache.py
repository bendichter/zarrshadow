"""Local SQLite-backed cache for remote chunk data.

Persists fetched byte ranges on disk so repeated reads of the same
remote chunks are served from the local cache instead of re-fetching
over HTTP. Supports optional LRU eviction via max_size_bytes.
"""

from __future__ import annotations

import os
import sqlite3
import time


class ChunkTooLargeError(Exception):
    pass


class LocalCache:
    """Persistent local cache for remote chunk data.

    Parameters
    ----------
    cache_dir : str or None
        Directory to store the cache database. Defaults to ``~/.zarrshadow/cache``.
    max_size_bytes : int or None
        Maximum total size of cached data in bytes. When exceeded, the
        least-recently-accessed chunks are evicted. None means no limit.
    """

    def __init__(
        self, *, cache_dir: str | None = None, max_size_bytes: int | None = None
    ):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.zarrshadow/cache")
        self._cache_dir = cache_dir
        os.makedirs(self._cache_dir, exist_ok=True)
        self._sqlite_client = _LocalCacheSQLiteClient(
            db_fname=os.path.join(self._cache_dir, "zarrshadow_cache.db"),
            max_size_bytes=max_size_bytes,
        )

    def get_remote_chunk(self, *, url: str, offset: int, size: int) -> bytes | None:
        """Retrieve a cached chunk, or None if not cached."""
        return self._sqlite_client.get_remote_chunk(url=url, offset=offset, size=size)

    def put_remote_chunk(self, *, url: str, offset: int, size: int, data: bytes) -> None:
        """Store a chunk in the cache.

        Raises
        ------
        ChunkTooLargeError
            If the chunk is >= 900 MB (SQLite BLOB limit).
        """
        if len(data) != size:
            raise ValueError("data size does not match size")
        self._sqlite_client.put_remote_chunk(url=url, offset=offset, size=size, data=data)


class _LocalCacheSQLiteClient:
    """SQLite backend for LocalCache with optional LRU eviction."""

    def __init__(self, *, db_fname: str, max_size_bytes: int | None = None):
        self._max_size_bytes = max_size_bytes
        self._conn = sqlite3.connect(db_fname, check_same_thread=False)
        self._lock = __import__("threading").Lock()
        self._cursor = self._conn.cursor()
        self._cursor.execute("PRAGMA journal_mode=WAL")
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS remote_chunks (
                url TEXT,
                offset INTEGER,
                size INTEGER,
                data BLOB,
                last_accessed REAL,
                PRIMARY KEY (url, offset, size)
            )
            """
        )
        self._conn.commit()

    def get_remote_chunk(self, *, url: str, offset: int, size: int) -> bytes | None:
        with self._lock:
            self._cursor.execute(
                "SELECT data FROM remote_chunks WHERE url = ? AND offset = ? AND size = ?",
                (url, offset, size),
            )
            row = self._cursor.fetchone()
            if row is None:
                return None
            # Update last_accessed timestamp
            self._cursor.execute(
                "UPDATE remote_chunks SET last_accessed = ? WHERE url = ? AND offset = ? AND size = ?",
                (time.time(), url, offset, size),
            )
            self._conn.commit()
            return row[0]

    def put_remote_chunk(self, *, url: str, offset: int, size: int, data: bytes) -> None:
        if size >= 900_000_000:
            raise ChunkTooLargeError("Cannot store blobs larger than 900 MB in LocalCache")
        with self._lock:
            self._cursor.execute(
                "INSERT OR REPLACE INTO remote_chunks (url, offset, size, data, last_accessed) "
                "VALUES (?, ?, ?, ?, ?)",
                (url, offset, size, data, time.time()),
            )
            self._conn.commit()
            if self._max_size_bytes is not None:
                self._evict()

    def _evict(self) -> None:
        """Delete least-recently-accessed chunks until total size is within limit.

        Caller must hold self._lock.
        """
        self._cursor.execute("SELECT SUM(size) FROM remote_chunks")
        total = self._cursor.fetchone()[0] or 0
        if total <= self._max_size_bytes:
            return
        self._cursor.execute(
            "SELECT url, offset, size FROM remote_chunks ORDER BY last_accessed ASC"
        )
        rows = self._cursor.fetchall()
        for url, offset, size in rows:
            if total <= self._max_size_bytes:
                break
            self._cursor.execute(
                "DELETE FROM remote_chunks WHERE url = ? AND offset = ? AND size = ?",
                (url, offset, size),
            )
            total -= size
        self._conn.commit()
