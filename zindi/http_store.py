"""A minimal read-only zarr v3 Store over plain HTTP.

Used to read the chunk index arrays of an RFS directory hosted on a web
server or S3 bucket. It reuses zindi's URL resolution and retry logic instead
of requiring aiohttp for fsspec's HTTP filesystem.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype

from .rfs_store import _apply_byte_range
from .url_resolver import resolve_url


class HttpStore(Store):
    """Read-only zarr v3 Store rooted at a base URL. Missing keys (404/403) read as None."""

    def __init__(self, base_url: str, *, session: requests.Session | None = None) -> None:
        super().__init__(read_only=True)
        self.base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._executor = ThreadPoolExecutor(max_workers=8)

    def __eq__(self, value: object) -> bool:
        return isinstance(value, HttpStore) and value.base_url == self.base_url

    @property
    def supports_writes(self) -> bool:  # type: ignore[override]
        return False

    @property
    def supports_deletes(self) -> bool:  # type: ignore[override]
        return False

    @property
    def supports_listing(self) -> bool:  # type: ignore[override]
        return False

    def _fetch(self, key: str) -> bytes | None:
        response = self._session.get(resolve_url(f"{self.base_url}/{key}"))
        if response.status_code in (403, 404):
            return None
        response.raise_for_status()
        return response.content

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if prototype is None:
            prototype = default_buffer_prototype()
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(self._executor, self._fetch, key)
        if data is None:
            return None
        if byte_range is not None:
            data = _apply_byte_range(data, byte_range)
        return prototype.buffer.from_bytes(data)

    async def get_partial_values(self, prototype: BufferPrototype, key_ranges: Any) -> list[Buffer | None]:
        return list(await asyncio.gather(*(self.get(k, prototype, r) for k, r in key_ranges)))

    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("HttpStore is read-only")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("HttpStore is read-only")

    async def list(self) -> AsyncIterator[str]:
        raise NotImplementedError("HttpStore does not support listing")
        yield ""

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError("HttpStore does not support listing")
        yield ""

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError("HttpStore does not support listing")
        yield ""
