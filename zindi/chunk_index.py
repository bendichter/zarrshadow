"""Chunk indexes for arrays with many chunks.

An array with more chunks than a threshold does not list its chunks in the
RFS refs. Its byte ranges are kept in an index array of shape
``(*chunk_grid, 2)`` and dtype uint64 holding ``(offset, nbytes)`` for each
chunk, the same layout as the zarr v3 sharding index. A chunk that is not
allocated in the HDF5 file holds ``MISSING`` in both fields.

In memory (as returned by ``generate_rfs``) an index is a numpy array. On disk
(``write_rfs`` to a directory) each index is a zarr v3 array stored under
``index/<array path>``, split into blocks of about ``INDEX_BLOCK_ENTRIES``
chunks and compressed with blosc zstd and byte shuffle. Readers load only the
blocks that cover the chunks they request.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

MISSING = np.iinfo(np.uint64).max
INDEX_BLOCK_ENTRIES = 65536


def index_block_shape(grid_shape: tuple[int, ...] | list[int]) -> list[int]:
    """Choose the block shape (over the chunk grid) for an on-disk index.

    Blocks are filled from the last grid dimension backward, so a block holds
    about INDEX_BLOCK_ENTRIES consecutive chunks in C order.
    """
    block = [1] * len(grid_shape)
    remaining = INDEX_BLOCK_ENTRIES
    for i in reversed(range(len(grid_shape))):
        block[i] = max(1, min(grid_shape[i], remaining))
        remaining //= block[i]
        if remaining <= 1:
            break
    return block


class ChunkIndex:
    """Look up chunk byte ranges for one array.

    Parameters
    ----------
    url : str
        URL or path of the file that holds every chunk of the array.
    source : numpy.ndarray, zarr.Array, or callable
        The index array, or a zero-argument callable that opens it. A callable
        is called on first lookup, so opening an RFS does not read any index.
    max_cached_blocks : int
        Number of index blocks to keep in memory when the source is a zarr
        array.
    """

    def __init__(self, url: str, source: Any, *, max_cached_blocks: int = 64) -> None:
        self.url = url
        self._source = source
        self._max_cached_blocks = max_cached_blocks
        self._blocks: OrderedDict[tuple[int, ...], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def array(self) -> Any:
        """The index as a numpy array or zarr array, opening it if needed."""
        with self._lock:
            if callable(self._source):
                self._source = self._source()
            return self._source

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape[:-1])

    def lookup(self, coords: tuple[int, ...]) -> tuple[int, int] | None:
        """Return (offset, nbytes) for the chunk at coords, or None if absent."""
        arr = self.array
        if len(coords) != arr.ndim - 1 or any(
            not 0 <= c < n for c, n in zip(coords, arr.shape[:-1])
        ):
            return None
        if isinstance(arr, np.ndarray):
            offset, nbytes = arr[coords]
        else:
            block_shape = arr.chunks[:-1]
            block_id = tuple(c // b for c, b in zip(coords, block_shape))
            block = self._get_block(arr, block_id, block_shape)
            offset, nbytes = block[tuple(c % b for c, b in zip(coords, block_shape))]
        if offset == MISSING:
            return None
        return int(offset), int(nbytes)

    def _get_block(
        self, arr: Any, block_id: tuple[int, ...], block_shape: tuple[int, ...]
    ) -> np.ndarray:
        with self._lock:
            block = self._blocks.get(block_id)
            if block is not None:
                self._blocks.move_to_end(block_id)
                return block
        selection = tuple(slice(i * b, (i + 1) * b) for i, b in zip(block_id, block_shape))
        block = arr[selection]
        with self._lock:
            self._blocks[block_id] = block
            while len(self._blocks) > self._max_cached_blocks:
                self._blocks.popitem(last=False)
        return block

    def iter_chunks(self) -> Iterator[tuple[tuple[int, ...], int, int]]:
        """Yield (coords, offset, nbytes) for every allocated chunk."""
        arr = np.asarray(self.array[...])
        present = np.argwhere(arr[..., 0] != MISSING)
        for coords in present:
            coords = tuple(int(c) for c in coords)
            offset, nbytes = arr[coords]
            yield coords, int(offset), int(nbytes)


def build_index(
    grid_shape: tuple[int, ...],
    for_each_chunk: Callable[[Callable[[tuple[int, ...], int, int], None]], None],
) -> np.ndarray:
    """Build an in-memory index by calling for_each_chunk with a setter."""
    index = np.full((*grid_shape, 2), MISSING, dtype=np.uint64)

    def set_chunk(coords: tuple[int, ...], offset: int, nbytes: int) -> None:
        index[coords] = (offset, nbytes)

    for_each_chunk(set_chunk)
    return index
