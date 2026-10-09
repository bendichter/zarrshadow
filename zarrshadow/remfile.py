"""File-like object for reading a remote file over HTTP.

Optimized for use with h5py: reads small chunks for metadata, then
adaptively increases chunk size for sequential access patterns.

Ported from lindi's LindiRemfile, with one addition: the nodes of a chunk
index are read ahead, in parallel (see _prefetch_chunk_btree).
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from .url_resolver import resolve_url

_DEFAULT_MIN_CHUNK_SIZE = 128 * 1024  # 128 KB
_DEFAULT_MAX_CACHE_SIZE = 1024 * 1024 * 1024  # 1 GB
_DEFAULT_CHUNK_INCREMENT_FACTOR = 1.7
_DEFAULT_MAX_CHUNK_SIZE = 100 * 1024 * 1024  # 100 MB
_NUM_REQUEST_RETRIES = 8
_PREFETCH_WORKERS = 32

# A version 1 B-tree node: signature, node type, level, entries used, and two
# sibling addresses. Type 1 nodes index the chunks of a dataset.
_BTREE_SIGNATURE = b"TREE"
_OFFSET_SIZE = 8
_BTREE_HEADER = 8 + 2 * _OFFSET_SIZE


def parse_chunk_btree_node(node: bytes, file_length: int) -> tuple[int, list[int]] | None:
    """The level and child addresses of a version 1 B-tree node that indexes chunks.

    Returns None for anything else. A key holds the chunk's size, its filter
    mask, and an offset for each dimension of the dataset plus one for the
    element, which is 0 in every key but the last. The number of dimensions is
    not in the node, so each is tried until the keys and children make sense.
    A wrong guess only means reading ahead what is not needed.
    """
    if len(node) < _BTREE_HEADER or node[:4] != _BTREE_SIGNATURE or node[4] != 1:
        return None
    level = node[5]
    entries = int.from_bytes(node[6:8], "little")
    if entries == 0:
        return None
    for dims in range(2, 34):  # HDF5 datasets have at most 32 dimensions
        key = 8 + 8 * dims
        if _BTREE_HEADER + (entries + 1) * key + entries * _OFFSET_SIZE > len(node):
            return None
        children = []
        for i in range(entries):
            at = _BTREE_HEADER + i * (key + _OFFSET_SIZE)
            if int.from_bytes(node[at + key - 8 : at + key], "little") != 0:
                break
            child = int.from_bytes(node[at + key : at + key + _OFFSET_SIZE], "little")
            if not 0 < child < file_length:
                break
            children.append(child)
        else:
            return level, children
    return None


class ZarrShadowRemfile:
    """A file-like object for reading a remote file over HTTP.

    Optimized for reading HDF5 files: starts with small reads for metadata,
    then adaptively increases chunk size when sequential access is detected.

    Parameters
    ----------
    url : str
        URL of the remote file.
    """

    def __init__(self, url: str) -> None:
        if not isinstance(url, str):
            raise TypeError("url must be a string")
        self._url = url
        self._memory_chunks: dict[int, bytes] = {}
        self._memory_chunk_indices: list[int] = []
        self._position = 0
        self._smart_loader_last_chunk_index_accessed = -99
        self._smart_loader_chunk_sequence_length = 1
        self._min_chunk_size = _DEFAULT_MIN_CHUNK_SIZE
        self._max_chunks_in_cache = int(_DEFAULT_MAX_CACHE_SIZE / self._min_chunk_size)
        self._max_chunk_size = _DEFAULT_MAX_CHUNK_SIZE

        # Get file length via aborted GET (works with presigned AWS URLs
        # where HEAD requests may not be supported)
        response = requests.get(resolve_url(self._url), stream=True)
        if response.status_code == 200:
            self.length = int(response.headers["Content-Length"])
        else:
            raise RuntimeError(
                f"Error getting file length: {response.status_code} {response.reason}"
            )
        response.close()

        self.session = requests.Session()
        # Nodes of chunk indexes, read ahead, by address
        self._nodes: dict[int, bytes] = {}
        self._indexes_read_ahead: set[int] = set()
        self._local = threading.local()

    def read(self, size: int | None = None) -> bytes:
        if size is None:
            raise ValueError("size argument is required")
        start = self._position
        node = self._nodes.get(start)
        if node is not None and size <= len(node):
            self._position += size
            return node[:size]
        data = self._read(size)
        if data[:4] == _BTREE_SIGNATURE and start not in self._indexes_read_ahead:
            self._prefetch_chunk_btree(start, data)
        return data

    def _prefetch_chunk_btree(self, address: int, node: bytes) -> None:
        """Read every node under this one, a level at a time, in parallel.

        h5py walks the B-tree of a chunked dataset one node at a time, and the
        nodes of a dataset written in pieces lie throughout the file, so each
        costs a request. For a dataset with hundreds of thousands of chunks
        that is thousands of requests in a row. All nodes have the size of
        this one, which is how much h5py reads of each.
        """
        self._indexes_read_ahead.add(address)
        size = len(node)
        level = [node]
        with ThreadPoolExecutor(_PREFETCH_WORKERS) as pool:
            while level:
                wanted = []
                for parent in level:
                    parsed = parse_chunk_btree_node(parent, self.length)
                    if parsed is None or parsed[0] == 0:
                        continue  # the children of a leaf are chunks of data
                    wanted += [a for a in parsed[1] if a not in self._nodes]
                fetched = list(pool.map(lambda a: self._fetch_node(a, size), wanted))
                level = []
                for child, data in zip(wanted, fetched):
                    self._nodes[child] = data
                    self._indexes_read_ahead.add(child)
                    level.append(data)

    def _fetch_node(self, address: int, size: int) -> bytes:
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._local.session = requests.Session()
        end = min(address + size, self.length) - 1
        return _fetch_bytes(session, resolve_url(self._url), address, end)

    def _read(self, size: int) -> bytes:
        chunk_start_index = self._position // self._min_chunk_size
        chunk_end_index = (self._position + size - 1) // self._min_chunk_size
        loaded_chunks = {}
        for chunk_index in range(chunk_start_index, chunk_end_index + 1):
            loaded_chunks[chunk_index] = self._load_chunk(chunk_index)

        if chunk_end_index == chunk_start_index:
            chunk = loaded_chunks[chunk_start_index]
            chunk_offset = self._position % self._min_chunk_size
            self._position += size
            return chunk[chunk_offset : chunk_offset + size]
        else:
            pieces = []
            for chunk_index in range(chunk_start_index, chunk_end_index + 1):
                chunk = loaded_chunks[chunk_index]
                if chunk_index == chunk_start_index:
                    chunk_offset = self._position % self._min_chunk_size
                    chunk_length = self._min_chunk_size - chunk_offset
                elif chunk_index == chunk_end_index:
                    chunk_offset = 0
                    chunk_length = size - sum(len(p) for p in pieces)
                else:
                    chunk_offset = 0
                    chunk_length = self._min_chunk_size
                pieces.append(chunk[chunk_offset : chunk_offset + chunk_length])
            ret = b"".join(pieces)
            self._position += size

            # Clean up cache if it's too large
            if len(self._memory_chunk_indices) > self._max_chunks_in_cache:
                cutoff = int(self._max_chunks_in_cache * 0.5)
                for ci in self._memory_chunk_indices[:cutoff]:
                    self._memory_chunks.pop(ci, None)
                self._memory_chunk_indices = self._memory_chunk_indices[cutoff:]

            return ret

    def _load_chunk(self, chunk_index: int) -> bytes:
        if chunk_index in self._memory_chunks:
            self._smart_loader_last_chunk_index_accessed = chunk_index
            return self._memory_chunks[chunk_index]

        # Smart loader: increase fetch size for sequential access
        if chunk_index == self._smart_loader_last_chunk_index_accessed + 1:
            self._smart_loader_chunk_sequence_length = round(
                self._smart_loader_chunk_sequence_length * _DEFAULT_CHUNK_INCREMENT_FACTOR + 0.5
            )
            max_seq = int(self._max_chunk_size / self._min_chunk_size)
            if self._smart_loader_chunk_sequence_length > max_seq:
                self._smart_loader_chunk_sequence_length = max_seq
            # Don't overshoot into already-loaded chunks
            for j in range(1, self._smart_loader_chunk_sequence_length):
                if chunk_index + j in self._memory_chunks:
                    self._smart_loader_chunk_sequence_length = j
                    break
        else:
            self._smart_loader_chunk_sequence_length = round(
                self._smart_loader_chunk_sequence_length / _DEFAULT_CHUNK_INCREMENT_FACTOR + 0.5
            )

        data_start = chunk_index * self._min_chunk_size
        data_end = data_start + self._min_chunk_size * self._smart_loader_chunk_sequence_length - 1
        if data_end >= self.length:
            data_end = self.length - 1

        x = _fetch_bytes(
            self.session, resolve_url(self._url), data_start, data_end
        )
        if not x:
            raise RuntimeError(f"Error loading chunk {chunk_index} from {self._url}")

        # Split the response into chunks and cache them
        if self._smart_loader_chunk_sequence_length == 1:
            self._memory_chunks[chunk_index] = x
            self._memory_chunk_indices.append(chunk_index)
        else:
            for i in range(self._smart_loader_chunk_sequence_length):
                if i * self._min_chunk_size >= len(x):
                    break
                self._memory_chunks[chunk_index + i] = x[
                    i * self._min_chunk_size : (i + 1) * self._min_chunk_size
                ]
                self._memory_chunk_indices.append(chunk_index + i)

        self._smart_loader_last_chunk_index_accessed = (
            chunk_index + self._smart_loader_chunk_sequence_length - 1
        )
        return x[: self._min_chunk_size]

    def seek(self, offset: int, whence: int = 0) -> None:
        if whence == 0:
            self._position = offset
        elif whence == 1:
            self._position += offset
        elif whence == 2:
            self._position = self.length + offset
        else:
            raise ValueError("whence must be 0, 1, or 2")

    def tell(self) -> int:
        return self._position

    def close(self) -> None:
        pass


def _fetch_bytes(
    session: requests.Session,
    url: str,
    start_byte: int,
    end_byte: int,
) -> bytes:
    """Fetch a byte range from a URL with retries."""
    for try_num in range(_NUM_REQUEST_RETRIES + 1):
        try:
            range_header = f"bytes={start_byte}-{end_byte}"
            response = session.get(url, headers={"Range": range_header})
            response.raise_for_status()
            return response.content
        except Exception as e:
            if try_num == _NUM_REQUEST_RETRIES:
                raise
            delay = 0.1 * 2**try_num
            print(f"Retry {try_num + 1} for {url}: {e}")
            time.sleep(delay)
    raise RuntimeError(f"Failed to fetch bytes from {url}")
