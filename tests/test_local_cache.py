"""Tests for local chunk caching."""

import tempfile

import h5py
import numpy as np
import pytest

from zarrshadow import LocalCache, generate_rfs, open_rfs
from zarrshadow.local_cache import ChunkTooLargeError


class TestLocalCacheDirect:
    """Test LocalCache get/put operations directly."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = LocalCache(cache_dir=self.tmpdir)

    def test_put_and_get(self):
        """Stored chunk can be retrieved."""
        data = b"hello world"
        self.cache.put_remote_chunk(url="http://example.com/f.h5", offset=100, size=len(data), data=data)
        result = self.cache.get_remote_chunk(url="http://example.com/f.h5", offset=100, size=len(data))
        assert result == data

    def test_cache_miss(self):
        """Uncached chunk returns None."""
        result = self.cache.get_remote_chunk(url="http://example.com/f.h5", offset=0, size=10)
        assert result is None

    def test_different_keys(self):
        """Different (url, offset, size) tuples are independent."""
        self.cache.put_remote_chunk(url="http://a.com/f.h5", offset=0, size=3, data=b"aaa")
        self.cache.put_remote_chunk(url="http://b.com/f.h5", offset=0, size=3, data=b"bbb")
        assert self.cache.get_remote_chunk(url="http://a.com/f.h5", offset=0, size=3) == b"aaa"
        assert self.cache.get_remote_chunk(url="http://b.com/f.h5", offset=0, size=3) == b"bbb"

    def test_overwrite(self):
        """Storing to the same key overwrites."""
        self.cache.put_remote_chunk(url="http://x.com/f.h5", offset=0, size=3, data=b"old")
        self.cache.put_remote_chunk(url="http://x.com/f.h5", offset=0, size=3, data=b"new")
        assert self.cache.get_remote_chunk(url="http://x.com/f.h5", offset=0, size=3) == b"new"

    def test_size_mismatch_raises(self):
        """Mismatched data length raises ValueError."""
        with pytest.raises(ValueError, match="data size does not match"):
            self.cache.put_remote_chunk(url="http://x.com/f.h5", offset=0, size=5, data=b"abc")

    def test_chunk_too_large(self):
        """Chunks >= 900 MB raise ChunkTooLargeError."""
        with pytest.raises(ChunkTooLargeError):
            self.cache.put_remote_chunk(
                url="http://x.com/f.h5", offset=0, size=900_000_000, data=b"\x00" * 900_000_000
            )


class TestLocalCacheEviction:
    """Test LRU eviction with max_size_bytes."""

    def test_eviction_removes_oldest(self):
        """When max_size_bytes is exceeded, least-recently-accessed chunks are evicted."""
        tmpdir = tempfile.mkdtemp()
        # Max 150 bytes — each chunk is 100 bytes, so only ~1 fits
        cache = LocalCache(cache_dir=tmpdir, max_size_bytes=150)

        chunk_a = b"a" * 100
        chunk_b = b"b" * 100

        cache.put_remote_chunk(url="http://x.com/f.h5", offset=0, size=100, data=chunk_a)
        # chunk_a should exist
        assert cache.get_remote_chunk(url="http://x.com/f.h5", offset=0, size=100) == chunk_a

        cache.put_remote_chunk(url="http://x.com/f.h5", offset=100, size=100, data=chunk_b)
        # chunk_b should exist; chunk_a should be evicted (total would be 200 > 150)
        assert cache.get_remote_chunk(url="http://x.com/f.h5", offset=100, size=100) == chunk_b
        assert cache.get_remote_chunk(url="http://x.com/f.h5", offset=0, size=100) is None

    def test_lru_order(self):
        """Accessing a chunk updates its timestamp, protecting it from eviction."""
        tmpdir = tempfile.mkdtemp()
        # Max 250 bytes — fits two 100-byte chunks but not three
        cache = LocalCache(cache_dir=tmpdir, max_size_bytes=250)

        cache.put_remote_chunk(url="http://x.com/f.h5", offset=0, size=100, data=b"a" * 100)
        cache.put_remote_chunk(url="http://x.com/f.h5", offset=100, size=100, data=b"b" * 100)

        # Access chunk_a to make it more recent than chunk_b
        cache.get_remote_chunk(url="http://x.com/f.h5", offset=0, size=100)

        # Insert chunk_c — should evict chunk_b (least recently accessed)
        cache.put_remote_chunk(url="http://x.com/f.h5", offset=200, size=100, data=b"c" * 100)

        assert cache.get_remote_chunk(url="http://x.com/f.h5", offset=0, size=100) == b"a" * 100
        assert cache.get_remote_chunk(url="http://x.com/f.h5", offset=100, size=100) is None
        assert cache.get_remote_chunk(url="http://x.com/f.h5", offset=200, size=100) == b"c" * 100

    def test_no_limit(self):
        """Without max_size_bytes, no eviction occurs."""
        tmpdir = tempfile.mkdtemp()
        cache = LocalCache(cache_dir=tmpdir)

        for i in range(10):
            cache.put_remote_chunk(url="http://x.com/f.h5", offset=i * 100, size=100, data=b"x" * 100)

        # All should still be present
        for i in range(10):
            assert cache.get_remote_chunk(url="http://x.com/f.h5", offset=i * 100, size=100) is not None


class TestLocalCacheIntegration:
    """Test cache integration with RFS read path."""

    def test_cache_with_local_hdf5(self):
        """Cache works end-to-end with a local HDF5 file.

        Local file reads are not cached (only remote URLs), but the
        cache parameter should not break anything.
        """
        tmpdir = tempfile.mkdtemp()
        h5_path = f"{tmpdir}/test.h5"

        data = np.arange(2000, dtype=np.float64)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("arr", data=data, chunks=(500,))

        rfs = generate_rfs(h5_path)
        cache = LocalCache(cache_dir=tmpdir)
        root = open_rfs(rfs, local_cache=cache)
        result = root["arr"][:]
        np.testing.assert_array_equal(result, data)
