"""Small chunks of a remote file that are asked for together are fetched together."""

import asyncio
import functools
import http.server
import re
import threading

import numpy as np
import pytest
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

import zarrshadow.rfs_store as rfs_store
from zarrshadow import RfsBuilder, RfsStore, VirtualArray, open_rfs

RECORD, SAMPLES, HEADER = 1044, 512, 20  # a record of a Neuralynx file: 20 bytes, then 512 int16 samples
N_RECORDS = 120


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static files with single byte-range support, which notes each range it is asked for."""

    ranges: list[tuple[int, int]] = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        match = re.match(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if not match:
            return super().do_GET()
        data = open(self.translate_path(self.path), "rb").read()
        start, end = int(match[1]), min(int(match[2]), len(data) - 1)
        self.ranges.append((start, end + 1))
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.wfile.write(data[start : end + 1])


@pytest.fixture
def served(tmp_path):
    """A file of records on a local server: its URL, its samples, and the ranges the server was asked for."""
    x = np.random.default_rng(18).integers(-3000, 3000, (N_RECORDS, SAMPLES)).astype("<i2")
    records = np.zeros(N_RECORDS, dtype=[("header", "u1", HEADER), ("samples", "<i2", SAMPLES)])
    records["samples"] = x
    (tmp_path / "channel.ncs").write_bytes(records.tobytes())
    _RangeHandler.ranges = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_RangeHandler, directory=str(tmp_path)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/channel.ncs", x.ravel(), _RangeHandler.ranges
    finally:
        server.shutdown()


def _rfs(url):
    """References to the samples of every record, each record a chunk."""
    builder = RfsBuilder()
    builder.add_group("")
    VirtualArray.blocks(
        url, shape=(N_RECORDS * SAMPLES,), chunk_shape=(SAMPLES,), dtype="<i2", record_size=RECORD, offset=HEADER
    ).add_to(builder, "data")
    return builder.build(record_sources=False)


def _gets(store, chunks):
    """Ask for several chunks at the same time, as zarr does for a selection."""

    async def together():
        return await asyncio.gather(*(store.get(f"data/c/{i}", default_buffer_prototype()) for i in chunks))

    return [buffer.to_bytes() for buffer in sync(together())]


def _chunk_range(i):
    return (HEADER + RECORD * i, HEADER + RECORD * i + 2 * SAMPLES)


def test_reading_an_array(served):
    """zarr reads the chunks of a selection together, as many at a time as its configuration allows."""
    url, x, ranges = served
    data = open_rfs(_rfs(url))["data"]
    with zarr.config.set({"async.concurrency": 200}):
        np.testing.assert_array_equal(data[...], x)
    # 120 chunks of 1024 bytes, 20 bytes apart: one request
    assert ranges == [(HEADER, _chunk_range(N_RECORDS - 1)[1])]

    ranges.clear()
    with zarr.config.set({"async.concurrency": 10}):
        np.testing.assert_array_equal(data[...], x)
    assert len(ranges) == 12

    ranges.clear()
    unmerged = open_rfs(_rfs(url), max_merge_size=0)["data"]
    np.testing.assert_array_equal(unmerged[SAMPLES : 6 * SAMPLES], x[SAMPLES : 6 * SAMPLES])
    assert sorted(ranges) == [_chunk_range(i) for i in range(1, 6)]


def test_which_reads_are_merged(served):
    url, x, ranges = served
    rfs = _rfs(url)
    chunks = [0, 1, 4, 5, 40]
    expected = [x[i * SAMPLES : (i + 1) * SAMPLES].tobytes() for i in chunks]

    def read(**options):
        ranges.clear()
        assert _gets(RfsStore(rfs, **options), chunks) == expected
        return sorted(ranges)

    one_run = lambda first, last: (_chunk_range(first)[0], _chunk_range(last)[1])  # noqa: E731
    # By default, chunks within 32 KiB of one another share a request, and chunk 40 is further away
    assert read() == [one_run(0, 5), _chunk_range(40)]
    # Chunks 0 and 1 are 20 bytes apart, and chunks 1 and 4 are two records and 20 bytes apart
    assert read(merge_gap=19) == [_chunk_range(i) for i in chunks]
    assert read(merge_gap=20) == [one_run(0, 1), one_run(4, 5), _chunk_range(40)]
    assert read(merge_gap=2 * RECORD + 20) == [one_run(0, 5), _chunk_range(40)]
    assert read(merge_gap=10**6) == [one_run(0, 40)]
    # A request may not ask for more than max_merge_size
    assert read(merge_gap=10**6, max_merge_size=3 * RECORD) == [one_run(0, 1), one_run(4, 5), _chunk_range(40)]
    assert read(max_merge_size=0) == [_chunk_range(i) for i in chunks]
    # Larger chunks are fetched on their own
    assert read(merge_below=2 * SAMPLES - 1) == [_chunk_range(i) for i in chunks]


def test_part_of_a_chunk_is_not_merged(served):
    from zarr.abc.store import RangeByteRequest

    url, x, ranges = served
    store = RfsStore(_rfs(url))

    async def together():
        prototype = default_buffer_prototype()
        return await asyncio.gather(
            store.get("data/c/3", prototype, RangeByteRequest(10, 30)), store.get("data/c/4", prototype)
        )

    part, whole = [buffer.to_bytes() for buffer in sync(together())]
    assert part == x[3 * SAMPLES : 4 * SAMPLES].tobytes()[10:30] and whole == x[4 * SAMPLES : 5 * SAMPLES].tobytes()
    assert sorted(ranges) == [(_chunk_range(3)[0] + 10, _chunk_range(3)[0] + 30), _chunk_range(4)]


def test_a_failed_request_reaches_every_caller(served, monkeypatch):
    url, _, _ = served
    store = RfsStore(_rfs(url))

    def fail(*args, **kwargs):
        raise OSError("connection lost")

    monkeypatch.setattr(rfs_store, "_read_bytes_from_url", fail)

    async def together():
        prototype = default_buffer_prototype()
        return await asyncio.gather(
            *(store.get(f"data/c/{i}", prototype) for i in range(3)), return_exceptions=True
        )

    results = sync(together())
    assert [type(result) for result in results] == [OSError] * 3


def test_merged_chunks_go_to_the_local_cache(served, tmp_path):
    from zarrshadow import LocalCache

    url, x, ranges = served
    cache = LocalCache(cache_dir=str(tmp_path / "cache"))
    expected = [x[i * SAMPLES : (i + 1) * SAMPLES].tobytes() for i in range(4)]
    assert _gets(RfsStore(_rfs(url), local_cache=cache), range(4)) == expected
    assert len(ranges) == 1
    # A second store finds each chunk in the cache and asks the server for nothing
    ranges.clear()
    assert _gets(RfsStore(_rfs(url), local_cache=cache), range(4)) == expected
    assert ranges == []
