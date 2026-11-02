"""Selections: arrays whose chunks are stored with other bytes in every record, read with the keep_bytes codec."""

import asyncio
import functools
import http.server
import json
import re
import threading

import numpy as np
import pytest
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest

import shadowzarr.rfs_store
from shadowzarr import RfsBuilder, RfsStore, open_rfs, write_rfs
from shadowzarr.builder import columns_selection
from shadowzarr.keep_bytes import KeepBytesCodec
from shadowzarr.rfs_store import _array_path

N_COLUMNS, HEADER, ROWS, CHUNK_ROWS = 17, 12, 10_000, 1000


@pytest.fixture
def recording(tmp_path):
    """A SpikeGLX-style file: 16 signal channels and a sync channel, interleaved, after a header."""
    x = np.random.default_rng(0).integers(-500, 500, (ROWS, N_COLUMNS)).astype("<i2")
    path = tmp_path / "raw.bin"
    path.write_bytes(b"\0" * HEADER + x.tobytes())
    return str(path), x


def _builder(url, x, columns, *, chunk_rows=CHUNK_ROWS, file_size=None):
    record_size, keep = columns_selection(x.shape[1], x.itemsize, columns)
    n_kept = sum(stop - start for start, stop in keep) // x.itemsize
    builder = RfsBuilder()
    builder.add_group("")
    builder.add_array("data", shape=[len(x), n_kept], data_type="int16", chunk_shape=[chunk_rows, n_kept])
    builder.add_selection("data", record_size, keep)
    builder.add_contiguous_chunks(
        "data", url=url, start=HEADER, shape=[len(x), n_kept], chunk_shape=[chunk_rows, n_kept],
        itemsize=x.itemsize, file_size=file_size, row_bytes=record_size,
    )
    return builder


def test_columns_selection():
    assert columns_selection(385, 2, slice(0, 384)) == (770, [[0, 768]])
    assert columns_selection(8, 4, [5, 2, 3, 7]) == (32, [[20, 24], [8, 16], [28, 32]])
    assert columns_selection(6, 2, slice(None, None, 2)) == (12, [[0, 2], [4, 6], [8, 10]])
    with pytest.raises(ValueError, match="not one of the 4 columns"):
        columns_selection(4, 2, [4])
    with pytest.raises(ValueError, match="No columns"):
        columns_selection(4, 2, [])


def _codecs(rfs, path="data"):
    return json.loads(rfs["refs"][f"{path}/zarr.json"])["codecs"]


def test_leading_columns(recording):
    path, x = recording
    rfs = _builder(path, x, slice(0, 16)).build()
    assert _codecs(rfs) == [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "shadowzarr.keep_bytes", "configuration": {"record_size": 34, "keep": [[0, 32]]}},
    ]
    data = open_rfs(rfs)["data"]
    assert data.shape == (ROWS, 16)
    np.testing.assert_array_equal(data[...], x[:, :16])
    np.testing.assert_array_equal(data[2345:6789, 3:9], x[2345:6789, 3:9])


@pytest.mark.parametrize("columns", [[16], [5, 2, 3, 16], slice(1, None, 4), slice(None)])
def test_any_columns_in_any_order(recording, columns):
    path, x = recording
    data = open_rfs(_builder(path, x, columns).build())["data"]
    np.testing.assert_array_equal(data[...], x[:, columns])


def test_short_last_chunk_is_padded(recording):
    """A last chunk that ends with the file is a short reference; the selection applies before padding."""
    path, x = recording
    builder = _builder(path, x, slice(0, 16), chunk_rows=3000)
    (last,) = [ref for key, ref in builder.refs.items() if key.startswith("data/c/")]
    assert last == [path, HEADER + 9000 * 34, 1000 * 34]
    np.testing.assert_array_equal(open_rfs(builder.build())["data"][...], x[:, :16])


def test_packet_headers(tmp_path):
    """Samples stored in packets, each with a header and a trailer."""
    x = np.random.default_rng(1).integers(-500, 500, (5000, 4)).astype("<i2")
    packets = np.zeros(5000, dtype=[("header", "u1", 6), ("samples", "<i2", 4), ("trailer", "u1", 2)])
    packets["header"], packets["samples"], packets["trailer"] = 0xAA, x, 0x55
    path = tmp_path / "packets.bin"
    path.write_bytes(packets.tobytes())

    builder = RfsBuilder()
    builder.add_group("")
    builder.add_array("data", shape=x.shape, data_type="int16", chunk_shape=[500, 4])
    builder.add_selection("data", record_size=16, keep=[[6, 14]])
    builder.add_strided_chunks("data", ndim=2, url=str(path), start=0, stride=8000, length=8000, count=10)
    np.testing.assert_array_equal(open_rfs(builder.build())["data"][...], x)


def test_written_forms(recording, tmp_path):
    path, x = recording
    rfs = _builder(path, x, slice(0, 16)).build()

    write_rfs(rfs, str(tmp_path / "raw.shadowzarr"))
    np.testing.assert_array_equal(open_rfs(str(tmp_path / "raw.shadowzarr"))["data"][...], x[:, :16])

    # A single file is plain version 1 kerchunk, with the selection in the array's codecs
    write_rfs(rfs, str(tmp_path / "raw.json"))
    written = json.loads((tmp_path / "raw.json").read_text())
    assert written["version"] == 1 and "gen" not in written
    np.testing.assert_array_equal(open_rfs(str(tmp_path / "raw.json"))["data"][...], x[:, :16])


def test_read_without_shadowzarr_store(recording, tmp_path):
    """fsspec's reference file system and zarr-python read the references, with the codec registered."""
    import fsspec
    import zarr
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

    path, x = recording
    write_rfs(_builder(path, x, [5, 2, 3, 16]).build(record_sources=False), str(tmp_path / "raw.json"))
    fs = fsspec.filesystem("reference", fo=str(tmp_path / "raw.json"))
    store = zarr.storage.FsspecStore(AsyncFileSystemWrapper(fs, asynchronous=True), read_only=True)
    np.testing.assert_array_equal(zarr.open_array(store, path="data", mode="r")[...], x[:, [5, 2, 3, 16]])


def test_invalid_selections(recording):
    path, x = recording
    builder = RfsBuilder()
    with pytest.raises(KeyError, match="before its selection"):
        builder.add_selection("data", 8, [[0, 2]])
    builder.add_array("data", shape=[10, 1], data_type="int16", chunk_shape=[10, 1])
    for record_size, keep in [(0, [[0, 1]]), (8, []), (8, [[4, 4]]), (8, [[6, 10]]), (8, [[-1, 2]])]:
        with pytest.raises(ValueError):
            builder.add_selection("data", record_size, keep)
    # A second selection replaces the first
    builder.add_selection("data", 8, [[0, 2]])
    builder.add_selection("data", 4, [[2, 4]])
    assert _codecs(builder.build())[1:] == [
        {"name": "shadowzarr.keep_bytes", "configuration": {"record_size": 4, "keep": [[2, 4]]}}
    ]

    # A reference that is not a whole number of records
    builder = _builder(path, x, slice(0, 16))
    builder.refs["data/c/0/0"] = [path, HEADER, 35]
    builder.gen.clear()
    with pytest.raises(ValueError, match="not a whole number of 34 byte records"):
        open_rfs(builder.build())["data"][:10]


def test_codec():
    codec = KeepBytesCodec(record_size=10, keep=[[2, 6], [8, 10]])  # 6 of every 10 bytes
    data = bytes(range(40))
    assert codec.apply(data) == bytes([2, 3, 4, 5, 8, 9, 12, 13, 14, 15, 18, 19, 22, 23, 24, 25, 28, 29,
                                       32, 33, 34, 35, 38, 39])
    assert codec.kept == 6 and codec.encoded_size(24) == 40
    assert KeepBytesCodec.from_dict(codec.to_dict()) == codec
    with pytest.raises(ValueError, match="not a whole number of 10 byte records"):
        codec.apply(data[:35])
    with pytest.raises(ValueError, match="Invalid keep_bytes"):
        KeepBytesCodec(record_size=10, keep=[[8, 12]])


def test_array_path():
    assert _array_path("a/b/c/3/0") == "a/b"
    assert _array_path("a/c") == "a"
    assert _array_path("c/1") == "" and _array_path("c") == ""
    assert _array_path("a/c/zarr.json") is None and _array_path("zarr.json") is None
    assert _array_path("c/data/c/0") == "c/data"


def _get(store, key, byte_range=None):
    buffer = asyncio.run(store.get(key, byte_range=byte_range))
    return None if buffer is None else buffer.to_bytes()


@pytest.fixture
def reads(monkeypatch):
    """Record the (offset, length) of every read from a file."""
    log = []
    read = shadowzarr.rfs_store._read_bytes_from_url_or_path

    def logged(url_or_path, offset, length, **kwargs):
        log.append((offset, length))
        return read(url_or_path, offset, length, **kwargs)

    monkeypatch.setattr(shadowzarr.rfs_store, "_read_bytes_from_url_or_path", logged)
    return log


def test_part_of_a_chunk_is_a_range_of_its_records(recording, reads):
    """A byte range of a chunk is a range of the records in the file, which the codec decodes."""
    path, x = recording
    store = RfsStore(_builder(path, x, slice(0, 16)).build())
    records = x[2000:3000].tobytes()  # chunk 2: 1,000 rows of 34 bytes

    reads.clear()
    assert _get(store, "data/c/2/0", RangeByteRequest(100 * 34, 110 * 34)) == records[3400:3740]
    assert reads == [(HEADER + 2100 * 34, 10 * 34)]
    assert _get(store, "data/c/2/0", SuffixByteRequest(68)) == records[-68:]
    assert _get(store, "data/c/2/0") == records


def test_part_of_a_chunk_without_a_selection(recording, reads):
    """Byte ranges of ordinary references are read from the file as ranges too."""
    path, x = recording
    builder = RfsBuilder()
    builder.add_group("")
    builder.add_array("data", shape=x.shape, data_type="int16", chunk_shape=[CHUNK_ROWS, N_COLUMNS])
    builder.add_contiguous_chunks(
        "data", url=path, start=HEADER, shape=x.shape, chunk_shape=[CHUNK_ROWS, N_COLUMNS], itemsize=2
    )
    store = RfsStore(builder.build())
    chunk = x[2000:3000].tobytes()
    reads.clear()
    assert _get(store, "data/c/2/0", RangeByteRequest(340, 680)) == chunk[340:680]
    assert reads == [(HEADER + 2000 * 34 + 340, 340)]
    assert _get(store, "zarr.json", RangeByteRequest(0, 5)) == _get(store, "zarr.json")[:5]


def test_part_of_a_padded_chunk(recording, reads):
    """The padding of a short last chunk, whole records of zeros, is not read from the file."""
    path, x = recording
    store = RfsStore(_builder(path, x, slice(0, 16), chunk_rows=3000).build())
    records = x[9000:].tobytes() + b"\0" * (2000 * 34)
    reads.clear()
    assert _get(store, "data/c/3/0", RangeByteRequest(990 * 34, 1010 * 34)) == records[990 * 34 : 1010 * 34]
    assert reads == [(HEADER + 9990 * 34, 10 * 34)]
    reads.clear()
    assert _get(store, "data/c/3/0", RangeByteRequest(2000 * 34, 2001 * 34)) == b"\0" * 34
    assert reads == []
    assert _get(store, "data/c/3/0") == records


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static files with single byte-range support."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        match = re.match(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if not match:
            return super().do_GET()
        data = open(self.translate_path(self.path), "rb").read()
        start, end = int(match[1]), min(int(match[2]), len(data) - 1)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.wfile.write(data[start : end + 1])


def test_remote_file(recording, tmp_path):
    """Remote chunks are fetched in merged requests, and the selection applies to each."""
    path, x = recording
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_RangeHandler, directory=str(tmp_path)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/raw.bin"
        data = open_rfs(_builder(url, x, [5, 2, 3, 16]).build(record_sources=False))["data"]
        np.testing.assert_array_equal(data[...], x[:, [5, 2, 3, 16]])
        np.testing.assert_array_equal(data[1500:1600], x[1500:1600][:, [5, 2, 3, 16]])
    finally:
        server.shutdown()
