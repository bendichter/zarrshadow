"""Tests for recording source files and detecting when they change."""

import http.server
import os
import re
import threading

import h5py
import numpy as np
import pytest
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from zindi import SourceChangedError, generate_rfs, open_rfs
from zindi import sources as sources_module
from zindi.rfs_store import RfsStore


def _write_h5(path):
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.arange(20_000, dtype="f8"), chunks=(1000,))


@pytest.fixture
def h5_path(tmp_path):
    path = str(tmp_path / "test.h5")
    _write_h5(path)
    return path


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves one file with Range, ETag, and If-Match, like S3."""

    file_path = ""
    etag = '"v1"'
    honor_if_match = True
    reported_total = None  # override the size in Content-Range

    def log_message(self, *args):
        pass

    def do_GET(self):
        data = open(self.file_path, "rb").read()
        if_match = self.headers.get("If-Match")
        if self.honor_if_match and if_match is not None and if_match != self.etag:
            self.send_response(412)
            self.end_headers()
            return
        match = re.match(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        total = self.reported_total or len(data)
        if match:
            start, end = int(match[1]), min(int(match[2]), len(data) - 1)
            body = data[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        else:
            body = data
            self.send_response(200)
        self.send_header("ETag", self.etag)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def served(h5_path):
    handler = type("Handler", (_RangeHandler,), {"file_path": h5_path})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/test.h5", handler
    server.shutdown()


def test_local_source_records_size(h5_path):
    rfs = generate_rfs(h5_path)
    assert rfs["sources"] == {h5_path: {"size": os.path.getsize(h5_path)}}


def test_record_sources_off(h5_path):
    assert "sources" not in generate_rfs(h5_path, record_sources=False)


def test_local_source_changed(h5_path, tmp_path):
    rfs = generate_rfs(h5_path)
    np.testing.assert_array_equal(open_rfs(rfs)["data"][:5], np.arange(5))
    with open(h5_path, "ab") as f:
        f.write(b"\0" * 10)
    with pytest.raises(SourceChangedError, match="has changed"):
        open_rfs(rfs)["data"][:5]
    # Checking can be turned off
    np.testing.assert_array_equal(open_rfs(rfs, validate_sources=False)["data"][:5], np.arange(5))


def test_http_source_records_etag_and_size(served, h5_path):
    url, _ = served
    rfs = generate_rfs(url)
    assert rfs["sources"][url] == {"size": os.path.getsize(h5_path), "etag": '"v1"'}
    np.testing.assert_array_equal(open_rfs(rfs)["data"][...], np.arange(20_000))


def test_http_source_changed_etag(served):
    url, handler = served
    rfs = generate_rfs(url)
    handler.etag = '"v2"'  # the file was replaced
    with pytest.raises(SourceChangedError, match="ETag"):
        open_rfs(rfs)["data"][:5]


def test_http_source_changed_size_without_if_match(served, h5_path):
    url, handler = served
    rfs = generate_rfs(url)
    handler.honor_if_match = False
    handler.reported_total = os.path.getsize(h5_path) + 100
    with pytest.raises(SourceChangedError, match="size"):
        open_rfs(rfs)["data"][:5]


def test_merged_range_reads_are_checked(served):
    url, handler = served
    store = RfsStore(generate_rfs(url, chunk_index_threshold=None))
    keys = [("data/c/0", None), ("data/c/1", None)]
    buffers = sync(store.get_partial_values(default_buffer_prototype(), keys))
    assert all(b is not None for b in buffers)
    handler.etag = '"v2"'
    with pytest.raises(SourceChangedError):
        sync(store.get_partial_values(default_buffer_prototype(), keys))


def test_describe_dandi_asset_uses_metadata(monkeypatch):
    calls = []

    class Response:
        ok = True

        def json(self):
            return {"contentSize": 123, "digest": {"dandi:dandi-etag": "abc-2"}}

    def fake_get(url, headers=None, **kwargs):
        calls.append(url)
        return Response()

    monkeypatch.setattr(sources_module.requests, "get", fake_get)
    url = "https://api.dandiarchive.org/api/assets/5a9cc6f1-aeaf-46cc-aae7-ea27960236ea/download/"
    assert sources_module.describe_source(url) == {"size": 123, "etag": '"abc-2"'}
    assert calls == ["https://api.dandiarchive.org/api/assets/5a9cc6f1-aeaf-46cc-aae7-ea27960236ea/"]
