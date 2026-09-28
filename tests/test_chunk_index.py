"""Tests for chunk indexes: arrays with many chunks stored as index arrays."""

import functools
import http.server
import json
import os
import threading

import h5py
import numpy as np
import pytest
import zarr

from zindi import generate_rfs, open_rfs, write_rfs
from zindi.chunk_index import INDEX_BLOCK_ENTRIES, MISSING, index_block_shape

THRESHOLD = 50


@pytest.fixture(scope="module")
def h5_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("h5") / "test.h5")
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        # 1-D, 500 chunks, last chunk partial
        f.create_dataset("series", data=rng.standard_normal(4995), chunks=(10,), compression="gzip")
        # 2-D grid of 30 x 4 chunks with partial edges
        f.create_dataset("matrix", data=rng.integers(0, 1000, (295, 38)).astype("int32"), chunks=(10, 10))
        # Only some chunks written; the rest are unallocated and read as fill value
        sparse = f.create_dataset("sparse", shape=(1000,), dtype="float32", chunks=(10,), fillvalue=-1.0)
        sparse[100:200] = np.arange(100, dtype="float32")
        sparse[900:910] = 7.0
        # Few chunks: stays in refs
        f.create_dataset("small", data=np.arange(100.0), chunks=(10,))
    return path


def _expected(h5_path):
    with h5py.File(h5_path, "r") as f:
        return {name: f[name][()] for name in ["series", "matrix", "sparse", "small"]}


@pytest.fixture(scope="module")
def rfs(h5_path):
    return generate_rfs(h5_path, chunk_index_threshold=THRESHOLD)


def test_indexed_arrays_have_no_chunk_refs(rfs):
    assert set(rfs["chunk_indexes"]) == {"series", "matrix", "sparse"}
    assert not any(k.startswith(("series/c/", "matrix/c/", "sparse/c/")) for k in rfs["refs"])
    assert any(k.startswith("small/c/") for k in rfs["refs"])


def test_index_contents(rfs):
    index = rfs["chunk_indexes"]["matrix"]["index"]
    assert index.shape == (30, 4, 2)
    assert index.dtype == np.uint64
    sparse = rfs["chunk_indexes"]["sparse"]["index"]
    written = sparse[:, 0] != MISSING
    assert written.sum() == 11
    assert written[10:20].all() and written[90]


def test_threshold_none_lists_every_chunk(h5_path):
    rfs = generate_rfs(h5_path, chunk_index_threshold=None)
    assert "chunk_indexes" not in rfs
    assert sum(k.startswith("series/c/") for k in rfs["refs"]) == 500


@pytest.mark.parametrize("form", ["memory", "directory", "json"])
def test_roundtrip(rfs, h5_path, tmp_path, form):
    if form == "memory":
        root = open_rfs(rfs)
    elif form == "directory":
        write_rfs(rfs, str(tmp_path / "test.zindi"))
        root = open_rfs(str(tmp_path / "test.zindi"))
    else:
        write_rfs(rfs, str(tmp_path / "test.zindi.json"))
        root = open_rfs(str(tmp_path / "test.zindi.json"))
    for name, expected in _expected(h5_path).items():
        np.testing.assert_array_equal(root[name][...], expected, err_msg=name)
    # Partial reads touch only some chunks
    np.testing.assert_array_equal(root["matrix"][13:47, 5:9], _expected(h5_path)["matrix"][13:47, 5:9])


def test_json_expansion_matches_unindexed(h5_path, rfs, tmp_path):
    write_rfs(rfs, str(tmp_path / "expanded.json"))
    with open(tmp_path / "expanded.json") as f:
        expanded = json.load(f)
    plain = generate_rfs(h5_path, chunk_index_threshold=None)
    assert expanded["refs"] == json.loads(json.dumps(plain["refs"]))


def test_directory_layout(rfs, tmp_path):
    out = str(tmp_path / "test.zindi")
    write_rfs(rfs, out)
    with open(os.path.join(out, "refs.json")) as f:
        header = json.load(f)
    assert header["chunk_indexes"]["series"] == {"url": rfs["chunk_indexes"]["series"]["url"]}
    arr = zarr.open_array(os.path.join(out, "index"), path="matrix", mode="r")
    assert arr.shape == (30, 4, 2)
    assert arr.chunks == (30, 4, 2)
    np.testing.assert_array_equal(arr[...], rfs["chunk_indexes"]["matrix"]["index"])
    # Rewriting replaces the index arrays
    write_rfs(rfs, out)


def test_write_refuses_foreign_directory(rfs, tmp_path):
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        write_rfs(rfs, str(tmp_path / "other"))


def test_directory_open_reads_no_index(rfs, tmp_path):
    out = str(tmp_path / "test.zindi")
    write_rfs(rfs, out)
    root = open_rfs(out)
    store = root.store
    assert all(callable(index._source) for index in store._indexes.values())
    root["series"][:5]
    assert not callable(store._indexes["series"]._source)
    assert callable(store._indexes["matrix"]._source)


def test_listing(rfs):
    root = open_rfs(rfs)
    assert sorted(root.array_keys()) == ["matrix", "series", "small", "sparse"]
    store = root.store

    async def listdir(prefix):
        return [k async for k in store.list_dir(prefix)]

    sync = zarr.core.sync.sync
    assert sync(listdir("sparse")) == ["c", "zarr.json"]
    assert sorted(sync(listdir("sparse/c")), key=int) == [str(i) for i in range(10, 20)] + ["90"]
    assert sorted(sync(listdir("matrix/c/3")), key=int) == ["0", "1", "2", "3"]
    assert sync(store.exists("sparse/c/15"))
    assert not sync(store.exists("sparse/c/50"))


def test_index_block_shape():
    assert index_block_shape([500]) == [500]
    assert index_block_shape([10**6]) == [INDEX_BLOCK_ENTRIES]
    assert index_block_shape([1000, 500]) == [131, 500]
    assert index_block_shape([10**6, 1]) == [INDEX_BLOCK_ENTRIES, 1]
    shape = index_block_shape([50, 400, 400])
    assert shape[1:] == [163, 400] and shape[0] == 1


def test_open_directory_over_http(rfs, h5_path, tmp_path):
    out = tmp_path / "served" / "test.zindi"
    write_rfs(rfs, str(out))
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(tmp_path / "served")
    )
    handler.log_message = lambda *args: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        root = open_rfs(f"http://127.0.0.1:{server.server_port}/test.zindi/")
        expected = _expected(h5_path)
        np.testing.assert_array_equal(root["sparse"][...], expected["sparse"])
        np.testing.assert_array_equal(root["matrix"][...], expected["matrix"])
    finally:
        server.shutdown()
