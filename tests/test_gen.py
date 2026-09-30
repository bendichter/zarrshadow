"""Tests for kerchunk gen entries: lazy evaluation and contiguous datasets."""

import json
import os

import fsspec
import h5py
import numpy as np
import pytest
import zarr
from zarr.core.sync import _collect_aiterator, sync

from zindi import generate_rfs, open_rfs, write_rfs
from zindi.gen import Generator, evaluate, render
from zindi.rfs_store import RfsStore


def test_evaluate_arithmetic():
    assert evaluate("12 + i * 256000", {"i": 3}) == 768012
    assert evaluate("(i + 1) * 4 // 2 % 5 - -1", {"i": 2}) == 2
    assert render("{{u}}", {"u": "https://example.org/raw.bin"}) == "https://example.org/raw.bin"


@pytest.mark.parametrize("expression", ["__import__('os')", "i.real", "[i]", "i ** 2", "j + 1"])
def test_evaluate_refuses_other_expressions(expression):
    with pytest.raises(ValueError):
        evaluate(expression, {"i": 1})


def test_generator_lookup():
    entry = {
        "key": "data/c/{{i}}/{{j}}",
        "url": "{{u}}",
        "offset": "{{10 + i * 100 + j * 10}}",
        "length": "10",
        "dimensions": {"i": {"start": 2, "stop": 8, "step": 2}, "j": [0, 3]},
    }
    g = Generator(entry, {"u": "file.bin"})
    assert g.lookup("data/c/4/3") == ["file.bin", 440, 10]
    assert g.lookup("data/c/3/0") is None  # off the step
    assert g.lookup("data/c/8/0") is None  # past stop
    assert g.lookup("data/c/2/1") is None  # not in the list
    assert g.lookup("other/c/2/0") is None
    assert len(list(g.items())) == 6


def test_raw_binary_via_gen(tmp_path):
    """A SpikeGLX-style file: interleaved int16 channels after a header."""
    n_ch, header, rows = 16, 12, 1000
    x = np.random.default_rng(0).integers(-500, 500, (10_000, n_ch)).astype("<i2")
    path = tmp_path / "raw.bin"
    path.write_bytes(b"\0" * header + x.tobytes())
    nbytes = rows * n_ch * 2
    array_meta = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(x.shape),
        "data_type": "int16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [rows, n_ch]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }
    rfs = {
        "version": 1,
        "templates": {"u": str(path)},
        "refs": {
            "zarr.json": json.dumps({"zarr_format": 3, "node_type": "group"}),
            "data/zarr.json": json.dumps(array_meta),
        },
        "gen": [
            {
                "key": "data/c/{{i}}/0",
                "url": "{{u}}",
                "offset": f"{{{{{header} + i * {nbytes}}}}}",
                "length": str(nbytes),
                "dimensions": {"i": {"stop": 10}},
            }
        ],
    }
    root = open_rfs(rfs)
    np.testing.assert_array_equal(root["data"][...], x)
    np.testing.assert_array_equal(root["data"][2345:6789, 3:9], x[2345:6789, 3:9])


@pytest.fixture
def contiguous_h5(tmp_path):
    path = str(tmp_path / "contiguous.h5")
    with h5py.File(path, "w", userblock_size=512) as f:  # MATLAB v7.3 files have one
        f.create_dataset("matrix", data=np.random.default_rng(1).standard_normal((1000, 7)))
        f.create_dataset("small", data=np.arange(100.0))
    return path


def _expected(path):
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in f}


def test_contiguous_dataset_split(contiguous_h5):
    rfs = generate_rfs(contiguous_h5, contiguous_chunk_bytes=8192)
    assert rfs["version"] == 2
    (entry,) = rfs["gen"]
    assert entry["key"] == "matrix/c/{{i}}/0"
    meta = json.loads(rfs["refs"]["matrix/zarr.json"])
    # 8192 bytes is 146 rows of 7 float64; 125 divides 1000, so no slab is short
    assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [125, 7]
    assert entry["dimensions"] == {"i": {"stop": 8}}
    assert not any(k.startswith("matrix/c/") for k in rfs["refs"])

    root = open_rfs(rfs)
    expected = _expected(contiguous_h5)
    np.testing.assert_array_equal(root["matrix"][...], expected["matrix"])
    np.testing.assert_array_equal(root["matrix"][140:160, 2:5], expected["matrix"][140:160, 2:5])
    np.testing.assert_array_equal(root["small"][...], expected["small"])

    store = RfsStore(rfs)
    listing = sync(_collect_aiterator(store.list_dir("matrix/c")))
    assert sorted(listing, key=int) == [str(i) for i in range(8)]
    assert sorted(sync(_collect_aiterator(store.list_dir("matrix")))) == ["c", "zarr.json"]


@pytest.mark.parametrize("last_in_file", [False, True])
def test_contiguous_split_uneven(tmp_path, last_in_file):
    """A length with no divisor near the slab height: the last slab is read at full
    length when the file extends far enough, and is a short ref at the end of the file."""
    path = str(tmp_path / "uneven.h5")
    x = np.random.default_rng(2).standard_normal(1009)  # prime, and above the inline threshold
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=x)
        if not last_in_file:
            f.create_dataset("after", data=np.zeros(1000))
    rfs = generate_rfs(path, contiguous_chunk_bytes=800)  # 100 elements per slab, 11 slabs
    (entry,) = [g for g in rfs["gen"] if g["key"].startswith("x/")]
    if last_in_file:
        assert entry["dimensions"] == {"i": {"stop": 10}}
        assert rfs["refs"]["x/c/10"][2] == 9 * 8  # short last slab
    else:
        assert entry["dimensions"] == {"i": {"stop": 11}}
        assert "x/c/10" not in rfs["refs"]
        # Every chunk is full size, so readers that do not pad read it too
        write_rfs(rfs, str(tmp_path / "uneven.json"))
        fs = fsspec.filesystem("reference", fo=str(tmp_path / "uneven.json"), asynchronous=True)
        store = zarr.storage.FsspecStore(fs, read_only=True, path="")
        np.testing.assert_array_equal(zarr.open_array(store, path="x", mode="r")[...], x)
    np.testing.assert_array_equal(open_rfs(rfs)["x"][...], x)


def test_contiguous_split_off(contiguous_h5):
    rfs = generate_rfs(contiguous_h5, contiguous_chunk_bytes=None)
    assert "gen" not in rfs and rfs["version"] == 1
    meta = json.loads(rfs["refs"]["matrix/zarr.json"])
    assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [1000, 7]


def test_gen_written_forms(contiguous_h5, tmp_path):
    rfs = generate_rfs(contiguous_h5, contiguous_chunk_bytes=8192)
    expected = _expected(contiguous_h5)["matrix"]

    # Directory: gen kept, version 2
    write_rfs(rfs, str(tmp_path / "c.zindi"))
    header = json.loads((tmp_path / "c.zindi" / "refs.json").read_text())
    assert header["version"] == 2 and len(header["gen"]) == 1
    np.testing.assert_array_equal(open_rfs(str(tmp_path / "c.zindi"))["matrix"][...], expected)

    # Single JSON: version 1 with every slab listed, readable by stock fsspec
    write_rfs(rfs, str(tmp_path / "c.json"))
    single = json.loads((tmp_path / "c.json").read_text())
    assert single["version"] == 1 and "gen" not in single
    fs = fsspec.filesystem("reference", fo=str(tmp_path / "c.json"), asynchronous=True)
    arr = zarr.open_array(zarr.storage.FsspecStore(fs, read_only=True, path=""), path="matrix", mode="r")
    np.testing.assert_array_equal(arr[...], expected)
