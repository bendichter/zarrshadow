"""Build reference file systems for non-HDF5 sources with RfsBuilder."""

import json

import fsspec
import numpy as np
import pytest
import zarr

from zindi import RfsBuilder, open_rfs, write_rfs
from zindi.builder import chunk_key
from zindi.chunk_index import MISSING

N_CH, HEADER, ROWS = 16, 12, 1000


@pytest.fixture
def raw_recording(tmp_path):
    """A SpikeGLX-style file: interleaved int16 channels after a header."""
    x = np.random.default_rng(0).integers(-500, 500, (10_000, N_CH)).astype("<i2")
    path = tmp_path / "raw.bin"
    path.write_bytes(b"\0" * HEADER + x.tobytes())
    return str(path), x


def _raw_builder(path, x):
    builder = RfsBuilder()
    builder.add_group("", {"source": "raw binary"})
    builder.add_array("data", shape=x.shape, data_type="int16", chunk_shape=[ROWS, N_CH])
    nbytes = ROWS * N_CH * 2
    builder.add_strided_chunks(
        "data", ndim=2, url=path, start=HEADER, stride=nbytes, length=nbytes, count=len(x) // ROWS
    )
    return builder


def test_chunk_key():
    assert chunk_key("a/b", (3, 0)) == "a/b/c/3/0"
    assert chunk_key("a", ()) == "a/c"
    assert chunk_key("", (1,)) == "c/1"


def test_strided_raw_recording(raw_recording):
    path, x = raw_recording
    rfs = _raw_builder(path, x).build()
    assert rfs["version"] == 2
    assert rfs["sources"] == {path: {"size": HEADER + x.nbytes}}
    root = open_rfs(rfs)
    assert root.attrs["source"] == "raw binary"
    np.testing.assert_array_equal(root["data"][...], x)
    np.testing.assert_array_equal(root["data"][2345:6789, 3:9], x[2345:6789, 3:9])


def test_strided_written_forms(raw_recording, tmp_path):
    path, x = raw_recording
    rfs = _raw_builder(path, x).build()

    write_rfs(rfs, str(tmp_path / "raw.zindi"))
    np.testing.assert_array_equal(open_rfs(str(tmp_path / "raw.zindi"))["data"][...], x)

    # The single JSON lists every chunk, so stock fsspec and zarr read it
    write_rfs(rfs, str(tmp_path / "raw.json"))
    assert json.loads((tmp_path / "raw.json").read_text())["version"] == 1
    fs = fsspec.filesystem("reference", fo=str(tmp_path / "raw.json"), asynchronous=True)
    store = zarr.storage.FsspecStore(fs, read_only=True, path="")
    np.testing.assert_array_equal(zarr.open_array(store, path="data", mode="r")[...], x)


def test_index_for_irregular_layout(tmp_path):
    """Chunks stored out of order, one never written: described by an index array."""
    x = np.arange(4000, dtype="<f8")
    chunks = [x[i * 500 : (i + 1) * 500] for i in range(8)]
    order = [3, 0, 7, 5, 1, 6, 2]  # chunk 4 is never written
    path = tmp_path / "shuffled.bin"
    index = np.full((8, 2), MISSING, dtype=np.uint64)
    with open(path, "wb") as f:
        for i in order:
            index[i] = (f.tell(), chunks[i].nbytes)
            f.write(chunks[i].tobytes())

    builder = RfsBuilder()
    builder.add_group("")
    builder.add_array("x", shape=x.shape, data_type="float64", chunk_shape=[500], fill_value=-1.0)
    builder.add_index("x", str(path), index)
    rfs = builder.build()

    expected = x.copy()
    expected[2000:2500] = -1.0  # the unwritten chunk reads as the fill value
    np.testing.assert_array_equal(open_rfs(rfs)["x"][...], expected)
    write_rfs(rfs, str(tmp_path / "shuffled.zindi"))
    np.testing.assert_array_equal(open_rfs(str(tmp_path / "shuffled.zindi"))["x"][...], expected)


def test_inline_chunks():
    builder = RfsBuilder()
    builder.add_group("")
    builder.add_array("small", shape=[3], data_type="int32", chunk_shape=[3])
    builder.add_inline_chunk("small", [0], np.array([1, 2, 3], dtype="<i4").tobytes())
    rfs = builder.build(record_sources=False)
    assert rfs["refs"]["small/c/0"].startswith("base64:")
    np.testing.assert_array_equal(open_rfs(rfs)["small"][...], [1, 2, 3])

    builder.add_inline_chunk("note", [], b"plain text")
    assert builder.refs["note/c"] == "plain text"
