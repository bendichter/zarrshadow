"""What VirtualiZarr parses, written and used through zarrshadow."""

import json

import numpy as np
import pytest

pytest.importorskip("virtualizarr")

from virtualizarr.manifests import ChunkManifest, ManifestArray, ManifestGroup, ManifestStore  # noqa: E402
from virtualizarr.manifests.utils import create_v3_array_metadata  # noqa: E402

from zarrshadow import RfsBuilder, open_rfs, stack, write_rfs  # noqa: E402
from zarrshadow.virtualizarr import manifest_store_to_rfs, virtual_array  # noqa: E402

BYTES = [{"name": "bytes", "configuration": {"endian": "little"}}]


def _raw_manifest_array(path, x, chunk_rows, offset=0, codecs=BYTES):
    """A ManifestArray for an uncompressed array stored in one piece, in slabs along the first axis."""
    row_bytes = x[0].nbytes
    n = -(-len(x) // chunk_rows)
    grid = (n,) + (1,) * (x.ndim - 1)
    lengths = np.full(n, chunk_rows * row_bytes, dtype=np.uint64)
    lengths[-1] = (len(x) - (n - 1) * chunk_rows) * row_bytes
    manifest = ChunkManifest.from_arrays(
        paths=np.full(grid, f"file://{path}", dtype=np.dtypes.StringDType()),
        offsets=(offset + np.arange(n, dtype=np.uint64) * np.uint64(chunk_rows * row_bytes)).reshape(grid),
        lengths=lengths.reshape(grid),
    )
    metadata = create_v3_array_metadata(
        shape=x.shape, data_type=x.dtype, chunk_shape=(chunk_rows, *x.shape[1:]), codecs=codecs
    )
    return ManifestArray(metadata=metadata, chunkmanifest=manifest)


@pytest.fixture
def recording(tmp_path):
    x = np.random.default_rng(0).integers(-500, 500, (10_000, 17)).astype("<i2")
    path = tmp_path / "raw.bin"
    path.write_bytes(b"\0" * 12 + x.tobytes())
    return str(path), x


def _read(virtual):
    builder = RfsBuilder()
    builder.add_group("")
    virtual.add_to(builder, "data")
    return open_rfs(builder.build())["data"][...]


def test_manifest_store_to_rfs(recording, tmp_path):
    """Groups, attributes, referenced chunks, chunks held in memory, and missing chunks all carry over."""
    path, x = recording
    signal = _raw_manifest_array(path, x, chunk_rows=1000, offset=12)
    labels = ManifestArray(
        metadata=create_v3_array_metadata(shape=(4,), data_type=np.dtype("uint8"), chunk_shape=(2,), fill_value=9),
        # one chunk held in memory, and one that was never written
        chunkmanifest=ChunkManifest(
            entries={"0": {"path": "", "offset": 0, "length": 2, "data": bytes([3, 1])}}, shape=(2,)
        ),
    )
    group = ManifestGroup(
        arrays={"labels": labels},
        groups={"acquisition": ManifestGroup(arrays={"signal": signal}, attributes={"rate": 30000.0})},
        attributes={"session": "a"},
    )

    rfs = manifest_store_to_rfs(ManifestStore(group))

    assert rfs["sources"] == {path: {"size": 12 + x.nbytes}}
    assert rfs["templates"] == {"u0": path}
    assert rfs["refs"]["acquisition/signal/c/3/0"] == ["{{u0}}", 12 + 3000 * 34, 34_000]
    root = open_rfs(rfs)
    assert dict(root.attrs) == {"session": "a"} and root["acquisition"].attrs["rate"] == 30000.0
    np.testing.assert_array_equal(root["acquisition/signal"][...], x)
    np.testing.assert_array_equal(root["labels"][...], [3, 1, 9, 9])
    # A group works as well as a store
    assert manifest_store_to_rfs(group)["refs"] == rfs["refs"]


def test_arrays_with_many_chunks_get_an_index(recording, tmp_path):
    path, x = recording
    signal = _raw_manifest_array(path, x, chunk_rows=5, offset=12)  # 2,000 chunks
    rfs = manifest_store_to_rfs(ManifestGroup(arrays={"signal": signal}))
    assert list(rfs["indexes"]) == ["signal"] and rfs["indexes"]["signal"]["index"].shape == (2000, 1, 2)
    assert not any(key.startswith("signal/c/") for key in rfs["refs"])
    write_rfs(rfs, str(tmp_path / "raw.zarrshadow"))
    np.testing.assert_array_equal(open_rfs(str(tmp_path / "raw.zarrshadow"))["signal"][...], x)

    listed = manifest_store_to_rfs(ManifestGroup(arrays={"signal": signal}), index_threshold=None)
    assert "indexes" not in listed and listed["refs"]["signal/c/7/0"] == ["{{u0}}", 12 + 35 * 34, 170]


def test_virtual_array_of_a_contiguous_block(recording):
    """An uncompressed array in one piece can be sliced, which VirtualiZarr's own arrays cannot do by column."""
    path, x = recording
    array = virtual_array(_raw_manifest_array(path, x, chunk_rows=1000, offset=12))
    assert array.shape == (10_000, 17) and array.chunk_shape == (1000, 17)
    np.testing.assert_array_equal(_read(array), x)
    np.testing.assert_array_equal(_read(array[:, :16]), x[:, :16])
    np.testing.assert_array_equal(_read(array[1234:5678, [16, 2]]), x[1234:5678][:, [16, 2]])
    np.testing.assert_array_equal(_read(array.transpose()), x.T)


def test_virtual_array_of_a_short_last_chunk_and_big_endian(tmp_path):
    x = np.random.default_rng(1).integers(-500, 500, (2500, 3)).astype(">i4")
    (tmp_path / "be.bin").write_bytes(x.tobytes())
    big = [{"name": "bytes", "configuration": {"endian": "big"}}]
    array = virtual_array(_raw_manifest_array(str(tmp_path / "be.bin"), x, chunk_rows=1000, codecs=big))
    np.testing.assert_array_equal(_read(array), x)
    np.testing.assert_array_equal(_read(array[2400:, 1]), x[2400:, 1])


def test_virtual_array_of_other_arrays(recording):
    """An array whose chunks are not one run of one file keeps its chunks, and still stacks and transposes."""
    path, x = recording
    metadata = create_v3_array_metadata(shape=(2000, 17), data_type=x.dtype, chunk_shape=(1000, 17), codecs=BYTES)
    # the second chunk comes before the first in the file
    manifest = ChunkManifest(
        entries={
            "0.0": {"path": f"file://{path}", "offset": 12 + 34_000, "length": 34_000},
            "1.0": {"path": f"file://{path}", "offset": 12, "length": 34_000},
        }
    )
    array = virtual_array(ManifestArray(metadata=metadata, chunkmanifest=manifest))
    expected = np.concatenate([x[1000:2000], x[:1000]])
    np.testing.assert_array_equal(_read(array), expected)
    np.testing.assert_array_equal(_read(stack([array, array])), np.stack([expected, expected]))
    np.testing.assert_array_equal(_read(array.transpose()), expected.T)
    with pytest.raises(NotImplementedError, match="contiguous can be sliced"):
        array[:, :16]


def test_hdf5_through_virtualizarr(tmp_path):
    """A file VirtualiZarr's HDF5 parser reads, written as a reference file system."""
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("virtualizarr.parsers.hdf")
    from obspec_utils.registry import ObjectStoreRegistry
    from obstore.store import LocalStore
    from virtualizarr.parsers import HDFParser

    rng = np.random.default_rng(2)
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as f:
        f.attrs["title"] = "test"
        group = f.create_group("acquisition")
        group.create_dataset(
            "chunked", data=rng.integers(0, 100, (4000, 16)).astype("i2"), chunks=(100, 16), compression="gzip"
        )
        group.create_dataset("contiguous", data=rng.standard_normal((1000, 4)))

    store = HDFParser()(url=f"file://{path}", registry=ObjectStoreRegistry({"file://": LocalStore()}))
    root = open_rfs(manifest_store_to_rfs(store))
    with h5py.File(path) as f:
        assert root.attrs["title"] == "test"
        np.testing.assert_array_equal(root["acquisition/chunked"][...], f["acquisition/chunked"][...])
        np.testing.assert_array_equal(root["acquisition/contiguous"][...], f["acquisition/contiguous"][...])
        # the contiguous dataset is one run of bytes, so it can be sliced by column
        contiguous = virtual_array(store._group.groups["acquisition"].arrays["contiguous"])
        np.testing.assert_array_equal(_read(contiguous[:, 1:3]), f["acquisition/contiguous"][:, 1:3])


def test_manifest_array_in_an_nwb_file(recording):
    """A ManifestArray as the data of an NWB series."""
    pynwb = pytest.importorskip("pynwb")
    pytest.importorskip("hdmf_zarr.nwb")
    from datetime import datetime, timezone

    from hdmf_zarr import NWBZarrIO

    from zarrshadow import RfsStore
    from zarrshadow.nwb import write_virtual_nwb

    path, x = recording
    array = virtual_array(_raw_manifest_array(path, x, chunk_rows=1000, offset=12))
    nwbfile = pynwb.NWBFile(
        session_description="s", identifier="i", session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    nwbfile.add_acquisition(pynwb.TimeSeries(name="signal", data=array[:, :16].placeholder(), unit="V", rate=30000.0))
    nwbfile.add_acquisition(pynwb.TimeSeries(name="sync", data=array[:, 16].placeholder(), unit="a.u.", rate=30000.0))
    rfs = write_virtual_nwb(nwbfile)
    assert json.loads(rfs["refs"]["acquisition/signal/data/zarr.json"])["shape"] == [10_000, 16]
    with NWBZarrIO(RfsStore(rfs), mode="r") as io:
        read = io.read()
        np.testing.assert_array_equal(read.acquisition["signal"].data[...], x[:, :16])
        np.testing.assert_array_equal(read.acquisition["sync"].data[...], x[:, 16])
        assert pynwb.validate(io=io) == []
