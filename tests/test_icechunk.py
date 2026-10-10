"""Reference file systems written into Icechunk repositories, and read back out of them."""

import json

import numpy as np
import pytest
import zarr
from zarr.core.sync import sync

icechunk = pytest.importorskip("icechunk")
pytest.importorskip("virtualizarr.parsers.icechunk")

from zarrshadow import RfsBuilder, VirtualArray, generate_rfs, open_rfs, write_rfs  # noqa: E402
from zarrshadow.icechunk import icechunk_to_rfs, rfs_to_icechunk  # noqa: E402


@pytest.fixture
def repo(tmp_path):
    """An empty repository that may read virtual chunks from the files under tmp_path."""
    prefix = f"file://{tmp_path}/"
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(prefix, icechunk.local_filesystem_store(str(tmp_path)))
    )
    return icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path / "repo")),
        config=config,
        authorize_virtual_chunk_access={prefix: icechunk.credentials.LocalFileSystemAccess},
    )


def _committed(repo, rfs, **kwargs):
    """Write rfs into the repository and return a session that reads what was committed."""
    session = repo.writable_session("main")
    rfs_to_icechunk(rfs, session, **kwargs)
    session.commit("Add the references")
    return repo.readonly_session("main")


def _arrays(group):
    return {path: array for path, array in group.members(max_depth=None) if isinstance(array, zarr.Array)}


def _assert_same(a, b):
    """Two groups hold the same hierarchy, attributes, and data."""
    assert dict(a.attrs) == dict(b.attrs)
    assert sorted(path for path, _ in a.members(max_depth=None)) == sorted(path for path, _ in b.members(max_depth=None))
    arrays_a, arrays_b = _arrays(a), _arrays(b)
    for path, array in arrays_a.items():
        other = arrays_b[path]
        assert array.metadata.to_dict() == other.metadata.to_dict(), path
        np.testing.assert_array_equal(array[...], other[...], err_msg=path)


def _chunk_kinds(session, path):
    """How many chunks of an array the repository holds as native, virtual, and inline."""

    async def kinds():
        found = []
        async for batch in session.store.array_chunk_iterator(path):
            found.extend(batch[1].tolist())
        return found

    found = sync(kinds())
    names = {"native": icechunk.ChunkType.native, "virtual": icechunk.ChunkType.virtual, "inline": icechunk.ChunkType.inline}
    return {name: found.count(int(kind)) for name, kind in names.items() if found.count(int(kind))}


@pytest.fixture
def hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(0)
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as f:
        f.attrs["title"] = "test"
        group = f.create_group("acquisition")
        group.attrs["rate"] = 30000.0
        # 2,000 chunks, which get a chunk index
        group.create_dataset(
            "indexed", data=rng.integers(0, 100, (4000, 16)).astype("i2"), chunks=(2, 16), compression="gzip"
        )
        group.create_dataset(
            "listed", data=rng.integers(0, 100, (400, 16)).astype("i2"), chunks=(100, 16), compression="gzip"
        )
        # stored in one piece, and presented as slabs by one gen entry
        group.create_dataset("contiguous", data=rng.standard_normal((1000, 4)))
        group.create_dataset("sparse", shape=(40,), chunks=(10,), dtype="i4", fillvalue=-1)[10:20] = 7
        f.create_dataset("scalar", data=3.5)
        f.create_dataset("name", data="a session")
        f.create_dataset("labels", data=["a", "bc", "def"], dtype=h5py.string_dtype())
    return str(path)


def test_hdf5_round_trip(hdf5, repo, tmp_path):
    """Every way a chunk can be located carries into the repository and back out of it."""
    rfs = generate_rfs(hdf5, contiguous_chunk_bytes=8000)
    assert list(rfs["indexes"]) == ["acquisition/indexed"] and len(rfs["gen"]) == 1
    original = open_rfs(rfs)

    session = _committed(repo, rfs)
    _assert_same(zarr.open_group(session.store, mode="r"), original)
    # The data stayed in the HDF5 file, and what the references held is in the repository
    assert _chunk_kinds(session, "acquisition/indexed") == {"virtual": 2000}
    assert _chunk_kinds(session, "acquisition/listed") == {"virtual": 4}
    assert _chunk_kinds(session, "acquisition/contiguous") == {"virtual": 4}
    assert _chunk_kinds(session, "acquisition/sparse") == {"inline": 1}  # 40 bytes, held in the references
    assert _chunk_kinds(session, "labels") == {"inline": 1}

    back = icechunk_to_rfs(session)
    assert back["sources"] == rfs["sources"]
    assert list(back["indexes"]) == ["acquisition/indexed"]
    np.testing.assert_array_equal(back["indexes"]["acquisition/indexed"]["index"], rfs["indexes"]["acquisition/indexed"]["index"])
    assert back["refs"]["acquisition/listed/c/3/0"][1:] == rfs["refs"]["acquisition/listed/c/3/0"][1:]
    assert back["refs"]["labels/c/0"] == rfs["refs"]["labels/c/0"]
    _assert_same(open_rfs(back), original)

    # and from a directory, whose chunk indexes are read when they are needed
    write_rfs(rfs, str(tmp_path / "data.zarrshadow"))
    other = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path / "other")),
        config=repo.config,
        authorize_virtual_chunk_access={f"file://{tmp_path}/": icechunk.credentials.LocalFileSystemAccess},
    )
    _assert_same(zarr.open_group(_committed(other, str(tmp_path / "data.zarrshadow")).store, mode="r"), original)


@pytest.fixture
def recording(tmp_path):
    """17 interleaved channels after a header, in a file whose name needs escaping in a URL."""
    x = np.random.default_rng(1).integers(-500, 500, (10_007, 17)).astype("<i2")
    path = tmp_path / "raw data.bin"
    path.write_bytes(b"\0" * 12 + x.tobytes())
    return str(path), x


def test_short_last_chunk_is_copied(recording, repo):
    """The last chunk ends with the file, short of a full chunk, so its padded bytes go into the repository."""
    path, x = recording
    builder = RfsBuilder()
    builder.add_group("")
    VirtualArray.contiguous(path, shape=x.shape, dtype="<i2", offset=12, chunk_bytes=34_000).add_to(builder, "data")
    rfs = builder.build()
    assert rfs["refs"]["data/c/10/0"] == [path, 12 + 10_000 * 34, 7 * 34]

    session = _committed(repo, rfs)
    np.testing.assert_array_equal(zarr.open_group(session.store, mode="r")["data"][...], x)
    assert _chunk_kinds(session, "data") == {"native": 1, "virtual": 10}

    back = icechunk_to_rfs(session)
    assert back["templates"] == {"u0": path} and back["refs"]["data/c/3/0"] == ["{{u0}}", 12 + 3000 * 34, 34_000]
    assert back["sources"] == rfs["sources"]
    np.testing.assert_array_equal(open_rfs(back)["data"][...], x)


def test_native_chunks_can_be_referenced(recording, repo, tmp_path):
    """With the repository's chunks directory, the chunks it stores are referenced where they are."""
    path, x = recording
    builder = RfsBuilder()
    builder.add_group("")
    VirtualArray.contiguous(path, shape=x.shape, dtype="<i2", offset=12, chunk_bytes=34_000).add_to(builder, "data")
    session = _committed(repo, builder.build())

    copied = icechunk_to_rfs(session)
    assert isinstance(copied["refs"]["data/c/10/0"], str)
    referenced = icechunk_to_rfs(session, native_chunks_prefix=str(tmp_path / "repo" / "chunks"))
    url, offset, length = referenced["refs"]["data/c/10/0"]
    assert url.startswith(str(tmp_path / "repo" / "chunks")) and (offset, length) == (0, 34_000)
    np.testing.assert_array_equal(open_rfs(referenced)["data"][...], x)


def test_selection(recording, repo):
    """An array that is some of the columns of a file has no virtual form in Icechunk."""
    path, x = recording
    builder = RfsBuilder()
    builder.add_group("")
    raw = VirtualArray.contiguous(path, shape=x.shape, dtype="<i2", offset=12, chunk_bytes=34_000)
    raw[:, :16].add_to(builder, "neural")
    raw.add_to(builder, "all")
    rfs = builder.build()

    with pytest.raises(NotImplementedError, match="copy_selections=True"):
        rfs_to_icechunk(rfs, repo.writable_session("main"))

    session = _committed(repo, rfs, copy_selections=True)
    root = zarr.open_group(session.store, mode="r")
    np.testing.assert_array_equal(root["neural"][...], x[:, :16])
    np.testing.assert_array_equal(root["all"][...], x)
    assert _chunk_kinds(session, "neural") == {"native": 10}
    assert _chunk_kinds(session, "all") == {"native": 1, "virtual": 10}


def test_file_outside_the_containers(recording, repo, tmp_path):
    path, x = recording
    builder = RfsBuilder()
    builder.add_group("")
    builder.add_array("data", shape=[1000, 17], data_type="int16", chunk_shape=[1000, 17])
    builder.add_chunk("data", [0, 0], path, 12, 34_000)
    rfs = builder.build()

    with pytest.raises(ValueError, match="not within any virtual chunk container"):
        rfs_to_icechunk(rfs, repo.writable_session("main"), url_for=lambda url: "s3://bucket/raw.bin")

    # url_for says where each file is, in both directions
    session = _committed(repo, rfs, url_for=lambda url: f"file://{tmp_path}/raw%20data.bin")
    np.testing.assert_array_equal(zarr.open_group(session.store, mode="r")["data"][...], x[:1000])
    back = icechunk_to_rfs(session, url_for=lambda location: "https://example.org/raw.bin", record_sources=False)
    assert back["refs"]["data/c/0/0"] == ["https://example.org/raw.bin", 12, 34_000]


def test_attributes_that_are_not_finite(repo):
    """Icechunk refuses NaN and Infinity as bare words, so they are written as strings."""
    builder = RfsBuilder()
    builder.add_group("", {"resolution": float("nan"), "limits": [float("-inf"), 1.5, float("inf")], "name": "a"})
    builder.add_array("data", shape=[2], data_type="float64", chunk_shape=[2], fill_value="NaN")
    rfs = builder.build()
    assert "NaN" in rfs["refs"]["zarr.json"] and '"NaN"' not in rfs["refs"]["zarr.json"]

    session = _committed(repo, rfs)
    root = zarr.open_group(session.store, mode="r")
    assert dict(root.attrs) == {"resolution": "NaN", "limits": ["-Infinity", 1.5, "Infinity"], "name": "a"}
    assert np.isnan(root["data"][...]).all()
    assert json.loads(icechunk_to_rfs(session)["refs"]["zarr.json"])["attributes"]["resolution"] == "NaN"


def test_virtual_nwb_file(repo, tmp_path):
    """A virtual NWB file reads through hdmf-zarr from the repository, and from the references made from it."""
    pynwb = pytest.importorskip("pynwb")
    pytest.importorskip("hdmf_zarr.nwb")
    from datetime import datetime, timezone

    from hdmf_zarr import NWBZarrIO

    from zarrshadow import RfsStore, stack
    from zarrshadow.nwb import write_virtual_nwb

    rng = np.random.default_rng(2)
    x = rng.integers(-500, 500, (10_000, 16)).astype("<i2")
    (tmp_path / "raw.bin").write_bytes(b"\0" * 12 + x.tobytes() + b"\0" * 64)
    raw = VirtualArray.contiguous(
        str(tmp_path / "raw.bin"), shape=x.shape, dtype="<i2", offset=12, chunk_bytes=32_000, file_size=320_076
    )
    y = rng.integers(-500, 500, (5000, 4)).astype(">i2")
    channels = []
    for j in range(4):
        (tmp_path / f"channel{j}.bin").write_bytes(b"H" * 8 + y[:, j].tobytes())
        channels.append(
            VirtualArray.contiguous(str(tmp_path / f"channel{j}.bin"), shape=[len(y)], dtype=">i2", offset=8)
        )
    nwbfile = pynwb.NWBFile(
        session_description="s", identifier="i", session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    nwbfile.add_acquisition(pynwb.TimeSeries(name="signal", data=raw.placeholder(), unit="V", rate=30000.0))
    nwbfile.add_acquisition(pynwb.TimeSeries(name="lfp", data=stack(channels, axis=1).placeholder(), unit="V", rate=1000.0))
    rfs = write_virtual_nwb(nwbfile)

    session = _committed(repo, rfs)
    assert _chunk_kinds(session, "acquisition/signal/data") == {"virtual": 10}
    assert _chunk_kinds(session, "acquisition/lfp/data") == {"virtual": 4}
    for store in (session.store, RfsStore(icechunk_to_rfs(session))):
        with NWBZarrIO(store, mode="r") as io:
            read = io.read()
            np.testing.assert_array_equal(read.acquisition["signal"].data[...], x)
            np.testing.assert_array_equal(read.acquisition["lfp"].data[...], y)
            assert pynwb.validate(io=io) == []
