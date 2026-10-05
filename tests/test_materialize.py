"""materialize: write the data a reference file system points at into a Zarr store of its own."""

import json
from datetime import datetime, timezone

import numpy as np
import pytest
import zarr

from zarrshadow import RfsBuilder, VirtualArray, materialize, open_rfs, write_rfs


@pytest.fixture
def virtual(tmp_path):
    """References to a slowly varying recording of 16 channels and a sync channel, and a small stored array."""
    x = np.cumsum(np.random.default_rng(0).integers(-3, 4, (50_000, 17)), axis=0).astype("<i2")
    (tmp_path / "raw.bin").write_bytes(b"\0" * 12 + x.tobytes())
    raw = VirtualArray.contiguous(str(tmp_path / "raw.bin"), shape=x.shape, dtype="<i2", offset=12)
    builder = RfsBuilder()
    builder.add_group("", {"session": "a"})
    builder.add_group("acquisition")
    raw[:, :16].add_to(builder, "acquisition/signal", attributes={"unit": "uV"}, dimension_names=["time", "channel"])
    raw[:, 16].add_to(builder, "acquisition/sync")
    builder.add_array("labels", shape=[4], data_type="uint8", chunk_shape=[4], codecs=[{"name": "bytes"}])
    builder.add_inline_chunk("labels", (0,), bytes([3, 1, 4, 1]))
    return builder.build(), x


def test_materialize(virtual, tmp_path):
    rfs, x = virtual
    report = materialize(rfs, str(tmp_path / "real.zarr"), chunk_bytes=2**18)

    # An ordinary Zarr store, with no references and nothing specific to zarrshadow
    root = zarr.open_group(str(tmp_path / "real.zarr"), mode="r")
    assert dict(root.attrs) == {"session": "a"}
    signal = root["acquisition/signal"]
    np.testing.assert_array_equal(signal[...], x[:, :16])
    np.testing.assert_array_equal(root["acquisition/sync"][...], x[:, 16])
    np.testing.assert_array_equal(root["labels"][...], [3, 1, 4, 1])
    assert signal.attrs["unit"] == "uV" and signal.metadata.dimension_names == ("time", "channel")
    assert signal.chunks[1] == 16 and signal.chunks[0] * 32 <= 2**18
    assert len(signal.metadata.codecs) == 2  # bytes and a compressor

    assert set(report) == {"acquisition/signal", "acquisition/sync"}
    assert report["acquisition/signal"]["nbytes"] == x[:, :16].nbytes
    assert report["acquisition/signal"]["nbytes_stored"] < x[:, :16].nbytes

    # The source file is no longer needed
    (tmp_path / "raw.bin").unlink()
    np.testing.assert_array_equal(signal[40_000:40_010], x[40_000:40_010, :16])


def test_from_a_written_file_into_a_store(virtual, tmp_path):
    rfs, x = virtual
    write_rfs(rfs, str(tmp_path / "virtual.zarrshadow"))
    store = zarr.storage.MemoryStore()
    materialize(str(tmp_path / "virtual.zarrshadow"), store, verify=True)
    np.testing.assert_array_equal(zarr.open_group(store, mode="r")["acquisition/signal"][...], x[:, :16])


def test_layout(virtual, tmp_path):
    """A layout function chooses how each array is stored, or leaves its chunks as they are."""
    rfs, x = virtual
    seen = []

    def layout(path, array):
        seen.append((path, array.shape))
        if path == "acquisition/sync":
            return None
        return {"chunks": (1000, 4), "compressors": None}

    report = materialize(rfs, str(tmp_path / "real.zarr"), layout=layout)
    assert sorted(seen) == [("acquisition/signal", (50_000, 16)), ("acquisition/sync", (50_000,))]
    root = zarr.open_group(str(tmp_path / "real.zarr"), mode="r")
    signal, sync = root["acquisition/signal"], root["acquisition/sync"]
    assert signal.chunks == (1000, 4) and len(signal.metadata.codecs) == 1
    np.testing.assert_array_equal(signal[...], x[:, :16])
    # Copied as stored: the same chunks as the references had, holding only the selected bytes
    assert sync.chunks == open_rfs(rfs)["acquisition/sync"].chunks and "acquisition/sync" not in report
    np.testing.assert_array_equal(sync[...], x[:, 16])


def test_slabs_smaller_than_the_array(virtual, tmp_path):
    rfs, x = virtual
    materialize(rfs, str(tmp_path / "real.zarr"), chunk_bytes=2**16, slab_bytes=2**18, verify=True)
    np.testing.assert_array_equal(zarr.open_group(str(tmp_path / "real.zarr"))["acquisition/signal"][...], x[:, :16])


def test_virtual_nwb(tmp_path):
    """A virtual NWB file becomes an ordinary NWB Zarr file that hdmf-zarr reads from its directory."""
    pynwb = pytest.importorskip("pynwb")
    pytest.importorskip("hdmf_zarr.nwb")
    from hdmf_zarr import NWBZarrIO

    from zarrshadow.nwb import write_virtual_nwb

    x = np.cumsum(np.random.default_rng(1).integers(-3, 4, (40_000, 8)), axis=0).astype("<i2")
    (tmp_path / "raw.bin").write_bytes(x.tobytes())
    raw = VirtualArray.contiguous(str(tmp_path / "raw.bin"), shape=x.shape, dtype="<i2")
    nwbfile = pynwb.NWBFile(
        session_description="s", identifier="i", session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    nwbfile.add_acquisition(pynwb.TimeSeries(name="raw", data=raw.placeholder(), unit="V", rate=30000.0))
    nwbfile.add_acquisition(pynwb.TimeSeries(name="position", data=np.arange(10.0), unit="m", rate=1.0))
    rfs = write_virtual_nwb(nwbfile)

    materialize(rfs, str(tmp_path / "session.nwb.zarr"), chunk_bytes=2**16)
    (tmp_path / "raw.bin").unlink()
    with NWBZarrIO(str(tmp_path / "session.nwb.zarr"), mode="r") as io:
        read = io.read()
        data = read.acquisition["raw"].data
        np.testing.assert_array_equal(data[...], x)
        np.testing.assert_array_equal(read.acquisition["position"].data[...], np.arange(10.0))
        assert pynwb.validate(io=io) == []
    # hdmf-zarr keeps a copy of each array's metadata in the root, which describes the new chunks
    root = json.loads((tmp_path / "session.nwb.zarr" / "zarr.json").read_text())
    stored = root["consolidated_metadata"]["metadata"]["acquisition/raw/data"]
    assert stored["chunk_grid"]["configuration"]["chunk_shape"] == list(data.chunks)
    assert len(stored["codecs"]) == 2


def test_transposed_array(tmp_path):
    """What is written has the array's own axis order and no transpose codec."""
    x = np.random.default_rng(2).integers(0, 4000, (40, 6, 8)).astype("<u2")
    (tmp_path / "movie.bin").write_bytes(x.tobytes())
    movie = VirtualArray.contiguous(str(tmp_path / "movie.bin"), shape=x.shape, dtype="<u2", chunk_bytes=960)
    builder = RfsBuilder()
    builder.add_group("")
    movie.transpose(0, 2, 1).add_to(builder, "movie")
    materialize(builder.build(), str(tmp_path / "real.zarr"))
    written = zarr.open_group(str(tmp_path / "real.zarr"), mode="r")["movie"]
    assert written.shape == (40, 8, 6) and "transpose" not in json.dumps(written.metadata.to_dict()["codecs"])
    np.testing.assert_array_equal(written[...], x.transpose(0, 2, 1))
