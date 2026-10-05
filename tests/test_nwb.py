"""Virtual NWB files: hdmf-zarr writes the structure, and the data stays in the source files."""

import json
from datetime import datetime, timezone

import numpy as np
import pytest

pynwb = pytest.importorskip("pynwb")
hdmf_zarr = pytest.importorskip("hdmf_zarr.nwb")

from hdmf_zarr import NWBZarrIO  # noqa: E402
from pynwb import NWBFile, TimeSeries  # noqa: E402
from pynwb.ecephys import ElectricalSeries  # noqa: E402

from zindi import RfsStore, VirtualArray, load_rfs, stack  # noqa: E402
from zindi.nwb import write_virtual_nwb  # noqa: E402


def _nwbfile(n_electrodes=0):
    nwbfile = NWBFile(
        session_description="a virtual file",
        identifier="virtual",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    if n_electrodes:
        device = nwbfile.create_device("probe")
        group = nwbfile.create_electrode_group("shank0", description="a shank", location="CA1", device=device)
        for _ in range(n_electrodes):
            nwbfile.add_electrode(group=group, location="CA1")
    return nwbfile


@pytest.fixture
def sources(tmp_path):
    """A recording of 16 channels and a sync channel in one file, and four more channels in a file each."""
    rng = np.random.default_rng(0)
    x = rng.integers(-500, 500, (10_000, 17)).astype("<i2")
    (tmp_path / "raw.bin").write_bytes(b"\0" * 12 + x.tobytes())
    raw = VirtualArray.contiguous(str(tmp_path / "raw.bin"), shape=x.shape, dtype="<i2", offset=12, chunk_bytes=34_000)
    y = rng.integers(-500, 500, (5000, 4)).astype(">i2")
    channels = []
    for j in range(4):
        path = tmp_path / f"channel{j}.bin"
        path.write_bytes(b"H" * 8 + y[:, j].tobytes())
        channels.append(VirtualArray.contiguous(str(path), shape=[len(y)], dtype=">i2", offset=8, chunk_bytes=4000))
    return raw, x, channels, y


def test_virtual_nwb(sources, tmp_path):
    raw, x, channels, y = sources
    nwbfile = _nwbfile(n_electrodes=16)
    region = nwbfile.create_electrode_table_region(list(range(16)), "all electrodes")
    nwbfile.add_acquisition(
        ElectricalSeries(
            name="ElectricalSeries", data=raw[:, :16].placeholder(), electrodes=region, rate=30000.0, conversion=1e-6
        )
    )
    nwbfile.add_acquisition(TimeSeries(name="sync", data=raw[:, 16].placeholder(), unit="a.u.", rate=30000.0))
    nwbfile.add_acquisition(TimeSeries(name="lfp", data=stack(channels, axis=1).placeholder(), unit="V", rate=1000.0))
    nwbfile.add_acquisition(TimeSeries(name="position", data=np.arange(10.0), unit="m", rate=1.0))

    rfs = write_virtual_nwb(nwbfile, str(tmp_path / "session.nwb.zindi"))

    # The signals are references into the five source files, and nothing of them is stored
    assert len(rfs["sources"]) == 5
    assert rfs["selections"] == {
        "acquisition/ElectricalSeries/data": {"record_size": 34, "keep": [[0, 32]]},
        "acquisition/sync/data": {"record_size": 34, "keep": [[32, 34]]},
    }
    signal_keys = [k for k in rfs["refs"] if "/data/c/" in k and "position" not in k]
    assert all(isinstance(rfs["refs"][k], list) for k in signal_keys)
    assert len(json.dumps(rfs)) < 300_000

    for opened in (rfs, load_rfs(str(tmp_path / "session.nwb.zindi"))):
        with NWBZarrIO(RfsStore(opened), mode="r") as io:
            read = io.read()
            series = read.acquisition["ElectricalSeries"]
            assert series.data.shape == (10_000, 16) and series.rate == 30000.0 and series.conversion == 1e-6
            np.testing.assert_array_equal(series.data[...], x[:, :16])
            np.testing.assert_array_equal(series.data[2500:2600, 3], x[2500:2600, 3])
            assert list(series.electrodes.data[:]) == list(range(16))
            assert series.electrodes.table is read.electrodes
            np.testing.assert_array_equal(read.acquisition["sync"].data[...], x[:, 16])
            np.testing.assert_array_equal(read.acquisition["lfp"].data[...], y)
            np.testing.assert_array_equal(read.acquisition["position"].data[...], np.arange(10.0))
            assert pynwb.validate(io=io) == []


def test_reading_fetches_no_signal(sources, monkeypatch):
    """Opening the file and looking at a dataset's shape reads none of the source files."""
    import zindi.rfs_store

    raw, x, _, _ = sources
    nwbfile = _nwbfile()
    nwbfile.add_acquisition(TimeSeries(name="raw", data=raw.placeholder(), unit="V", rate=30000.0))
    rfs = write_virtual_nwb(nwbfile)

    reads = []
    read_bytes = zindi.rfs_store._read_bytes_from_url_or_path

    def logged(url_or_path, offset, length, **kwargs):
        reads.append((offset, length))
        return read_bytes(url_or_path, offset, length, **kwargs)

    monkeypatch.setattr(zindi.rfs_store, "_read_bytes_from_url_or_path", logged)
    with NWBZarrIO(RfsStore(rfs), mode="r") as io:
        data = io.read().acquisition["raw"].data
        assert data.shape == (10_000, 17) and reads == []
        np.testing.assert_array_equal(data[1500:1510], x[1500:1510])
        assert len(reads) == 1


def test_single_byte_values(tmp_path):
    """Single bytes have no byte order, which zarr writes differently in the array's metadata."""
    x = np.random.default_rng(1).integers(0, 255, (300, 4, 5)).astype("u1")
    (tmp_path / "movie.bin").write_bytes(x.tobytes())
    movie = VirtualArray.contiguous(str(tmp_path / "movie.bin"), shape=x.shape, dtype="u1", chunk_bytes=2000)
    nwbfile = _nwbfile()
    nwbfile.add_acquisition(TimeSeries(name="movie", data=movie.placeholder(), unit="a.u.", rate=30.0))
    with NWBZarrIO(RfsStore(write_virtual_nwb(nwbfile)), mode="r") as io:
        np.testing.assert_array_equal(io.read().acquisition["movie"].data[...], x)


def test_file_without_virtual_arrays():
    nwbfile = _nwbfile()
    nwbfile.add_acquisition(TimeSeries(name="position", data=np.arange(10.0), unit="m", rate=1.0))
    rfs = write_virtual_nwb(nwbfile)
    assert rfs["version"] == 1 and "sources" in rfs and rfs["sources"] == {}
    with NWBZarrIO(RfsStore(rfs), mode="r") as io:
        np.testing.assert_array_equal(io.read().acquisition["position"].data[...], np.arange(10.0))
