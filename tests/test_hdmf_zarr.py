"""Read a zindi RFS of a pynwb-written file with hdmf-zarr's NWBZarrIO."""

import warnings
from datetime import datetime, timezone

import numpy as np
import pytest

from zindi import generate_rfs, write_rfs
from zindi.open_rfs import load_rfs
from zindi.rfs_store import RfsStore

pynwb = pytest.importorskip("pynwb")
hdmf_zarr_nwb = pytest.importorskip("hdmf_zarr.nwb")

from hdmf.backends.hdf5 import H5DataIO  # noqa: E402
from pynwb import NWBHDF5IO, NWBFile, TimeSeries  # noqa: E402
from pynwb.ecephys import ElectricalSeries  # noqa: E402
from pynwb.file import Subject  # noqa: E402


@pytest.fixture(scope="module")
def nwb_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("nwb") / "test.nwb"
    nwb = NWBFile(
        session_description="zindi test",
        identifier="abc",
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        subject=Subject(subject_id="m1", species="Mus musculus", age="P90D"),
        keywords=["a", "b"],
    )
    device = nwb.create_device("probe")
    group = nwb.create_electrode_group("shank0", description="d", location="CA1", device=device)
    for i in range(8):
        nwb.add_electrode(group=group, location="CA1", x=float(i), y=0.0, z=np.nan)
    data = np.random.default_rng(0).integers(-1000, 1000, (30000, 8)).astype("int16")
    es = ElectricalSeries(
        name="ElectricalSeries",
        data=H5DataIO(data, chunks=(1000, 8), compression="gzip"),
        electrodes=nwb.create_electrode_table_region(list(range(8)), "all"),
        rate=30000.0,
    )
    nwb.add_acquisition(es)
    # A TimeSeries whose data is a link to another dataset
    nwb.add_acquisition(
        TimeSeries(name="linked", data=es, unit="V", timestamps=np.arange(30000) / 30000.0)
    )
    nwb.add_unit_column("quality", "q")
    for u in range(5):
        nwb.add_unit(spike_times=np.sort(np.random.rand(10 + u)), quality="good", electrodes=[u])
    # The trials timeseries column is a compound dataset with a reference field
    nwb.add_trial(start_time=0.0, stop_time=1.0, timeseries=[es])
    nwb.add_trial(start_time=1.0, stop_time=2.0, timeseries=[es])
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return str(path)


@pytest.fixture(scope="module", params=["refs", "chunk_index", "directory", "contiguous_gen"])
def nwb_pair(request, nwb_path, tmp_path_factory):
    """Yield the file as read by NWBHDF5IO and by NWBZarrIO over the zindi RFS.

    The ElectricalSeries data has 30 chunks, so a threshold of 10 gives it a
    chunk index. "directory" writes that RFS to disk and reads it back.
    """
    if request.param == "refs":
        rfs = generate_rfs(nwb_path)
    elif request.param == "contiguous_gen":
        # linked/timestamps is a contiguous 240 KB dataset; split it into gen slabs
        rfs = generate_rfs(nwb_path, contiguous_chunk_bytes=16 * 1024)
        assert any(g["key"].startswith("acquisition/linked/timestamps/") for g in rfs["gen"])
    else:
        rfs = generate_rfs(nwb_path, chunk_index_threshold=10)
        assert "acquisition/ElectricalSeries/data" in rfs["indexes"]
    if request.param == "directory":
        out = str(tmp_path_factory.mktemp("rfs") / "test.zindi")
        write_rfs(rfs, out)
        rfs = load_rfs(out)
    with NWBHDF5IO(nwb_path, "r") as h5io:
        zio = hdmf_zarr_nwb.NWBZarrIO(RfsStore(rfs), mode="r")
        yield h5io.read(), zio.read()
        zio.close()


def test_no_dtype_inference_warnings(nwb_path):
    """hdmf-zarr warns when a dataset lacks _DTYPE and it has to infer one."""
    rfs = generate_rfs(nwb_path)
    assert rfs["refs"]["zarr.json"].count('".specloc":"specifications"') == 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with hdmf_zarr_nwb.NWBZarrIO(RfsStore(rfs), mode="r") as zio:
            zio.read()
    inferred = [w for w in caught if "Inferred dtype" in str(w.message)]
    assert not inferred, str(inferred[0].message)


def test_file_level_fields(nwb_pair):
    ref, got = nwb_pair
    assert got.session_description == ref.session_description
    assert got.session_start_time == ref.session_start_time
    assert list(got.keywords[:]) == list(ref.keywords[:])
    assert got.subject.age == ref.subject.age


def test_electrical_series(nwb_pair):
    ref, got = nwb_pair
    es_ref, es_got = ref.acquisition["ElectricalSeries"], got.acquisition["ElectricalSeries"]
    np.testing.assert_array_equal(es_got.data[:], es_ref.data[:])
    assert es_got.rate == es_ref.rate
    assert es_got.electrodes.data[:].tolist() == es_ref.electrodes.data[:].tolist()
    assert es_got.electrodes.table.name == "electrodes"


def test_electrodes_table(nwb_pair):
    ref, got = nwb_pair
    cols = ["location", "x"]
    assert (
        got.electrodes.to_dataframe()[cols].values.tolist()
        == ref.electrodes.to_dataframe()[cols].values.tolist()
    )
    assert got.electrodes["group"][0].name == "shank0"
    assert got.electrode_groups["shank0"].device.name == "probe"


def test_data_link(nwb_pair):
    ref, got = nwb_pair
    np.testing.assert_array_equal(
        got.acquisition["linked"].data[:10], ref.acquisition["linked"].data[:10]
    )


def test_units(nwb_pair):
    ref, got = nwb_pair
    assert [len(x) for x in got.units["spike_times"][:]] == [
        len(x) for x in ref.units["spike_times"][:]
    ]
    assert list(got.units["quality"][:]) == list(ref.units["quality"][:])


def test_trials_compound_reference(nwb_pair):
    _, got = nwb_pair
    assert got.trials["timeseries"][0][0][2].name == "ElectricalSeries"
    assert len(got.trials.to_dataframe()) == 2
