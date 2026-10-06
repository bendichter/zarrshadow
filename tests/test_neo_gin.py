"""Tests of the NEO generator against real recordings.

The recordings are NEO's testing data, hosted on GIN at
https://gin.g-node.org/NeuralEnsemble/ephy_testing_data. They are downloaded
with datalad to the folder NEO uses, ~/ephy_testing_data, or the folder named
by the EPHY_TESTING_DATA_FOLDER environment variable; see gin_data.py. The files listed for
each reader are the ones NEO tests that reader with.

These tests download about 300 MB and are not run by default. Run them with
``pytest -m gin``.
"""

import os

import numpy as np
import pytest

neo_rawio = pytest.importorskip("neo.rawio")

from neo.rawio.baserawio import BaseRawWithBufferApiIO  # noqa: E402

from zarrshadow import generate_rfs_neo, open_rfs  # noqa: E402

pytestmark = pytest.mark.gin

FILES = {
    "AxonRawIO": [
        "axon/File_axon_1.abf",
        "axon/File_axon_2.abf",
        "axon/File_axon_3.abf",
        "axon/File_axon_4.abf",
        "axon/File_axon_5.abf",
        "axon/File_axon_6.abf",
        "axon/File_axon_7.abf",
        "axon/test_file_edr3.abf",
    ],
    "BrainVisionRawIO": [
        "brainvision/File_brainvision_1.vhdr",
        "brainvision/File_brainvision_2.vhdr",
        "brainvision/File_brainvision_3_float32.vhdr",
        "brainvision/File_brainvision_3_int16.vhdr",
        "brainvision/File_brainvision_3_int32.vhdr",
        "brainvision/File_brainvision_4_float32.vhdr",
    ],
    "ElanRawIO": ["elan/File_elan_1.eeg"],
    "MicromedRawIO": [
        "micromed/File_micromed_1.TRC",
        "micromed/File_mircomed2.TRC",
        "micromed/File_mircomed2_2segments.TRC",
    ],
    "NeuroNexusRawIO": ["neuronexus/allego_1/allego_2__uid0701-13-04-49.xdat.json"],
    "NeuroScopeRawIO": [
        "neuroscope/test1/test1",
        "neuroscope/test1/test1.dat",
        "neuroscope/dataset_1/YutaMouse42-151117.eeg",
    ],
    "OpenEphysBinaryRawIO": [
        "openephysbinary/v0.5.3_two_neuropixels_stream",
        "openephysbinary/v0.4.4.1_with_video_tracking",
        "openephysbinary/v0.5.x_two_nodes",
        "openephysbinary/v0.6.x_neuropixels_multiexp_multistream",
        "openephysbinary/v0.6.x_neuropixels_with_sync",
        "openephysbinary/v0.6.x_neuropixels_missing_folders",
        "openephysbinary/v0.6.x_onebox_neuropixels",
        "openephysbinary/neural_and_non_neural_data_mixed",
    ],
    "RawBinarySignalRawIO": ["rawbinarysignal/File_rawbinary_10kHz_2channels_16bit.raw"],
    "RawMCSRawIO": ["rawmcs/raw_mcs_with_header_1.raw"],
    "SpikeGLXRawIO": [
        "spikeglx/Noise4Sam_g0",
        "spikeglx/TEST_20210920_0_g0",
        "spikeglx/multi_trigger_multi_gate/SpikeGLX/5-19-2022-CI0/5-19-2022-CI0_g0",
        "spikeglx/multi_trigger_multi_gate/SpikeGLX/5-19-2022-CI0/5-19-2022-CI0_g1",
        "spikeglx/multi_trigger_multi_gate/SpikeGLX/5-19-2022-CI0",
        "spikeglx/multi_trigger_multi_gate/SpikeGLX/5-19-2022-CI1",
        "spikeglx/multi_trigger_multi_gate/SpikeGLX/5-19-2022-CI2",
        "spikeglx/multi_trigger_multi_gate/SpikeGLX/5-19-2022-CI3",
        "spikeglx/multi_trigger_multi_gate/SpikeGLX/5-19-2022-CI4",
        "spikeglx/multi_trigger_multi_gate/SpikeGLX/5-19-2022-CI5",
        "spikeglx/NP2_with_sync",
        "spikeglx/NP2_no_sync",
        "spikeglx/NP2_subset_with_sync",
        "spikeglx/np_ultra_stub",
        "spikeglx/multi_probe_multi_dock_multi_shank_filename_without_info",
        "spikeglx/multi_trigger_multi_gate/CatGT/CatGT-A",
        "spikeglx/multi_trigger_multi_gate/CatGT/CatGT-B",
        "spikeglx/multi_trigger_multi_gate/CatGT/CatGT-C",
        "spikeglx/multi_trigger_multi_gate/CatGT/CatGT-D",
        "spikeglx/multi_trigger_multi_gate/CatGT/CatGT-E",
        "spikeglx/multi_trigger_multi_gate/CatGT/Supercat-A",
        "spikeglx/onebox/run_with_only_adc",
    ],
    "WinEdrRawIO": [
        "winedr/File_WinEDR_1.EDR",
        "winedr/File_WinEDR_2.EDR",
        "winedr/File_WinEDR_3.EDR",
    ],
    "WinWcpRawIO": [
        "winwcp/File_winwcp_1.wcp",
        "winwcp/file_with_recording_time/File_winwcp_2.wcp",
    ],
}

# Maxwell stores its signals in HDF5 with a proprietary compression filter,
# which has no Zarr codec, so its files cannot be referenced.
UNSUPPORTED_FILES = {
    "MaxwellRawIO": [
        "maxwell/MaxOne_data/Record/000011/data.raw.h5",
        "maxwell/MaxTwo_data/Network/000028/data.raw.h5",
    ],
}


def _cases(files):
    return [pytest.param(name, path, id=path) for name, paths in files.items() for path in paths]


def _download(folder):
    """Download one top-level folder of the testing data and return the data root."""
    from gin_data import fetch

    path = fetch("ephys", folder)
    return path[: -len(folder)].rstrip("/")


def _reader(name, path):
    cls = getattr(neo_rawio, name)
    local = os.path.join(_download(path.split("/")[0]), path)
    reader = cls(dirname=local) if cls.rawmode == "one-dir" else cls(filename=local)
    reader.parse_header()
    return reader


def _columns(spec):
    if spec is None:
        return slice(None)
    if isinstance(spec, dict):
        return slice(spec["start"], spec["stop"], spec["step"])
    return spec


@pytest.mark.parametrize(("name", "path"), _cases(FILES))
def test_signals_match_neo(name, path):
    """Every stream of every segment reads the same through the references as through NEO."""
    reader = _reader(name, path)
    root = open_rfs(generate_rfs_neo(reader))
    single = reader.block_count() == 1 and reader.segment_count(0) == 1
    compared = 0
    for block in range(reader.block_count()):
        for seg in range(reader.segment_count(block)):
            for stream_index, stream in enumerate(reader.header["signal_streams"]):
                buffer_id = str(stream["buffer_id"]).replace("/", "_")
                arr = root[buffer_id if single else f"block{block}/segment{seg}/{buffer_id}"]
                (info,) = [s for s in arr.attrs["neo"]["streams"] if s["id"] == str(stream["id"])]
                expected = reader.get_analogsignal_chunk(
                    block_index=block, seg_index=seg, stream_index=stream_index
                )
                data = np.asarray(arr[...])
                if arr.attrs["neo"]["time_axis"] == 1:
                    data = data.T
                np.testing.assert_array_equal(data[:, _columns(info["columns"])], expected)
                compared += 1
    assert compared > 0


@pytest.mark.parametrize(("name", "path"), _cases(UNSUPPORTED_FILES))
def test_unsupported_compression(name, path):
    with pytest.raises(RuntimeError, match="filter id 401"):
        generate_rfs_neo(_reader(name, path))


def test_every_reader_with_buffer_api_is_listed():
    """A reader that gains the buffer description API in a new NEO release needs files here."""
    readers = {cls.__name__ for cls in neo_rawio.rawiolist if issubclass(cls, BaseRawWithBufferApiIO)}
    assert readers == set(FILES) | set(UNSUPPORTED_FILES)


def _stream_key(reader, block, seg, stream):
    name = str(stream["id"]).replace("/", "_")
    single = reader.block_count() == 1 and reader.segment_count(0) == 1
    return name if single else f"block{block}/segment{seg}/{name}"


@pytest.mark.parametrize(("name", "path"), _cases(FILES))
def test_stream_arrays_match_neo(name, path):
    """Each stream as its own array, holding only its channels, reads the same as through NEO."""
    from zarrshadow import RfsBuilder, virtual_arrays_neo

    reader = _reader(name, path)
    arrays = virtual_arrays_neo(reader, chunk_bytes=2**20)
    builder = RfsBuilder()
    builder.add_group("")
    for i, array in enumerate(arrays.values()):
        array.add_to(builder, f"a{i}", dimension_names=["time", "channel"])
    root = open_rfs(builder.build())
    paths = {key: f"a{i}" for i, key in enumerate(arrays)}
    compared = 0
    for block in range(reader.block_count()):
        for seg in range(reader.segment_count(block)):
            for stream_index, stream in enumerate(reader.header["signal_streams"]):
                key = _stream_key(reader, block, seg, stream)
                expected = reader.get_analogsignal_chunk(block_index=block, seg_index=seg, stream_index=stream_index)
                assert arrays[key].shape == expected.shape
                np.testing.assert_array_equal(root[paths[key]][...], expected)
                channels = reader.header["signal_channels"]
                n_channels = int((channels["stream_id"] == stream["id"]).sum())
                assert len(arrays[key].attributes["channel_ids"]) == n_channels == expected.shape[1]
                compared += 1
    assert compared == len(arrays) > 0


def test_spikeglx_as_virtual_nwb(tmp_path):
    """A SpikeGLX recording as an NWB file: the neural channels and the sync channel of one file, as two series."""
    pynwb = pytest.importorskip("pynwb")
    pytest.importorskip("hdmf_zarr.nwb")
    from datetime import datetime, timezone

    from hdmf_zarr import NWBZarrIO
    from pynwb.ecephys import ElectricalSeries

    from zarrshadow import RfsStore, materialize, virtual_arrays_neo
    from zarrshadow.nwb import write_virtual_nwb

    reader = _reader("SpikeGLXRawIO", "spikeglx/Noise4Sam_g0")
    arrays = virtual_arrays_neo(reader)
    ap, sync = arrays["imec0.ap"], arrays["imec0.ap-SYNC"]
    assert ap.shape[1] == 384 and sync.shape[1] == 1

    nwbfile = pynwb.NWBFile(
        session_description="SpikeGLX test recording",
        identifier="Noise4Sam_g0",
        session_start_time=datetime(2020, 11, 3, tzinfo=timezone.utc),
    )
    device = nwbfile.create_device("Neuropixels")
    group = nwbfile.create_electrode_group("imec0", description="probe", location="unknown", device=device)
    for _ in range(ap.shape[1]):
        nwbfile.add_electrode(group=group, location="unknown")
    nwbfile.add_acquisition(
        ElectricalSeries(
            name="ElectricalSeriesAP",
            data=ap.placeholder(),
            electrodes=nwbfile.create_electrode_table_region(list(range(ap.shape[1])), "all electrodes"),
            rate=ap.attributes["sampling_rate"],
            starting_time=ap.attributes["t_start"],
            conversion=1e-6,
            channel_conversion=ap.attributes["gain"],
        )
    )
    nwbfile.add_acquisition(
        pynwb.TimeSeries(name="sync", data=sync.placeholder(), unit="a.u.", rate=sync.attributes["sampling_rate"])
    )
    rfs = write_virtual_nwb(nwbfile)

    # Both series point into the one .ap.bin file, each keeping its own bytes of every sample
    assert len(rfs["sources"]) == 1
    assert rfs["selections"] == {
        "acquisition/ElectricalSeriesAP/data": {"record_size": 770, "keep": [[0, 768]]},
        "acquisition/sync/data": {"record_size": 770, "keep": [[768, 770]]},
    }
    streams = [str(s) for s in reader.header["signal_streams"]["id"]]
    with NWBZarrIO(RfsStore(rfs), mode="r") as io:
        read = io.read()
        for series, stream in (("ElectricalSeriesAP", "imec0.ap"), ("sync", "imec0.ap-SYNC")):
            expected = reader.get_analogsignal_chunk(block_index=0, seg_index=0, stream_index=streams.index(stream))
            np.testing.assert_array_equal(read.acquisition[series].data[...], expected)
        assert pynwb.validate(io=io) == []

    # Materialized, it is an ordinary NWB Zarr file, compressed, that no longer needs the recording
    report = materialize(rfs, str(tmp_path / "session.nwb.zarr"))
    ap_report = report["acquisition/ElectricalSeriesAP/data"]
    assert ap_report["nbytes"] == ap.shape[0] * 384 * 2 and ap_report["nbytes_stored"] < ap_report["nbytes"]
    with NWBZarrIO(str(tmp_path / "session.nwb.zarr"), mode="r") as io:
        read = io.read()
        for series, stream in (("ElectricalSeriesAP", "imec0.ap"), ("sync", "imec0.ap-SYNC")):
            expected = reader.get_analogsignal_chunk(block_index=0, seg_index=0, stream_index=streams.index(stream))
            np.testing.assert_array_equal(read.acquisition[series].data[...], expected)
        assert pynwb.validate(io=io) == []
