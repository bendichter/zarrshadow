"""Tests of the NEO generator against real recordings.

The recordings are NEO's testing data, hosted on GIN at
https://gin.g-node.org/NeuralEnsemble/ephy_testing_data. They are downloaded
with datalad to the folder NEO uses, ~/ephy_testing_data, or the folder named
by the EPHY_TESTING_DATA_FOLDER environment variable. The files listed for
each reader are the ones NEO tests that reader with.

These tests download about 300 MB and are not run by default. Run them with
``pytest -m gin``.
"""

import functools
import os
from pathlib import Path

import numpy as np
import pytest

neo_rawio = pytest.importorskip("neo.rawio")

from neo.rawio.baserawio import BaseRawWithBufferApiIO  # noqa: E402

from zindi import generate_rfs_neo, open_rfs  # noqa: E402

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


@functools.cache
def _download(folder):
    """Download one top-level folder of the testing data and return the data root.

    A complete local copy is used as it is, without contacting GIN, so that a
    cached copy still works when GIN refuses the connection. Set
    EPHY_TESTING_DATA_UPDATE=1 to bring an existing copy up to date.
    """
    from neo.utils.datasets import download_dataset, get_local_testing_data_folder

    root = Path(get_local_testing_data_folder())
    update = os.environ.get("EPHY_TESTING_DATA_UPDATE", "") not in ("", "0")
    if update or not _is_complete(root / folder):
        pytest.importorskip("datalad")
        download_dataset(remote_path=folder)
    return str(root)


def _is_complete(folder):
    """Whether a folder exists and holds the content of every file in it.

    A file whose content has not been downloaded is a broken link.
    """
    return folder.is_dir() and all(path.exists() for path in folder.rglob("*"))


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
