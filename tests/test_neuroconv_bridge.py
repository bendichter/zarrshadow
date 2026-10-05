"""The NeuroConv bridge: the NWB files NeuroConv builds, with their signals as references.

The comparisons with NeuroConv need NeuroConv, SpikeInterface, and Zarr v3 in
one environment. As of October 2026 that takes an override of SpikeInterface's
zarr<3 requirement, and NeuroConv 0.10 reads zarr.codec_registry at import,
which Zarr v3 removed, so it is put back here before NeuroConv is imported.
"""

import os
from datetime import datetime, timezone

import numpy as np
import pytest

pynwb = pytest.importorskip("pynwb")
pytest.importorskip("hdmf_zarr.nwb")

from hdmf.data_utils import DataChunkIterator  # noqa: E402
from hdmf_zarr import NWBZarrIO  # noqa: E402

from zindi import RfsStore, open_rfs  # noqa: E402
from zindi.neuroconv_bridge import NotVirtualizable, virtualize  # noqa: E402
from zindi.nwb import write_virtual_nwb  # noqa: E402


def _import_neuroconv():
    import numcodecs.registry
    import zarr

    if not hasattr(zarr, "codec_registry"):
        zarr.codec_registry = numcodecs.registry.codec_registry
    return pytest.importorskip("neuroconv.datainterfaces")


def test_iterator_that_reads_no_file():
    """An iterator over values in memory cannot be referenced: an error, or with strict off, a copy."""
    nwbfile = pynwb.NWBFile(
        session_description="s", identifier="i", session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    data = DataChunkIterator(np.arange(20.0).reshape(10, 2))
    nwbfile.add_acquisition(pynwb.TimeSeries(name="computed", data=data, unit="V", rate=1.0))
    with pytest.raises(NotVirtualizable, match="does not read a SpikeInterface recording"):
        virtualize(nwbfile)
    assert virtualize(nwbfile, strict=False) == {}
    with NWBZarrIO(RfsStore(write_virtual_nwb(nwbfile)), mode="r") as io:
        np.testing.assert_array_equal(io.read().acquisition["computed"].data[...], np.arange(20.0).reshape(10, 2))


CASES = {
    "SpikeGLX AP band": (
        "SpikeGLXRecordingInterface",
        {"folder_path": "spikeglx/Noise4Sam_g0", "stream_id": "imec0.ap"},
        "ElectricalSeriesAP/data",
        (1155, 384),
    ),
    "SpikeGLX NIDQ": (
        "SpikeGLXNIDQInterface",
        {"folder_path": "spikeglx/Noise4Sam_g0"},
        "TimeSeriesNIDQ/data",
        (60864, 8),
    ),
    "Open Ephys binary": (
        "OpenEphysBinaryRecordingInterface",
        {
            "folder_path": "openephysbinary/v0.5.3_two_neuropixels_stream/Record_Node_107",
            "stream_name": "Record_Node_107#Neuropix-PXI-116.0",
        },
        "ElectricalSeries/data",
        (30000, 384),
    ),
    "Open Ephys binary with a sync channel": (
        "OpenEphysBinaryRecordingInterface",
        {
            "folder_path": "openephysbinary/v0.6.x_neuropixels_with_sync",
            "stream_name": "Record Node 104#Neuropix-PXI-100.ProbeA-AP",
        },
        "ElectricalSeries/data",
        (1000, 384),
    ),
    "Neuroscope": ("NeuroScopeRecordingInterface", {"file_path": "neuroscope/test1/test1.dat"}, "ElectricalSeries/data", (10, 8)),
    "MCS raw": ("MCSRawRecordingInterface", {"file_path": "rawmcs/raw_mcs_with_header_1.raw"}, "ElectricalSeries/data", (100000, 60)),
}


@pytest.mark.gin
@pytest.mark.parametrize("case", CASES)
def test_same_as_neuroconv(case, tmp_path):
    """Every dataset of the virtual file equals the file NeuroConv writes for the same recording."""
    import h5py
    from test_neo_gin import _download

    datainterfaces = _import_neuroconv()
    interface_name, source, replaced_name, shape = CASES[case]
    source = {
        key: os.path.join(_download(value.split("/")[0]), value) if key.endswith("_path") else value
        for key, value in source.items()
    }
    interface = getattr(datainterfaces, interface_name)(**source)
    metadata = interface.get_metadata()
    metadata["NWBFile"].setdefault("session_start_time", datetime(2026, 1, 1, tzinfo=timezone.utc))

    interface.run_conversion(nwbfile_path=str(tmp_path / "copied.nwb"), metadata=metadata, overwrite=True)

    nwbfile = interface.create_nwbfile(metadata=metadata)
    replaced = virtualize(nwbfile)
    assert {name: array.shape for name, array in replaced.items()} == {replaced_name: shape}
    rfs = write_virtual_nwb(nwbfile)
    assert len(rfs["sources"]) == 1

    virtual = open_rfs(rfs)
    different, compared = [], 0

    def compare(name, obj):
        nonlocal compared
        if not isinstance(obj, h5py.Dataset) or name.startswith("specifications/"):
            return
        if h5py.check_ref_dtype(obj.dtype) is not None or obj.dtype.kind == "V":
            return  # references and tables of them, which each file encodes its own way
        copied, referenced = obj[()], virtual[name]
        referenced = np.asarray(referenced[...] if referenced.shape else referenced[()])
        if obj.dtype.kind in "iufb":
            same = np.array_equal(copied, referenced)
        else:
            strings = [v.decode() if isinstance(v, bytes) else str(v) for v in np.atleast_1d(copied).ravel()]
            same = strings == [str(v) for v in np.atleast_1d(referenced).ravel()]
        compared += 1
        if not same:
            different.append(name)

    with h5py.File(tmp_path / "copied.nwb", "r") as copied_file:
        copied_file.visititems(compare)
    assert different == ["file_create_date"] and compared > 10

    with NWBZarrIO(RfsStore(rfs), mode="r") as io:
        io.read()
        assert pynwb.validate(io=io) == []
