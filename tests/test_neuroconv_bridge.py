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

from zarrshadow import RfsStore, open_rfs  # noqa: E402
from zarrshadow.neuroconv_bridge import NotVirtualizable, virtualize  # noqa: E402
from zarrshadow.nwb import write_virtual_nwb  # noqa: E402


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


class _Pages:
    """What the bridge reads from one of roiextractors' TIFF extractors: a table of the page that holds each frame."""

    def __init__(self, file_paths, rows, shape, num_planes=1):
        table = np.zeros(len(rows), dtype=[("file_index", "u4"), ("IFD_index", "u4")])
        table["file_index"], table["IFD_index"] = zip(*rows)
        self._file_paths, self._frames_to_ifd_table = [str(p) for p in file_paths], table
        self.is_volumetric = num_planes > 1
        self._shape, self._num_planes = shape, num_planes

    def get_num_samples(self):
        return len(self._frames_to_ifd_table) // self._num_planes

    def get_num_planes(self):
        return self._num_planes

    def get_frame_shape(self):
        return self._shape


def _imaging_array(extractor, maxshape, dtype):
    from types import SimpleNamespace

    from zarrshadow.neuroconv_bridge import _array_for_imaging

    iterator = SimpleNamespace(imaging_extractor=extractor, maxshape=maxshape, dtype=np.dtype(dtype))
    return _array_for_imaging(iterator, None)


def _read(virtual):
    from zarrshadow import RfsBuilder

    builder = RfsBuilder()
    builder.add_group("")
    virtual.add_to(builder, "data")
    return open_rfs(builder.build())["data"][...]


def test_imaging_frames_from_tiff_pages(tmp_path):
    """Frames are picked out of the pages of several files, and come out as NeuroConv writes them."""
    tifffile = pytest.importorskip("tifffile")
    x = np.random.default_rng(0).integers(0, 4000, (2, 12, 48, 64)).astype("uint16")
    for i in range(2):
        tifffile.imwrite(tmp_path / f"part{i}.tif", x[i])
    paths = [tmp_path / "part0.tif", tmp_path / "part1.tif"]

    # one channel of two that alternate page by page, across both files
    rows = [(f, page) for f in range(2) for page in range(1, 12, 2)]
    array = _imaging_array(_Pages(paths, rows, (48, 64)), (12, 64, 48), "uint16")
    expected = np.concatenate([x[0, 1::2], x[1, 1::2]])
    np.testing.assert_array_equal(_read(array), expected.transpose(0, 2, 1))

    # volumes of three planes: (frame, width, height, plane)
    rows = [(f, page) for f in range(2) for page in range(12)]
    volumes = _imaging_array(_Pages(paths, rows, (48, 64), num_planes=3), (8, 64, 48, 3), "uint16")
    expected = x.reshape(8, 3, 48, 64).transpose(0, 3, 2, 1)
    np.testing.assert_array_equal(_read(volumes), expected)


def test_imaging_frames_from_several_extractors(tmp_path):
    """Extractors one after another in time, as for Bruker's one file per frame."""
    from types import SimpleNamespace

    tifffile = pytest.importorskip("tifffile")
    x = np.random.default_rng(1).integers(0, 4000, (5, 20, 30)).astype("uint16")
    parts = []
    for i in range(5):
        tifffile.imwrite(tmp_path / f"frame{i}.tif", x[i])
        parts.append(
            SimpleNamespace(
                file_path=tmp_path / f"frame{i}.tif", get_num_samples=lambda: 1, get_image_shape=lambda: (20, 30)
            )
        )
    extractor = SimpleNamespace(_imaging_extractors=parts, is_volumetric=False, get_frame_shape=lambda: (20, 30))
    array = _imaging_array(extractor, (5, 30, 20), "uint16")
    np.testing.assert_array_equal(_read(array), x.transpose(0, 2, 1))


def test_imaging_frames_that_cannot_be_referenced(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("imagecodecs")
    x = np.random.default_rng(2).integers(0, 4000, (4, 48, 64)).astype("uint16")
    with tifffile.TiffWriter(tmp_path / "compressed.tif") as writer:
        for frame in x:
            writer.write(frame, compression="zlib", contiguous=False)
    tifffile.imwrite(tmp_path / "plain.tif", x, photometric="minisblack")
    rows = [(0, page) for page in range(4)]

    with pytest.raises(NotVirtualizable, match="compressed or stored in several pieces"):
        _imaging_array(_Pages([tmp_path / "compressed.tif"], rows, (48, 64)), (4, 64, 48), "uint16")
    # an extractor that takes part of each page
    with pytest.raises(NotVirtualizable, match="part of a page"):
        _imaging_array(_Pages([tmp_path / "plain.tif"], rows, (40, 64)), (4, 64, 40), "uint16")
    with pytest.raises(NotVirtualizable, match="has no page 7"):
        _imaging_array(_Pages([tmp_path / "plain.tif"], [(0, 7)], (48, 64)), (1, 64, 48), "uint16")
    with pytest.raises(NotVirtualizable, match="NeuroConv writes"):
        _imaging_array(_Pages([tmp_path / "plain.tif"], rows, (48, 64)), (4, 64, 48), "float32")


def test_imaging_frames_from_hdf5(tmp_path):
    """roiextractors reads channel 0 of a (channel, frame, rows, columns) dataset, here big-endian."""
    from types import SimpleNamespace

    h5py = pytest.importorskip("h5py")
    x = np.random.default_rng(3).integers(0, 4000, (2, 10, 12, 16)).astype(">u2")
    with h5py.File(tmp_path / "movie.h5", "w") as f:
        f.create_dataset("mov", data=x)
        f.create_dataset("chunked", data=x, chunks=(1, 1, 12, 16))
    extractor = SimpleNamespace(filepath=tmp_path / "movie.h5", _mov_field="mov")
    array = _imaging_array(extractor, (10, 16, 12), ">u2")
    np.testing.assert_array_equal(_read(array), x[0].transpose(0, 2, 1))

    extractor._mov_field = "chunked"
    with pytest.raises(NotVirtualizable, match="chunked"):
        _imaging_array(extractor, (10, 16, 12), ">u2")


EPHYS, OPHYS = "ephys", "ophys"

# Each case: the testing dataset, the NeuroConv interface, its arguments (paths are relative to the dataset),
# the dataset that virtualize replaces with its shape, and how many source files the references point into.
CASES = {
    "SpikeGLX AP band": (
        EPHYS,
        "SpikeGLXRecordingInterface",
        {"folder_path": "spikeglx/Noise4Sam_g0", "stream_id": "imec0.ap"},
        ("ElectricalSeriesAP/data", (1155, 384)),
        1,
    ),
    "SpikeGLX NIDQ": (
        EPHYS,
        "SpikeGLXNIDQInterface",
        {"folder_path": "spikeglx/Noise4Sam_g0"},
        ("TimeSeriesNIDQ/data", (60864, 8)),
        1,
    ),
    "Open Ephys binary": (
        EPHYS,
        "OpenEphysBinaryRecordingInterface",
        {
            "folder_path": "openephysbinary/v0.5.3_two_neuropixels_stream/Record_Node_107",
            "stream_name": "Record_Node_107#Neuropix-PXI-116.0",
        },
        ("ElectricalSeries/data", (30000, 384)),
        1,
    ),
    "Open Ephys binary with a sync channel": (
        EPHYS,
        "OpenEphysBinaryRecordingInterface",
        {
            "folder_path": "openephysbinary/v0.6.x_neuropixels_with_sync",
            "stream_name": "Record Node 104#Neuropix-PXI-100.ProbeA-AP",
        },
        ("ElectricalSeries/data", (1000, 384)),
        1,
    ),
    "Neuroscope": (
        EPHYS,
        "NeuroScopeRecordingInterface",
        {"file_path": "neuroscope/test1/test1.dat"},
        ("ElectricalSeries/data", (10, 8)),
        1,
    ),
    "MCS raw": (
        EPHYS,
        "MCSRawRecordingInterface",
        {"file_path": "rawmcs/raw_mcs_with_header_1.raw"},
        ("ElectricalSeries/data", (100000, 60)),
        1,
    ),
    "TIFF stack": (
        OPHYS,
        "TiffImagingInterface",
        {"file_path": "imaging_datasets/Tif/demoMovie.tif", "sampling_frequency": 30.0},
        ("MicroscopySeries/data", (2000, 80, 60)),
        1,
    ),
    "ScanImage": (
        OPHYS,
        "ScanImageImagingInterface",
        {"file_path": "imaging_datasets/ScanImage/scanimage_20220801_single.tif"},
        ("TwoPhotonSeries/data", (3, 1024, 1024)),
        1,
    ),
    "ScanImage, one of two channels": (
        OPHYS,
        "ScanImageImagingInterface",
        {
            "file_path": "imaging_datasets/ScanImage/planar_two_channels_single_file/planar_two_ch_single_files_00001_00001.tif",
            "channel_name": "Channel 2",
        },
        ("TwoPhotonSeriesChannel2/data", (1000, 20, 20)),
        1,
    ),
    "ScanImage volumes": (
        OPHYS,
        "ScanImageImagingInterface",
        {
            "file_path": "imaging_datasets/ScanImage/volumetric_single_channel_single_file/vol_one_ch_single_files_00002_00001.tif"
        },
        ("TwoPhotonSeries/data", (100, 20, 20, 9)),
        1,
    ),
    "ScanImage volumes, one slice sample of one channel": (
        OPHYS,
        "ScanImageImagingInterface",
        {
            "file_path": "imaging_datasets/ScanImage/scanimage_20220923_roi.tif",
            "channel_name": "Channel 1",
            "slice_sample": 1,
        },
        ("TwoPhotonSeriesChannel1/data", (3, 256, 528, 2)),
        1,
    ),
    "HDF5 movie": (
        OPHYS,
        "Hdf5ImagingInterface",
        {"file_path": "imaging_datasets/hdf5/demoMovie.hdf5"},
        ("MicroscopySeries/data", (2000, 80, 60)),
        1,
    ),
    "Bruker, one file per frame": (
        OPHYS,
        "BrukerTiffSinglePlaneImagingInterface",
        {"folder_path": "imaging_datasets/BrukerTif/NCCR32_2022_11_03_IntoTheVoid_t_series-005"},
        ("TwoPhotonSeriesCh2/data", (10, 64, 64)),
        10,
    ),
    "Thor, three files": (
        OPHYS,
        "ThorImagingInterface",
        {"file_path": "imaging_datasets/ThorlabsTiff/single_channel_single_plane/20231018-002/ChanA_001_001_001_001.tif"},
        ("TwoPhotonSeriesDefault/data", (3, 512, 512)),
        3,
    ),
    "Micro-Manager, three files": (
        OPHYS,
        "MicroManagerTiffImagingInterface",
        {"folder_path": "imaging_datasets/MicroManagerTif/TS12_20220407_20hz_noteasy_1"},
        ("MicroscopySeries/data", (15, 1024, 1024)),
        3,
    ),
}

# Interfaces whose data cannot be referenced, and what the error says
REFUSED = {
    "Thor, LZW compressed": (
        "ThorImagingInterface",
        {
            "file_path": "imaging_datasets/ThorlabsTiff/multi_channel_multi_plane/lzw_compressed/ChanA_0001_0001_0001_0001.tif",
            "channel_name": "ChanA",
        },
        "compressed or stored in several pieces",
    ),
    "Bruker volumes": (
        "BrukerTiffMultiPlaneImagingInterface",
        {"folder_path": "imaging_datasets/BrukerTif/NCCR32_2022_11_03_IntoTheVoid_t_series-005"},
        "assembles volumes from several extractors",
    ),
}


def _local_source(dataset, source):
    """An interface's arguments with its paths downloaded and made absolute.

    A file is downloaded with the rest of its folder, since a recording is often several files. In the ephys
    dataset that is the format's whole folder, which the other GIN tests use too.
    """
    from gin_data import fetch, folder

    local = {}
    for key, value in source.items():
        if not key.endswith("_path"):
            local[key] = value
            continue
        if dataset == EPHYS:
            fetch(dataset, value.split("/")[0])
        else:
            fetch(dataset, value if key == "folder_path" else os.path.dirname(value))
        local[key] = os.path.join(folder(dataset), value)
    return local


@pytest.mark.gin
@pytest.mark.parametrize("case", CASES)
def test_same_as_neuroconv(case, tmp_path):
    """Every dataset of the virtual file equals the file NeuroConv writes for the same recording."""
    import h5py

    datainterfaces = _import_neuroconv()
    dataset, interface_name, source, replaced, n_sources = CASES[case]
    interface = getattr(datainterfaces, interface_name)(**_local_source(dataset, source))
    metadata = interface.get_metadata()
    metadata["NWBFile"].setdefault("session_start_time", datetime(2026, 1, 1, tzinfo=timezone.utc))

    interface.run_conversion(nwbfile_path=str(tmp_path / "copied.nwb"), metadata=metadata, overwrite=True)

    nwbfile = interface.create_nwbfile(metadata=metadata)
    virtual_arrays = virtualize(nwbfile)
    assert [(name, array.shape) for name, array in virtual_arrays.items()] == [replaced]
    rfs = write_virtual_nwb(nwbfile)
    assert len(rfs["sources"]) == n_sources

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
            same = np.array_equal(copied, referenced, equal_nan=obj.dtype.kind == "f")
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


@pytest.mark.gin
@pytest.mark.parametrize("case", REFUSED)
def test_refused(case):
    """Data that cannot be referenced raises, and with strict off is left for NeuroConv to copy."""
    datainterfaces = _import_neuroconv()
    interface_name, source, message = REFUSED[case]
    interface = getattr(datainterfaces, interface_name)(**_local_source(OPHYS, source))
    metadata = interface.get_metadata()
    metadata["NWBFile"].setdefault("session_start_time", datetime(2026, 1, 1, tzinfo=timezone.utc))
    nwbfile = interface.create_nwbfile(metadata=metadata)
    with pytest.raises(NotVirtualizable, match=message):
        virtualize(nwbfile)
    assert virtualize(nwbfile, strict=False) == {}
