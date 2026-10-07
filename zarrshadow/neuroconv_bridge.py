"""Make the NWB files NeuroConv builds virtual.

NeuroConv builds an NWBFile in memory in which each large dataset is an
iterator that reads the source file when the NWB file is written. virtualize
swaps those iterators for references, so that nothing is read:

    nwbfile = interface.create_nwbfile(metadata=interface.get_metadata())
    virtualize(nwbfile)
    write_virtual_nwb(nwbfile, "session.nwb.zarrshadow")

Everything else in the file, the metadata and the tables NeuroConv builds, is
written as NeuroConv made it. This covers recordings that SpikeInterface
reads through a NEO reader that virtual_arrays_neo supports: SpikeGLX, Open
Ephys binary, Neuroscope, Blackrock, SpikeGadgets, and others.

Experimental. As of October 2026 SpikeInterface requires zarr<3 and NeuroConv
0.10 reads an attribute at import that Zarr v3 removed, so NeuroConv and
zarrshadow install together only with the zarr requirement overridden; see
docs/virtual-nwb-plan.md.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .neo_rawio import virtual_arrays_neo
from .virtual import VirtualArray


class NotVirtualizable(NotImplementedError):
    """A dataset NeuroConv would copy from its source cannot be referenced."""


def virtualize(
    nwbfile: Any,
    *,
    url_for: Callable[[str], str] | None = None,
    chunk_bytes: int = 4 * 2**20,
    strict: bool = True,
) -> dict[str, VirtualArray]:
    """Replace the data iterators in an NWBFile from NeuroConv with references.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        A file in memory, from an interface's or a converter's create_nwbfile.
    url_for : callable or None
        Maps a local file path to the URL the references should point to.
    chunk_bytes : int
        Approximate size of a chunk of a referenced array.
    strict : bool
        Raise NotVirtualizable for a data iterator that cannot be referenced.
        Otherwise leave it, and its data is copied into the file when written.

    Returns
    -------
    dict
        The VirtualArray now behind each replaced dataset, by "<object name>/<field>".
    """
    from hdmf.data_utils import AbstractDataChunkIterator, DataIO

    arrays_by_reader: dict[int, dict[str, VirtualArray]] = {}
    replaced = {}
    for obj in [nwbfile, *nwbfile.all_children()]:
        fields = getattr(obj, "fields", {})
        for field, value in list(fields.items()):
            iterator = value.data if isinstance(value, DataIO) else value
            if not isinstance(iterator, AbstractDataChunkIterator) or hasattr(value, "virtual"):
                continue
            try:
                virtual = _array_for_iterator(iterator, arrays_by_reader, url_for, chunk_bytes)
            except NotVirtualizable:
                if strict:
                    raise
                continue
            fields[field] = virtual.placeholder()
            replaced[f"{obj.name}/{field}"] = virtual
    return replaced


def _array_for_iterator(
    iterator: Any,
    arrays_by_reader: dict[int, dict[str, VirtualArray]],
    url_for: Callable[[str], str] | None,
    chunk_bytes: int,
) -> VirtualArray:
    """The VirtualArray holding what a NeuroConv data iterator would read."""
    if getattr(iterator, "imaging_extractor", None) is not None:
        return _array_for_imaging(iterator, url_for)
    recording = getattr(iterator, "recording", None)
    if recording is None:
        raise NotVirtualizable(f"{type(iterator).__name__} does not read a SpikeInterface recording")
    if getattr(iterator, "return_in_uV", False) or getattr(iterator, "return_scaled", False):
        raise NotVirtualizable("The series stores scaled values, which are not what the file holds")

    # Walk through channel slices to the recording that reads the file
    channel_ids = getattr(iterator, "channel_ids", None)
    channel_ids = [str(c) for c in (recording.get_channel_ids() if channel_ids is None else channel_ids)]
    base = recording
    while not hasattr(base, "neo_reader"):
        parent = getattr(base, "_parent", None)
        if parent is None or type(base).__name__ != "ChannelSliceRecording":
            raise NotVirtualizable(
                f"{type(base).__name__} is not read through a NEO reader, or changes the samples it reads"
            )
        # A slice may rename channels; follow the ids back to the parent's
        renamed = dict(zip((str(c) for c in base._renamed_channel_ids), (str(c) for c in base._channel_ids)))
        channel_ids = [renamed[c] for c in channel_ids]
        base = parent
    if getattr(base, "inverted_gain", False):
        raise NotVirtualizable("SpikeInterface negates this recording's samples, so the file holds other values")

    reader = base.neo_reader
    try:
        arrays = arrays_by_reader.setdefault(
            id(reader), virtual_arrays_neo(reader, url_for=url_for, chunk_bytes=chunk_bytes)
        )
    except (ValueError, NotImplementedError) as e:
        raise NotVirtualizable(str(e)) from e
    segment = base._recording_segments[getattr(iterator, "segment_index", 0)]
    name = str(base.stream_id).replace("/", "_")
    single = reader.block_count() == 1 and reader.segment_count(0) == 1
    array = arrays[name if single else f"block{segment.block_index}/segment{segment.segment_index}/{name}"]

    # SpikeInterface names channels by NEO's ids or, for some readers, by its names
    for known in (array.attributes["channel_ids"], array.attributes["channel_names"]):
        if set(channel_ids) <= set(known) and len(set(known)) == len(known):
            columns = [known.index(c) for c in channel_ids]
            break
    else:
        raise NotVirtualizable(f"The channels of stream {base.stream_id!r} do not match the recording's")
    if columns != list(range(array.shape[1])):
        attributes = array.attributes
        array = array[:, columns]
        array.attributes = attributes
    return array


def _array_for_imaging(iterator: Any, url_for: Callable[[str], str] | None) -> VirtualArray:
    """The VirtualArray holding the frames a NeuroConv imaging iterator would read.

    The frames come from TIFF pages or from an HDF5 dataset, one frame to a
    chunk. NeuroConv writes frames as (frame, width, height) and, for
    volumes, (frame, width, height, plane), where the files hold (rows,
    columns), so the result is transposed.
    """
    extractor = iterator.imaging_extractor
    url_for = url_for or (lambda path: path)
    if hasattr(extractor, "_mov_field"):
        frames = _hdf5_frames(extractor, url_for)
    else:
        frames = _tiff_frames(extractor, url_for)
    array = frames.transpose(0, 2, 1, 3) if frames.ndim == 4 else frames.transpose(0, 2, 1)
    expected = tuple(int(n) for n in iterator.maxshape)
    # the byte order of the file is in the array's codecs, not its data type
    expected_dtype = np.dtype(iterator.dtype).newbyteorder("=")
    if array.shape != expected or array.dtype != expected_dtype:
        raise NotVirtualizable(
            f"The files give {array.shape} {array.dtype}, and NeuroConv writes {expected} {expected_dtype}"
        )
    return array


def _frame_pages(extractor: Any) -> list[list[tuple[str, int]]]:
    """For each frame of a roiextractors TIFF extractor, the (file, page) of each of its planes."""
    parts = getattr(extractor, "_imaging_extractors", None)
    if parts is not None:
        # several extractors one after another in time, as for Bruker's one file per frame
        if getattr(extractor, "is_volumetric", False):
            raise NotVirtualizable(f"{type(extractor).__name__} assembles volumes from several extractors")
        return [pages for part in parts for pages in _frame_pages(part)]

    table = getattr(extractor, "_frames_to_ifd_table", None)
    file_paths = getattr(extractor, "_file_paths", None) or getattr(extractor, "file_paths", None)
    if table is not None and file_paths is not None:
        # a table of which page of which file holds each plane of each frame, ordered by frame and then plane
        num_planes = int(extractor.get_num_planes()) if getattr(extractor, "is_volumetric", False) else 1
        num_samples = int(extractor.get_num_samples())
        if len(table) < num_samples * num_planes:
            raise NotVirtualizable(f"{type(extractor).__name__} lists fewer pages than its frames need")
        return [
            [
                (str(file_paths[int(row["file_index"])]), int(row["IFD_index"]))
                for row in table[sample * num_planes : (sample + 1) * num_planes]
            ]
            for sample in range(num_samples)
        ]

    file_path = getattr(extractor, "file_path", None)
    if file_path is not None:
        # one file whose pages are the frames, in order
        return [[(str(file_path), page)] for page in range(int(extractor.get_num_samples()))]
    raise NotVirtualizable(f"{type(extractor).__name__} does not say which pages of which files hold its frames")


def _tiff_frames(extractor: Any, url_for: Callable[[str], str]) -> VirtualArray:
    """Frames stored as TIFF pages, as (frame, rows, columns) or (frame, rows, columns, plane)."""
    import tifffile

    pages = _frame_pages(extractor)
    volumetric = bool(getattr(extractor, "is_volumetric", False))
    shape_of = getattr(extractor, "get_frame_shape", None) or extractor.get_image_shape
    rows, columns = (int(n) for n in shape_of()[:2])

    wanted: dict[str, set[int]] = {}
    for planes in pages:
        for path, page in planes:
            wanted.setdefault(path, set()).add(page)
    located: dict[tuple[str, int], tuple[int, int]] = {}
    dtype = None
    for path, indices in wanted.items():
        with tifffile.TiffFile(path, _multifile=False) as tif:
            for index in sorted(indices):
                try:
                    page = tif.pages[index]
                except IndexError as e:
                    raise NotVirtualizable(f"{path} has no page {index}") from e
                page_dtype = np.dtype(page.dtype).newbyteorder(tif.byteorder)
                nbytes = rows * columns * page_dtype.itemsize
                if page.shape != (rows, columns):
                    raise NotVirtualizable(
                        f"Page {index} of {path} is {page.shape} and a frame is {(rows, columns)}: "
                        "frames that are part of a page cannot be referenced yet"
                    )
                # uncompressed, and its strips one after another with nothing between them
                if page.compression != 1 or not page.is_contiguous or sum(page.databytecounts) != nbytes:
                    raise NotVirtualizable(
                        f"Page {index} of {path} is compressed or stored in several pieces, "
                        "so it cannot be referenced as one chunk"
                    )
                if dtype is not None and page_dtype != dtype:
                    raise NotVirtualizable("The pages do not all have the same data type")
                dtype = page_dtype
                located[path, index] = (int(page.dataoffsets[0]), nbytes)

    num_planes = len(pages[0]) if pages else 1
    chunks: dict[tuple[int, ...], tuple[str, int, int]] = {}
    for sample, planes in enumerate(pages):
        for plane, (path, index) in enumerate(planes):
            chunks[(sample, 0, 0, plane) if volumetric else (sample, 0, 0)] = (url_for(path), *located[path, index])
    shape = (len(pages), rows, columns, num_planes) if volumetric else (len(pages), rows, columns)
    chunk_shape = (1, rows, columns, 1) if volumetric else (1, rows, columns)
    return VirtualArray.from_chunks(chunks, shape=shape, chunk_shape=chunk_shape, dtype=dtype)


def _hdf5_frames(extractor: Any, url_for: Callable[[str], str]) -> VirtualArray:
    """Frames of roiextractors' Hdf5ImagingExtractor, which reads channel 0 of a (channel, frame, rows, columns) dataset."""
    import h5py

    path = str(extractor.filepath)
    with h5py.File(path, "r") as f:
        dataset = f[extractor._mov_field]
        offset = dataset.id.get_offset()
        if dataset.ndim != 4 or dataset.chunks is not None or offset is None:
            raise NotVirtualizable(
                f"Dataset {extractor._mov_field!r} of {path} is chunked or not (channel, frame, rows, columns)"
            )
        _, num_samples, rows, columns = (int(n) for n in dataset.shape)
        dtype = dataset.dtype
    # channel 0 is the first run of frames in the dataset; one frame to a chunk
    return VirtualArray.contiguous(
        url_for(path),
        shape=(num_samples, rows, columns),
        dtype=dtype,
        offset=int(offset),
        chunk_bytes=rows * columns * dtype.itemsize,
    )
