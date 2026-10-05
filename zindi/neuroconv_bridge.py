"""Make the NWB files NeuroConv builds virtual.

NeuroConv builds an NWBFile in memory in which each large dataset is an
iterator that reads the source file when the NWB file is written. virtualize
swaps those iterators for references, so that nothing is read:

    nwbfile = interface.create_nwbfile(metadata=interface.get_metadata())
    virtualize(nwbfile)
    write_virtual_nwb(nwbfile, "session.nwb.zindi")

Everything else in the file, the metadata and the tables NeuroConv builds, is
written as NeuroConv made it. This covers recordings that SpikeInterface
reads through a NEO reader with the buffer description API: SpikeGLX, Open
Ephys binary, Neuroscope, and the others that virtual_arrays_neo supports.

Experimental. As of October 2026 SpikeInterface requires zarr<3 and NeuroConv
0.10 reads an attribute at import that Zarr v3 removed, so NeuroConv and
zindi install together only with the zarr requirement overridden; see
docs/virtual-nwb-plan.md.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
