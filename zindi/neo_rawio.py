"""Generate a zarr v3 reference file system (RFS) from a NEO raw reader.

NEO reads most electrophysiology formats. Readers of formats that store the
signals of a segment as one uncompressed array (13 in neo 0.14, among them
SpikeGLX, Open Ephys binary, Axon, BrainVision, Neuroscope, and Maxwell)
describe where each array is through NEO's buffer description API: for raw
binary buffers the file, dtype, byte offset, and shape, and for HDF5 buffers
the file and dataset. This generator turns those descriptions into
references, so the signals can be read as Zarr arrays without converting or
copying the files.

Each buffer becomes one array. For a reader with one block and one segment
it is at "<buffer id>", and otherwise at "block<b>/segment<s>/<buffer id>".
Raw buffers are stored time by channel, as NEO describes them, and are cut
into chunks of whole samples along time, which are evenly spaced in the file
and become one gen entry. The array's "neo" attribute describes the streams
in the buffer: which columns belong to each, the sampling rate, t_start, and
each channel's id, name, units, gain, and offset.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np

from .builder import RfsBuilder, bytes_codecs, contiguous_chunk_shape, zarr_data_type
from .virtual import VirtualArray


def generate_rfs_neo(
    reader: Any,
    *,
    url_for: Callable[[str], str] | None = None,
    chunk_bytes: int = 4 * 2**20,
    chunk_index_threshold: int | None = 1000,
    record_sources: bool = True,
) -> dict:
    """Generate a zarr v3 reference file system from a NEO raw reader.

    Parameters
    ----------
    reader : neo.rawio.BaseRawIO
        A NEO raw reader for local files, such as
        ``neo.rawio.SpikeGLXRawIO(dirname=...)``. It must support the buffer
        description API. Its header is parsed if it has not been.
    url_for : callable or None
        Maps a local file path to the URL the references should point to, for
        example where the files are hosted. By default references use the
        local paths.
    chunk_bytes : int
        Approximate size of a chunk of a raw buffer. Default 4 MiB.
    chunk_index_threshold : int or None
        For HDF5 buffers, datasets with more chunks than this get a chunk index.
    record_sources : bool
        Record the size and, for remote URLs, the ETag of each referenced file.

    Returns
    -------
    dict
        A reference file system dict; see zindi.builder.
    """
    import neo

    if getattr(reader, "header", None) is None:
        reader.parse_header()
    if not reader.has_buffer_description_api():
        raise ValueError(
            f"{type(reader).__name__} does not describe its signal buffers, so its "
            "files cannot be referenced; zindi supports the NEO readers that do"
        )
    url_for = url_for or (lambda path: path)

    builder = RfsBuilder()
    builder.add_group("", {"neo": {"rawio": type(reader).__name__, "neo_version": neo.__version__}})
    n_blocks = reader.block_count()
    single = n_blocks == 1 and reader.segment_count(0) == 1
    for block in range(n_blocks):
        if not single:
            builder.add_group(f"block{block}")
        for seg in range(reader.segment_count(block)):
            if not single:
                builder.add_group(f"block{block}/segment{seg}")
            for buffer in reader.header["signal_buffers"]:
                buffer_id = str(buffer["id"])
                try:
                    desc = reader.get_analogsignal_buffer_description(
                        block_index=block, seg_index=seg, buffer_id=buffer_id
                    )
                except KeyError:
                    continue  # this buffer is not in this segment
                name = buffer_id.replace("/", "_")
                path = name if single else f"block{block}/segment{seg}/{name}"
                attrs = {"neo": _buffer_attributes(reader, block, seg, buffer, desc)}
                if desc["type"] == "raw":
                    _add_raw_buffer(builder, path, desc, attrs, url_for, chunk_bytes)
                elif desc["type"] == "hdf5":
                    _add_hdf5_buffer(builder, path, desc, attrs, url_for, chunk_bytes, chunk_index_threshold)
                else:
                    raise ValueError(f"Unsupported NEO buffer type {desc['type']!r} for buffer {buffer_id!r}")
    return builder.build(record_sources=record_sources)


def virtual_arrays_neo(
    reader: Any,
    *,
    url_for: Callable[[str], str] | None = None,
    chunk_bytes: int = 4 * 2**20,
) -> dict[str, VirtualArray]:
    """One VirtualArray for each signal stream of a NEO raw reader.

    generate_rfs_neo describes each buffer as the file stores it, which may
    hold several streams side by side, such as the neural channels and the
    sync channel of a SpikeGLX file. Here each stream is its own array, time
    by channel, holding only its channels. Each array's attributes give the
    stream's sampling rate, t_start, and its channels' ids, names, units,
    gains, and offsets.

    The arrays are keyed by stream id for a reader with one block and one
    segment, and otherwise by "block<b>/segment<s>/<stream id>". The
    parameters are those of generate_rfs_neo. Only raw binary buffers are
    supported.
    """
    if getattr(reader, "header", None) is None:
        reader.parse_header()
    if not reader.has_buffer_description_api():
        raise ValueError(
            f"{type(reader).__name__} does not describe its signal buffers, so its "
            "files cannot be referenced; zindi supports the NEO readers that do"
        )
    url_for = url_for or (lambda path: path)
    n_blocks = reader.block_count()
    single = n_blocks == 1 and reader.segment_count(0) == 1
    buffers = {str(buffer["id"]): buffer for buffer in reader.header["signal_buffers"]}
    arrays: dict[str, VirtualArray] = {}
    for block in range(n_blocks):
        for seg in range(reader.segment_count(block)):
            for stream in reader.header["signal_streams"]:
                stream_id, buffer_id = str(stream["id"]), str(stream["buffer_id"])
                try:
                    desc = reader.get_analogsignal_buffer_description(
                        block_index=block, seg_index=seg, buffer_id=buffer_id
                    )
                except KeyError:
                    continue  # this buffer is not in this segment
                if desc["type"] != "raw":
                    raise NotImplementedError(
                        f"Stream {stream_id!r} is in a {desc['type']} buffer; only raw binary buffers "
                        "can be split into streams"
                    )
                _check_raw_layout(desc)
                file_path = str(desc["file_path"])
                buffer_array = VirtualArray.contiguous(
                    url_for(file_path),
                    shape=[int(n) for n in desc["shape"]],
                    dtype=np.dtype(desc["dtype"]),
                    offset=int(desc["file_offset"]),
                    file_size=os.path.getsize(file_path),
                    chunk_bytes=chunk_bytes,
                )
                columns = reader._stream_buffer_slice.get(stream_id)
                array = buffer_array if columns is None else buffer_array[:, columns]
                info = _buffer_attributes(reader, block, seg, buffers[buffer_id], desc)
                (stream_info,) = [s for s in info["streams"] if s["id"] == stream_id]
                array.attributes = {k: v for k, v in stream_info.items() if k != "columns"}
                name = stream_id.replace("/", "_")
                arrays[name if single else f"block{block}/segment{seg}/{name}"] = array
    return arrays


def _check_raw_layout(desc: dict) -> None:
    if desc.get("order", "C") != "C" or desc.get("time_axis", 0) != 0:
        raise NotImplementedError(
            f"raw buffers are supported in C order with time first; got order "
            f"{desc.get('order')!r} and time_axis {desc.get('time_axis', 0)}"
        )


def _add_raw_buffer(
    builder: RfsBuilder,
    path: str,
    desc: dict,
    attrs: dict,
    url_for: Callable[[str], str],
    chunk_bytes: int,
) -> None:
    """A raw binary buffer: an uncompressed array stored in one piece."""
    _check_raw_layout(desc)
    dtype = np.dtype(desc["dtype"])
    shape = [int(n) for n in desc["shape"]]
    chunk_shape = contiguous_chunk_shape(shape, dtype.itemsize, chunk_bytes)
    builder.add_array(
        path,
        shape=shape,
        data_type=zarr_data_type(dtype),
        chunk_shape=[max(c, 1) for c in chunk_shape],
        codecs=bytes_codecs(dtype),
        fill_value=0,
        attributes=attrs,
        dimension_names=["time", "channel"][: len(shape)],
    )
    if int(np.prod(shape)) == 0:
        return
    file_path = str(desc["file_path"])
    builder.add_contiguous_chunks(
        path,
        url=url_for(file_path),
        start=int(desc["file_offset"]),
        shape=shape,
        chunk_shape=chunk_shape,
        itemsize=dtype.itemsize,
        file_size=os.path.getsize(file_path),
    )


def _add_hdf5_buffer(
    builder: RfsBuilder,
    path: str,
    desc: dict,
    attrs: dict,
    url_for: Callable[[str], str],
    chunk_bytes: int,
    chunk_index_threshold: int | None,
) -> None:
    """An HDF5 buffer, such as Maxwell's: one dataset, referenced by the HDF5 generator."""
    from .hdf5 import add_hdf5_dataset

    file_path = str(desc["file_path"])
    time_axis = desc.get("time_axis", 0)
    add_hdf5_dataset(
        builder,
        path,
        file_path,
        desc["hdf5_path"],
        url=url_for(file_path),
        attributes=attrs,
        dimension_names=["time", "channel"] if time_axis == 0 else ["channel", "time"],
        chunk_index_threshold=chunk_index_threshold,
        contiguous_chunk_bytes=chunk_bytes,
    )


def _buffer_attributes(reader: Any, block: int, seg: int, buffer: Any, desc: dict) -> dict:
    """What NEO knows about the streams stored in one buffer, in JSON form."""
    header = reader.header
    channels = header["signal_channels"]
    buffer_id = str(buffer["id"])
    streams = []
    for stream_index, stream in enumerate(header["signal_streams"]):
        if str(stream["buffer_id"]) != buffer_id:
            continue
        stream_id = str(stream["id"])
        chans = channels[channels["stream_id"] == stream["id"]]
        streams.append({
            "id": stream_id,
            "name": str(stream["name"]),
            "columns": _columns(reader._stream_buffer_slice.get(stream_id)),
            "sampling_rate": float(chans["sampling_rate"][0]) if len(chans) else None,
            "t_start": float(reader.get_signal_t_start(block, seg, stream_index)),
            "channel_ids": [str(c) for c in chans["id"]],
            "channel_names": [str(c) for c in chans["name"]],
            "units": [str(c) for c in chans["units"]],
            "gain": [float(c) for c in chans["gain"]],
            "offset": [float(c) for c in chans["offset"]],
        })
    return {
        "rawio": type(reader).__name__,
        "block": block,
        "segment": seg,
        "buffer_id": buffer_id,
        "buffer_name": str(buffer["name"]),
        "time_axis": int(desc.get("time_axis", 0)),
        "streams": streams,
    }


def _columns(buffer_slice: Any) -> Any:
    """Which columns of the buffer a stream uses: all (None), a slice, or a list."""
    if buffer_slice is None:
        return None
    if isinstance(buffer_slice, slice):
        return {"start": buffer_slice.start, "stop": buffer_slice.stop, "step": buffer_slice.step}
    return [int(i) for i in np.asarray(buffer_slice).ravel()]
