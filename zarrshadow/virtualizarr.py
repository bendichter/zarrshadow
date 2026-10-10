"""Use what VirtualiZarr parses in zarrshadow.

VirtualiZarr (https://virtualizarr.readthedocs.io) parses many formats,
among them NetCDF, HDF5, GRIB, FITS, Zarr, and kerchunk references, into a
ManifestStore: Zarr metadata plus, for each array, a manifest of where its
chunks are. The functions here take those manifests:

    from virtualizarr.parsers import HDFParser
    from zarrshadow.virtualizarr import manifest_store_to_rfs

    store = HDFParser()(url="file:///data/air.nc", registry=registry)
    rfs = manifest_store_to_rfs(store)
    write_rfs(rfs, "air.zarrshadow")

manifest_store_to_rfs writes a whole store as a reference file system, with a
chunk index for each array that has many chunks. virtual_array takes one
ManifestArray as a VirtualArray, to stack, transpose, or put in an NWB file.

Requires virtualizarr (pip install zarrshadow[virtualizarr]).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np

from .builder import RfsBuilder, chunk_key
from .chunk_index import MISSING
from .virtual import VirtualArray, _Place

# What VirtualiZarr puts in a manifest's paths for a chunk held in memory
_INLINED = "__inlined__"


def manifest_store_to_rfs(
    store: Any,
    *,
    index_threshold: int | None = 1000,
    record_sources: bool = True,
    url_for: Callable[[str], str] | None = None,
) -> dict:
    """Write a VirtualiZarr ManifestStore as a reference file system.

    Parameters
    ----------
    store : virtualizarr.manifests.ManifestStore or ManifestGroup
        What a VirtualiZarr parser returned, or a group of one.
    index_threshold : int or None
        An array with more chunks than this, all in one file, gets a chunk
        index in place of one ref per chunk.
    record_sources : bool
        Record the size and, for remote URLs, the ETag of each referenced file.
    url_for : callable or None
        Maps each path in the manifests to the URL or path the references
        should use. RfsStore reads local paths and http(s) URLs, so paths of
        other kinds, such as s3://, need mapping. By default a path is kept.

    Returns
    -------
    dict
        A reference file system dict; see zarrshadow.builder.
    """
    builder = RfsBuilder()
    _add_manifest_store(builder, store, index_threshold=index_threshold, url_for=url_for)
    return builder.build(record_sources=record_sources)


def _add_manifest_store(
    builder: RfsBuilder,
    store: Any,
    *,
    index_threshold: int | None,
    url_for: Callable[[str], str] | None = None,
) -> None:
    """The groups, arrays, and chunk locations of a ManifestStore or ManifestGroup."""

    def location(path: str) -> str:
        return _location(path if url_for is None else url_for(path))

    def walk(group: Any, path: str) -> None:
        builder.refs[f"{path}/zarr.json" if path else "zarr.json"] = _metadata_json(group.metadata)
        for name, array in group.arrays.items():
            array_path = f"{path}/{name}" if path else name
            builder.refs[f"{array_path}/zarr.json"] = _array_metadata_json(array.metadata)
            _add_manifest(builder, array_path, array, index_threshold, lambda coords: coords, location)
        for name, subgroup in group.groups.items():
            walk(subgroup, f"{path}/{name}" if path else name)

    walk(getattr(store, "_group", store), "")


def virtual_array(array: Any) -> VirtualArray:
    """A VirtualiZarr ManifestArray as a VirtualArray.

    An uncompressed array stored in one piece of one file becomes a contiguous
    VirtualArray, which can be sliced along any axis. Any other array keeps
    its chunks as they are, and can be stacked and transposed but not sliced.
    """
    contiguous = _as_contiguous(array)
    return contiguous if contiguous is not None else _FromManifest(array)


class _FromManifest(VirtualArray):
    """A VirtualiZarr ManifestArray, with the chunks its manifest lists."""

    def __init__(self, array: Any) -> None:
        metadata = json.loads(_metadata_json(array.metadata))
        self._array = array
        self.shape = tuple(metadata["shape"])
        self.chunk_shape = tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"])
        self.data_type = metadata["data_type"]
        self.codecs = metadata["codecs"]
        self.fill_value = metadata["fill_value"]
        self.attributes = dict(metadata.get("attributes", {}))
        self.dimension_names = metadata.get("dimension_names")

    def _add_chunks(self, builder: RfsBuilder, path: str, place: _Place) -> None:
        _add_manifest(builder, path, self._array, 1000, place)


def _add_manifest(
    builder: RfsBuilder,
    path: str,
    array: Any,
    index_threshold: int | None,
    place: _Place,
    location: Callable[[str], str] | None = None,
) -> None:
    """The chunk locations in a ManifestArray's manifest, as refs or as a chunk index."""
    location = location or _location
    manifest = array.manifest
    paths, offsets, lengths = manifest._paths, manifest._offsets, manifest._lengths
    inlined = getattr(manifest, "_inlined", {})
    for coords, data in inlined.items():
        builder.add_inline_chunk(path, place(list(coords)), data)
    referenced = (paths != "") & (paths != _INLINED)
    n_referenced = int(referenced.sum())
    if n_referenced == 0:
        return
    urls = np.unique(paths[referenced])
    identity = place(list(range(paths.ndim))) == list(range(paths.ndim))
    if identity and not inlined and len(urls) == 1 and index_threshold is not None and n_referenced > index_threshold:
        index = np.full((*paths.shape, 2), MISSING, dtype=np.uint64)
        index[referenced, 0], index[referenced, 1] = offsets[referenced], lengths[referenced]
        builder.add_index(path, location(str(urls[0])), index)
        return
    for coords in np.argwhere(referenced):
        coords = tuple(int(c) for c in coords)
        builder.refs[chunk_key(path, place(list(coords)))] = [
            location(str(paths[coords])),
            int(offsets[coords]),
            int(lengths[coords]),
        ]


def _as_contiguous(array: Any) -> VirtualArray | None:
    """The array as a contiguous VirtualArray, if it is uncompressed and stored in one piece of one file."""
    metadata = json.loads(_metadata_json(array.metadata))
    codecs = metadata["codecs"]
    shape = tuple(metadata["shape"])
    chunk_shape = tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"])
    data_type = metadata["data_type"]
    if len(codecs) != 1 or codecs[0]["name"] != "bytes" or not isinstance(data_type, str) or not shape:
        return None
    try:
        dtype = np.dtype(data_type)
    except TypeError:
        return None
    if dtype.kind not in "iufb" or chunk_shape[1:] != shape[1:] or 0 in shape:
        return None
    if codecs[0].get("configuration", {}).get("endian") == "big":
        dtype = dtype.newbyteorder(">")

    manifest = array.manifest
    paths = manifest._paths.reshape(-1)
    offsets = manifest._offsets.reshape(-1).astype(np.int64)
    lengths = manifest._lengths.reshape(-1).astype(np.int64)
    if getattr(manifest, "_inlined", {}) or len(set(paths.tolist())) != 1 or paths[0] in ("", _INLINED):
        return None
    row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
    chunk_bytes = chunk_shape[0] * row_bytes
    # every chunk must follow the one before it, and only the last may be short
    last_bytes = (shape[0] - (len(offsets) - 1) * chunk_shape[0]) * row_bytes
    if not np.array_equal(offsets, offsets[0] + np.arange(len(offsets)) * chunk_bytes):
        return None
    if np.any(lengths[:-1] != chunk_bytes) or lengths[-1] not in (chunk_bytes, last_bytes):
        return None
    result = VirtualArray.contiguous(
        _location(str(paths[0])),
        shape=shape,
        dtype=dtype,
        offset=int(offsets[0]),
        chunk_bytes=chunk_bytes,
        attributes=metadata.get("attributes", {}),
    )
    result.dimension_names = metadata.get("dimension_names")
    return result


def _metadata_json(metadata: Any) -> str:
    """A zarr metadata object as the text of its zarr.json."""
    from zarr.core.buffer import default_buffer_prototype

    return metadata.to_buffer_dict(default_buffer_prototype())["zarr.json"].to_bytes().decode()


def _array_metadata_json(metadata: Any) -> str:
    """An array's zarr.json, naming the chunk keys that the references use, which are joined by "/"."""
    text = _metadata_json(metadata)
    meta = json.loads(text)
    encoding = {"name": "default", "configuration": {"separator": "/"}}
    if meta.get("chunk_key_encoding") == encoding:
        return text
    meta["chunk_key_encoding"] = encoding
    return json.dumps(meta, separators=(",", ":"))


def _location(path: str) -> str:
    """A manifest's path as zarrshadow refers to files: local paths without the file scheme."""
    return path[len("file://") :] if path.startswith("file://") else path
