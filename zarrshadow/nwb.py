"""Write NWB files whose large datasets are references to other files.

Build an NWBFile with pynwb as usual, and give each large dataset a
VirtualArray's placeholder as its data:

    arrays = virtual_arrays_neo(reader)
    series = ElectricalSeries(name="ElectricalSeries", data=arrays["imec0.ap"].placeholder(), ...)
    nwbfile.add_acquisition(series)
    rfs = write_virtual_nwb(nwbfile, "session.nwb.zarrshadow")

hdmf-zarr writes the file's structure: the groups, attributes, object ids,
references, the cached specification, and every dataset that holds real data.
For a placeholder it writes only the array's metadata, and write_virtual_nwb
adds the VirtualArray's chunk locations there. No signal data is read or
copied. Read the result with

    NWBZarrIO(RfsStore(load_rfs("session.nwb.zarrshadow")), mode="r")

Requires pynwb and hdmf-zarr 0.14 or later (pip install zarrshadow[nwb]).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .builder import RfsBuilder, inline_value, write_rfs
from .virtual import VirtualArray

_LAYOUT_FIELDS = ("shape", "data_type", "chunk_grid", "codecs", "fill_value")


def placeholder(virtual: VirtualArray) -> Any:
    """An hdmf-zarr data object that stands for a VirtualArray in an NWBFile.

    Written by hdmf-zarr, it creates an array laid out like the VirtualArray
    and no chunks. write_virtual_nwb then points the array at the files.
    """
    from hdmf.data_utils import AbstractDataChunkIterator
    from hdmf_zarr import ZarrDataIO
    from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
    from zarr.registry import get_codec_class

    class _NoData(AbstractDataChunkIterator):
        """Has a shape and a dtype and yields no chunks."""

        def __init__(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
            self._shape, self._dtype = shape, dtype

        def __iter__(self) -> _NoData:
            return self

        def __next__(self) -> Any:
            raise StopIteration

        def recommended_chunk_shape(self) -> None:
            return None

        def recommended_data_shape(self) -> tuple[int, ...]:
            return self._shape

        @property
        def dtype(self) -> np.dtype:
            return self._dtype

        @property
        def maxshape(self) -> tuple[int, ...]:
            return self._shape

    codecs = [get_codec_class(codec["name"]).from_dict(codec) for codec in virtual.codecs]
    serializers = [codec for codec in codecs if isinstance(codec, ArrayBytesCodec)]
    if len(serializers) != 1:
        raise ValueError(f"Expected one array-to-bytes codec, found {len(serializers)} in {virtual.codecs}")
    data = ZarrDataIO(
        _NoData(tuple(virtual.shape), virtual.dtype),
        chunks=tuple(virtual.chunk_shape),
        filters=[codec for codec in codecs if isinstance(codec, ArrayArrayCodec)] or None,
        serializer=serializers[0],
        compressors=[codec for codec in codecs if isinstance(codec, BytesBytesCodec)] or False,
        fillvalue=virtual.fill_value,
    )
    data.virtual = virtual
    return data


def write_virtual_nwb(nwbfile: Any, output_path: str | None = None, *, record_sources: bool = True) -> dict:
    """Write an NWBFile whose placeholders become references, and return the reference file system.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        A file in memory. Datasets given VirtualArray.placeholder() as data
        become references; everything else is stored in the result.
    output_path : str or None
        Where to write the result, as for write_rfs: a directory, or a single
        file if the path ends in ".json". None writes nothing.
    record_sources : bool
        Record the size and, for remote URLs, the ETag of each referenced file.
    """
    from hdmf_zarr import NWBZarrIO
    from zarr.storage import MemoryStore

    placeholders = _find_placeholders(nwbfile)
    store = MemoryStore()
    with NWBZarrIO(store, mode="w") as io:
        io.write(nwbfile)
    written = _dump(store)

    builder = RfsBuilder()
    for key, data in written.items():
        builder.refs[key] = inline_value(data)

    paths = _paths_by_object_id(written)
    for object_id, field, virtual in placeholders:
        if object_id not in paths:
            raise RuntimeError(f"hdmf-zarr did not write the object {object_id} that holds a virtual array")
        path = paths[object_id] if field is None else f"{paths[object_id]}/{field}"
        meta = written.get(f"{path}/zarr.json")
        meta = None if meta is None else json.loads(meta)
        if meta is None or meta.get("node_type") != "array":
            raise RuntimeError(f"Expected hdmf-zarr to write the virtual array at {path!r}, and found no array there")
        expected = _normalize(virtual.layout(), virtual)
        actual = _normalize({name: meta.get(name) for name in _LAYOUT_FIELDS}, virtual)
        for name in _LAYOUT_FIELDS:
            if actual[name] != expected[name]:
                raise RuntimeError(
                    f"hdmf-zarr wrote {path!r} with {name} {actual[name]!r}, and the virtual array "
                    f"has {expected[name]!r}; its chunks would not decode"
                )
        virtual.add_to(builder, path, metadata=False)

    rfs = builder.build(record_sources=record_sources)
    if output_path is not None:
        write_rfs(rfs, output_path)
    return rfs


def _find_placeholders(nwbfile: Any) -> list[tuple[str, str | None, VirtualArray]]:
    """Every placeholder in the file: (object id of its holder, field name or None, VirtualArray).

    The field is None when the holder is itself a dataset, such as a column of a table.
    """
    from hdmf.container import Data

    found = []
    for obj in [nwbfile, *nwbfile.all_children()]:
        if isinstance(obj, Data):
            virtual = getattr(obj.data, "virtual", None)
            if isinstance(virtual, VirtualArray):
                found.append((obj.object_id, None, virtual))
            continue
        for field, value in obj.fields.items():
            virtual = getattr(value, "virtual", None)
            if isinstance(virtual, VirtualArray):
                found.append((obj.object_id, field, virtual))
    return found


def _paths_by_object_id(written: dict[str, bytes]) -> dict[str, str]:
    """The path of every group and array hdmf-zarr wrote, by the object id in its attributes."""
    paths = {}
    for key, data in written.items():
        if key == "zarr.json" or key.endswith("/zarr.json"):
            object_id = json.loads(data).get("attributes", {}).get("object_id")
            if object_id is not None:
                paths[object_id] = key[: -len("/zarr.json")] if "/" in key else ""
    return paths


def _normalize(layout: dict, virtual: VirtualArray) -> dict:
    """A layout in the form zarr writes it, so that two descriptions of one layout compare equal."""
    from zarr.registry import get_codec_class

    layout = dict(layout)
    codecs = []
    for codec in layout["codecs"] or []:
        codec = get_codec_class(codec["name"]).from_dict(codec).to_dict()
        if codec["name"] == "bytes" and isinstance(virtual.data_type, str) and virtual.dtype.itemsize == 1:
            codec = {"name": "bytes"}  # single bytes have no byte order
        codecs.append(codec)
    layout["codecs"] = codecs
    return layout


def _dump(store: Any) -> dict[str, bytes]:
    """Every key and value of a zarr store."""
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    async def dump() -> dict[str, bytes]:
        out = {}
        async for key in store.list():
            buffer = await store.get(key, prototype=default_buffer_prototype())
            out[key] = buffer.to_bytes()
        return out

    return sync(dump())
