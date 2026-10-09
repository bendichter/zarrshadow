"""Generate a zarr v3 reference file system (RFS) from an HDF5 file.

This is zarrshadow's generator for HDF5, including NWB files and MATLAB v7.3 .mat
files. It walks the file with h5py and describes it through RfsBuilder: Zarr
v3 metadata for every group and array, small datasets inline, and the byte
range of every chunk in the original file, so zarr reads the data without
copying it.

HDF5 features that Zarr lacks (links, object references, compound types,
scalars) follow the unified convention shared with hdmf-zarr:
  https://github.com/NeurodataWithoutBorders/lindi/issues/125
  https://github.com/hdmf-dev/hdmf-zarr/issues/335
"""

from __future__ import annotations

import base64
import json
from typing import Any, Callable

import h5py
import numpy as np
from tqdm import tqdm

from .attr_conversion import h5_attr_to_zarr
from .builder import DEFAULT_CODECS, RfsBuilder, contiguous_chunk_shape, metadata_key, zarr_data_type
from .chunk_index import build_index
from .h5_chunk_utils import (
    apply_to_all_chunk_info,
    get_byte_range_for_contiguous_dataset,
    get_max_num_chunks,
)
from .h5_filters_to_codecs import h5_filters_to_codec_pipeline

STRING_CODECS = [{"name": "vlen-utf8", "configuration": {}}]


def generate_rfs(
    hdf5_url_or_path: str,
    *,
    local_hdf5_path: str | None = None,
    h5f: h5py.File | None = None,
    chunk_index_threshold: int | None = 1000,
    contiguous_chunk_bytes: int | None = 4 * 2**20,
    record_sources: bool = True,
) -> dict:
    """Generate a zarr v3 reference file system from an HDF5 file.

    Parameters
    ----------
    hdf5_url_or_path : str
        URL or local path of the HDF5 file. If a URL (http/https), it is
        used both as the source for reading metadata and as the target for
        chunk references. For remote files, zarrshadow uses its built-in Remfile
        to read the file over HTTP.
    local_hdf5_path : str or None
        Path to a local copy of the HDF5 file to read metadata from. If
        provided, metadata is read from this local file but chunk references
        still point to hdf5_url_or_path.
    h5f : h5py.File or None
        An already-open h5py.File object. If provided, it is used directly
        and neither hdf5_url_or_path nor local_hdf5_path are opened.
    chunk_index_threshold : int or None
        Arrays with more chunks than this get a chunk index (a numpy array of
        byte ranges, see ``zarrshadow.chunk_index``) in place of one ref per chunk.
        None lists every chunk in refs.
    contiguous_chunk_bytes : int or None
        A contiguous (unchunked) HDF5 dataset larger than this is presented as
        chunks of about this many bytes along its first axis, described by one
        "gen" entry, so reading part of it does not fetch all of it. None keeps
        each contiguous dataset as a single chunk. Default 4 MiB.
    record_sources : bool
        Record the size and, for remote files, the ETag of each file the
        references point into, under "sources", so readers can detect a file
        that has changed since. For DANDI assets these come from the asset
        metadata. Default True.

    Returns
    -------
    dict
        A reference file system dict with keys "refs" and "version", and
        "indexes" mapping array paths to {"url", "index"} when any array is
        indexed, and "gen" when any contiguous dataset is split. Either makes
        "version" 2.
    """
    builder = RfsBuilder()
    opts: dict[str, Any] = {
        "chunk_index_threshold": chunk_index_threshold,
        "contiguous_chunk_bytes": contiguous_chunk_bytes,
    }

    def process(opened: h5py.File, raw_source: str) -> None:
        opts["offset_shift"] = _detect_offset_shift(opened, _raw_reader(raw_source))
        opts["spec_location"] = _spec_location(opened)
        _process_group(opened, "", builder, hdf5_url_or_path, opened, **opts)

    if h5f is not None:
        process(h5f, hdf5_url_or_path)
    elif local_hdf5_path is not None:
        with h5py.File(local_hdf5_path, "r") as opened:
            process(opened, local_hdf5_path)
    elif hdf5_url_or_path.startswith("http://") or hdf5_url_or_path.startswith("https://"):
        from .remfile import ZarrShadowRemfile

        remf = ZarrShadowRemfile(hdf5_url_or_path)
        with h5py.File(remf, "r") as opened:
            process(opened, hdf5_url_or_path)
    else:
        with h5py.File(hdf5_url_or_path, "r") as opened:
            process(opened, hdf5_url_or_path)

    _add_dtype_attrs(builder.refs)
    return builder.build(record_sources=record_sources)


def add_hdf5_dataset(
    builder: RfsBuilder,
    path: str,
    local_path: str,
    dataset_path: str,
    *,
    url: str | None = None,
    attributes: dict | None = None,
    dimension_names: list[str] | None = None,
    chunk_index_threshold: int | None = 1000,
    contiguous_chunk_bytes: int | None = 4 * 2**20,
) -> None:
    """Add one dataset of an HDF5 file to builder at path.

    The array's attributes are the dataset's HDF5 attributes updated with
    attributes. References point to url, which defaults to local_path.
    """
    with h5py.File(local_path, "r") as h5f:
        shift = _detect_offset_shift(h5f, _raw_reader(local_path))
        _process_dataset(
            h5f[dataset_path], path, builder, url or local_path, h5f,
            chunk_index_threshold=chunk_index_threshold,
            contiguous_chunk_bytes=contiguous_chunk_bytes,
            offset_shift=shift,
        )
    meta = json.loads(builder.refs[metadata_key(path)])
    meta.setdefault("attributes", {}).update(attributes or {})
    if dimension_names is not None:
        meta["dimension_names"] = list(dimension_names)
    builder.set_metadata(path, meta)


# ---------------------------------------------------------------------------
# Walking the file
# ---------------------------------------------------------------------------


def _process_group(
    item: h5py.Group,
    path: str,
    builder: RfsBuilder,
    url: str,
    h5f: h5py.File,
    **opts: Any,
) -> None:
    """Add a group's metadata, then its children."""
    # A soft link is recorded in its parent's _LINKS; don't recurse into the target
    if path:
        link = h5f.get("/" + path, getlink=True)
        if isinstance(link, h5py.SoftLink):
            return

    attrs = _collect_attrs(item, h5f=h5f, label=path or "(root)")

    # hdmf-zarr stores the root .specloc as a plain path, not a reference
    specloc = attrs.get(".specloc")
    if not path and isinstance(specloc, dict) and "_REFERENCE" in specloc:
        attrs[".specloc"] = specloc["_REFERENCE"]["path"].lstrip("/")

    links = _collect_child_links(item, h5f)
    if links:
        attrs["_LINKS"] = links

    builder.add_group(path, attrs)

    for name in item.keys():
        child_path = f"{path}/{name}" if path else name
        if isinstance(h5f.get("/" + child_path, getlink=True), h5py.SoftLink):
            continue  # already recorded in _LINKS
        child = item[name]
        if isinstance(child, h5py.Group):
            _process_group(child, child_path, builder, url, h5f, **opts)
        elif isinstance(child, h5py.Dataset):
            _process_dataset(child, child_path, builder, url, h5f, **opts)


def _process_dataset(
    ds: h5py.Dataset,
    path: str,
    builder: RfsBuilder,
    url: str,
    h5f: h5py.File,
    *,
    chunk_index_threshold: int | None,
    contiguous_chunk_bytes: int | None,
    offset_shift: int,
    spec_location: str | None = None,
) -> None:
    """Add a dataset's metadata and chunk locations, or its data inline."""
    attrs = _collect_attrs(ds, h5f=h5f, label=path)

    if _should_inline(ds):
        # hdmf-zarr reads a cached specification as a scalar string. Files
        # written by HDMF before 2.0 hold it in an array of one.
        cached_spec = spec_location is not None and path.startswith(spec_location + "/")
        _process_inline_dataset(ds, path, builder, attrs, h5f, scalar_if_single=cached_spec)
        return

    if ds.chunks:
        chunks = list(ds.chunks)
    else:
        chunks = contiguous_chunk_shape(ds.shape, ds.dtype.itemsize, contiguous_chunk_bytes)
    chunks = [max(c, 1) for c in chunks]  # Zarr doesn't allow zero-size chunks

    if ds.dtype.kind == "V" and ds.dtype.fields is not None:
        # Compound: zarr v3's struct data_type carries the fields
        data_type: str | dict = _compound_dtype_to_zarr_v3(ds.dtype)
        fill_value = _encode_compound_fill_value(ds.dtype)
    else:
        data_type = _numpy_dtype_to_zarr_v3(ds.dtype)
        fill_value = _encode_fill_value(ds.fillvalue, ds.dtype)

    builder.add_array(
        path,
        shape=ds.shape,
        data_type=data_type,
        chunk_shape=chunks,
        codecs=h5_filters_to_codec_pipeline(ds),
        fill_value=fill_value,
        attributes=attrs,
    )
    if np.prod(ds.shape) > 0:
        _add_chunk_refs(ds, path, builder, url, chunk_index_threshold, chunks, offset_shift)


def _process_inline_dataset(
    ds: h5py.Dataset,
    path: str,
    builder: RfsBuilder,
    attrs: dict,
    h5f: h5py.File,
    *,
    scalar_if_single: bool = False,
) -> None:
    """Store a small dataset, string data, or references in the RFS itself.

    With scalar_if_single, a string dataset of shape (1,) is stored as a scalar.
    """
    data = ds[()]

    def add_strings(strings: list[str], shape: list[int]) -> None:
        if scalar_if_single and shape == [1]:
            shape = []
        builder.add_array(
            path, shape=shape, data_type="string", chunk_shape=shape,
            codecs=STRING_CODECS, fill_value="", attributes=attrs,
        )
        builder.add_inline_chunk(path, [0] * len(shape), _encode_vlen_utf8(strings))

    if ds.ndim == 0:
        if isinstance(data, h5py.Reference):
            # Scalar object reference: the target path as a string
            attrs["_DTYPE"] = "object_reference"
            add_strings([h5f[data].name], [])
            return
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        if isinstance(data, str):
            add_strings([data], [])
            return
        data = np.asarray(data)
    else:
        if h5py.check_dtype(ref=ds.dtype) == h5py.Reference:
            # Object reference array: target paths as strings
            paths = []
            for item in np.nditer(ds[...], flags=["refs_ok"]):
                val = item.item()
                paths.append(h5f[val].name if isinstance(val, h5py.Reference) else "")
            attrs["_DTYPE"] = "object_reference"
            add_strings(paths, list(ds.shape))
            return

        if ds.dtype.kind in ("O", "U", "S"):
            strings = []
            for item in np.nditer(ds[...], flags=["refs_ok"]):
                val = item.item()
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                strings.append(str(val) if val is not None else "")
            add_strings(strings, list(ds.shape))
            return

    if ds.dtype.kind == "V" and ds.dtype.fields is not None:
        # Compound: reference fields become path strings
        dtype = data.dtype
        ref_fields = _get_reference_fields(dtype)
        if ref_fields:
            data, dtype = _resolve_compound_references(data, dtype, ref_fields, h5f)
            attrs["_REFERENCE_FIELDS"] = ref_fields
        data_type: str | dict = _compound_dtype_to_zarr_v3(dtype)
        fill_value = _encode_compound_fill_value(dtype)
    else:
        dtype = data.dtype
        data_type = _numpy_dtype_to_zarr_v3(dtype)
        fill_value = _encode_fill_value(ds.fillvalue, dtype)

    shape = list(data.shape)
    builder.add_array(
        path, shape=shape, data_type=data_type, chunk_shape=shape,
        codecs=DEFAULT_CODECS, fill_value=fill_value, attributes=attrs,
    )
    if dtype.byteorder == ">":
        data = data.astype(dtype.newbyteorder("<"))
    builder.add_inline_chunk(path, [0] * len(shape), data.tobytes())


def _add_chunk_refs(
    ds: h5py.Dataset,
    path: str,
    builder: RfsBuilder,
    url: str,
    chunk_index_threshold: int | None,
    chunk_shape: list[int],
    offset_shift: int = 0,
) -> None:
    """Add a dataset's chunk locations: one ref each, a chunk index, or strided slabs.

    offset_shift is added to every byte offset h5py reports (see _detect_offset_shift).
    """
    if ds.chunks is not None:
        chunk_size = ds.chunks
        num_chunks = get_max_num_chunks(shape=ds.shape, chunk_size=chunk_size)
        if chunk_index_threshold is not None and num_chunks > chunk_index_threshold:
            _add_chunk_index(ds, path, builder, url, num_chunks, offset_shift)
            return
        pbar = tqdm(total=num_chunks, desc=f"Chunk refs for {path}", leave=True, delay=2)

        def store_chunk_info(chunk_info: Any) -> None:
            coords = [a // b for a, b in zip(chunk_info.chunk_offset, chunk_size)]
            builder.add_chunk(path, coords, url, chunk_info.byte_offset + offset_shift, chunk_info.size)
            pbar.update()

        apply_to_all_chunk_info(ds, store_chunk_info)
        pbar.close()
        return

    # Contiguous dataset: one chunk, or equal slabs along the first axis
    byte_offset, _ = get_byte_range_for_contiguous_dataset(ds)
    builder.add_contiguous_chunks(
        path,
        url=url,
        start=byte_offset + offset_shift,
        shape=ds.shape,
        chunk_shape=chunk_shape,
        itemsize=ds.dtype.itemsize,
        file_size=ds.file.id.get_filesize(),
    )


def _add_chunk_index(
    ds: h5py.Dataset,
    path: str,
    builder: RfsBuilder,
    url: str,
    num_chunks: int,
    offset_shift: int = 0,
) -> None:
    """Record a dataset's chunk byte ranges in an index array."""
    chunk_size = ds.chunks
    grid_shape = tuple(-(-a // b) for a, b in zip(ds.shape, chunk_size))
    pbar = tqdm(total=num_chunks, desc=f"Chunk index for {path}", leave=True, delay=2)

    def for_each_chunk(set_chunk: Any) -> None:
        def store_chunk_info(chunk_info: Any) -> None:
            coords = tuple(a // b for a, b in zip(chunk_info.chunk_offset, chunk_size))
            set_chunk(coords, chunk_info.byte_offset + offset_shift, chunk_info.size)
            pbar.update()

        apply_to_all_chunk_info(ds, store_chunk_info)

    builder.add_index(path, url, build_index(grid_shape, for_each_chunk))
    pbar.close()


# ---------------------------------------------------------------------------
# Byte offsets in files with a userblock
# ---------------------------------------------------------------------------


def _raw_reader(path_or_url: str) -> Callable[[int, int], bytes]:
    """Read raw bytes from the HDF5 file, bypassing h5py."""
    from .rfs_store import _read_bytes_from_url_or_path

    return lambda offset, length: _read_bytes_from_url_or_path(path_or_url, offset, length)


def _first_stored_block(h5f: h5py.File) -> tuple[int, bytes] | None:
    """The offset h5py reports for some stored data, and the bytes it holds there.

    Uses the first chunk of the first chunked dataset with any chunks written,
    or else the start of the first contiguous dataset with storage.
    """
    contiguous: tuple[int, bytes] | None = None
    found: tuple[int, bytes] | None = None

    def visit(name: str, obj: Any) -> Any:
        nonlocal contiguous, found
        if not isinstance(obj, h5py.Dataset) or obj.size == 0 or obj.dtype.kind in "OV":
            return None
        if obj.chunks is not None:
            try:
                info = obj.id.get_chunk_info(0)
            except Exception:
                return None
            if info.byte_offset is None:
                return None
            _, raw = obj.id.read_direct_chunk(info.chunk_offset)
            found = (info.byte_offset, bytes(raw[:64]))
            return True  # stop visiting
        if contiguous is None and obj.id.get_offset() is not None:
            first = obj[:8] if obj.ndim == 1 else obj[(0,) * (obj.ndim - 1)][:8]
            contiguous = (obj.id.get_offset(), np.asarray(first).tobytes())
        return None

    h5f.visititems(visit)
    return found or contiguous


def _detect_offset_shift(h5f: h5py.File, read_bytes: Callable[[int, int], bytes]) -> int:
    """How far to shift the byte offsets h5py reports to get file offsets.

    A file with a userblock (MATLAB v7.3 files have 512 bytes) is laid out
    after it. HDF5 1.14 and later report absolute file offsets, but HDF5 1.10
    reports offsets relative to the superblock, which puts every reference
    off by the size of the userblock. Instead of relying on the library
    version, read one stored block at the reported offset and with the
    userblock added, and keep whichever matches what h5py reads.
    """
    userblock = h5f.userblock_size
    if not userblock:
        return 0
    block = _first_stored_block(h5f)
    if block is None:
        return 0
    offset, expected = block
    for shift in (0, userblock):
        if read_bytes(offset + shift, len(expected)) == expected:
            return shift
    raise RuntimeError(
        f"Byte offsets reported by HDF5 do not match the file, with or without the "
        f"{userblock}-byte userblock; chunk references would be wrong"
    )


# ---------------------------------------------------------------------------
# Attributes, links, and types
# ---------------------------------------------------------------------------


def _spec_location(h5f: h5py.File) -> str | None:
    """The path of the group that holds an NWB file's cached specifications."""
    specloc = h5f.attrs.get(".specloc")
    if not isinstance(specloc, h5py.Reference):
        return None
    try:
        return h5f[specloc].name.strip("/")
    except (KeyError, ValueError):
        return None


def _collect_attrs(
    item: h5py.Group | h5py.Dataset, *, h5f: h5py.File, label: str
) -> dict[str, Any]:
    """Collect and convert all attributes from an HDF5 item."""
    attrs: dict[str, Any] = {}
    for key in item.attrs:
        try:
            val = h5_attr_to_zarr(item.attrs[key], label=f"{label}.{key}", h5f=h5f)
            if val is not None:
                attrs[key] = val
        except (ValueError, Exception):
            # Skip attributes that can't be converted
            pass
    return attrs


def _collect_child_links(item: h5py.Group, h5f: h5py.File) -> list[dict]:
    """Collect soft links among children of a group (unified convention)."""
    links = []
    for name in item.keys():
        child_path = item.name.rstrip("/") + "/" + name
        if child_path.startswith("//"):
            child_path = child_path[1:]
        child_link = h5f.get(child_path, getlink=True)
        if isinstance(child_link, h5py.SoftLink):
            links.append({
                "name": name,
                "source": ".",
                "path": child_link.path,
            })
    return links


def _should_inline(ds: h5py.Dataset) -> bool:
    """Decide whether a dataset should be inlined in the RFS."""
    if ds.ndim == 0:
        return True
    if ds.dtype.kind in ("O", "U", "S"):
        # String/object data - always inline
        return True
    if ds.dtype.kind == "V" and ds.dtype.fields is not None:
        # Compound with reference fields must always be inlined
        # (HDF5 reference handles are opaque — can't use byte-range refs)
        if _compound_has_references(ds.dtype):
            return True
        # Regular compound - inline small ones, use byte-range refs for large
        if ds.size < 1000:
            return True
        return False
    if ds.size < 1000:
        return True
    return False


def _numpy_dtype_to_zarr_v3(dtype: np.dtype) -> str:
    """Convert a numpy dtype to a zarr v3 data_type string."""
    return zarr_data_type(dtype)


def _compound_dtype_to_zarr_v3(dtype: np.dtype) -> dict:
    """Convert a numpy structured dtype to a zarr v3 struct data_type dict.

    struct is the registered data type for records of named fields
    (https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/struct).
    Its earlier name, structured, with fields as [name, type] pairs, may be
    read but is not to be written.
    """
    fields = []
    for field_name in dtype.names:
        field_dtype = dtype[field_name]
        if field_dtype.kind == "S":
            zarr_type: str | dict = {
                "name": "null_terminated_bytes",
                "configuration": {"length_bytes": field_dtype.itemsize},
            }
        elif field_dtype.kind == "U":
            # Fixed-length unicode (e.g. resolved reference paths)
            zarr_type = {
                "name": "fixed_length_utf32",
                "configuration": {"length_bytes": field_dtype.itemsize},
            }
        else:
            zarr_type = _numpy_dtype_to_zarr_v3(field_dtype)
        fields.append({"name": field_name, "data_type": zarr_type})

    return {
        "name": "struct",
        "configuration": {"fields": fields},
    }


def _compound_has_references(dtype: np.dtype) -> bool:
    """Check if a compound dtype has any reference fields."""
    for field_name in dtype.names:
        if h5py.check_dtype(ref=dtype[field_name]) == h5py.Reference:
            return True
    return False


def _get_reference_fields(dtype: np.dtype) -> list[str]:
    """Return names of reference fields in a compound dtype."""
    return [
        name for name in dtype.names
        if h5py.check_dtype(ref=dtype[name]) == h5py.Reference
    ]


def _resolve_compound_references(
    data: np.ndarray,
    dtype: np.dtype,
    ref_fields: list[str],
    h5f: h5py.File,
) -> tuple[np.ndarray, np.dtype]:
    """Resolve reference fields in compound data to path strings.

    Returns a new array with reference fields replaced by fixed-length
    Unicode strings containing target paths, and the new dtype.
    """
    # First pass: resolve all references to find max path length per field
    resolved: dict[str, list[str]] = {name: [] for name in ref_fields}
    flat = data.ravel()
    for i in range(len(flat)):
        for name in ref_fields:
            val = flat[i][name]
            if isinstance(val, h5py.Reference):
                resolved[name].append(h5f[val].name)
            else:
                resolved[name].append("")

    # Build new dtype with string fields replacing reference fields
    new_fields = []
    for name in dtype.names:
        if name in ref_fields:
            max_len = max((len(s) for s in resolved[name]), default=1)
            new_fields.append((name, f"U{max_len}"))
        else:
            new_fields.append((name, dtype[name]))
    new_dtype = np.dtype(new_fields)

    # Build new array
    new_data = np.empty(data.shape, dtype=new_dtype)
    new_flat = new_data.ravel()
    for i in range(len(flat)):
        vals = []
        for name in dtype.names:
            if name in ref_fields:
                vals.append(resolved[name][i])
            else:
                vals.append(flat[i][name])
        new_flat[i] = tuple(vals)

    return new_data, new_dtype


def _encode_compound_fill_value(dtype: np.dtype) -> str:
    """Encode a compound fill value as base64 zero bytes."""
    return base64.b64encode(b"\x00" * dtype.itemsize).decode("ascii")


def _encode_fill_value(fill_value: Any, dtype: np.dtype) -> Any:
    """Encode a fill value for JSON serialization."""
    if fill_value is None:
        return 0 if dtype.kind in ("i", "u", "f") else None
    if isinstance(fill_value, (np.integer,)):
        return int(fill_value)
    if isinstance(fill_value, (np.floating,)):
        v = float(fill_value)
        if np.isnan(v):
            return "NaN"
        if v == float("inf"):
            return "Infinity"
        if v == float("-inf"):
            return "-Infinity"
        return v
    if isinstance(fill_value, (np.bool_,)):
        return bool(fill_value)
    if isinstance(fill_value, bytes):
        return fill_value.decode("utf-8", errors="replace")
    return fill_value


def _encode_vlen_utf8(strings: list[str]) -> bytes:
    """Encode a list of strings in numcodecs vlen-utf8 format.

    Format: 4-byte LE count, then for each string: 4-byte LE length + utf8 bytes.
    """
    import struct

    parts = [struct.pack("<I", len(strings))]
    for s in strings:
        encoded = s.encode("utf-8")
        parts.append(struct.pack("<I", len(encoded)))
        parts.append(encoded)
    return b"".join(parts)


def _add_dtype_attrs(refs: dict) -> None:
    """Set the _DTYPE attribute on every non-compound array, as hdmf-zarr does.

    Values follow hdmf-zarr: the numpy type name for numeric arrays and "str"
    for strings. Compound arrays carry their fields in the struct
    data_type and get no _DTYPE. Arrays that already have one (object
    references) are left alone.
    """
    for key, val in refs.items():
        if not (key == "zarr.json" or key.endswith("/zarr.json")):
            continue
        meta = json.loads(val)
        if meta.get("node_type") != "array":
            continue
        data_type = meta["data_type"]
        attrs = meta.setdefault("attributes", {})
        if not isinstance(data_type, str) or "_DTYPE" in attrs:
            continue
        attrs["_DTYPE"] = "str" if data_type == "string" else data_type
        refs[key] = json.dumps(meta, separators=(",", ":"))
