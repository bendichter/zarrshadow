"""Generate a zarr v3 reference file system (RFS) from an HDF5 file.

The RFS is a JSON-serializable dict that describes the HDF5 file's group/array
hierarchy using zarr v3 metadata, with chunk references pointing to byte ranges
in the original HDF5 file. This allows zarr to read HDF5 data without copying.

The RFS follows the unified convention from:
  https://github.com/NeurodataWithoutBorders/lindi/issues/125
  https://github.com/hdmf-dev/hdmf-zarr/issues/335
"""

from __future__ import annotations

import base64
import json
import os
import shutil
from typing import Any

import h5py
import numpy as np
import zarr
from tqdm import tqdm
from zarr.codecs import BloscCodec

from .attr_conversion import h5_attr_to_zarr
from .chunk_index import MISSING, ChunkIndex, build_index, index_block_shape
from .h5_chunk_utils import (
    apply_to_all_chunk_info,
    get_byte_range_for_contiguous_dataset,
    get_max_num_chunks,
)
from .h5_filters_to_codecs import h5_filters_to_codec_pipeline


def generate_rfs(
    hdf5_url_or_path: str,
    *,
    local_hdf5_path: str | None = None,
    h5f: h5py.File | None = None,
    chunk_index_threshold: int | None = 1000,
) -> dict:
    """Generate a zarr v3 reference file system from an HDF5 file.

    Parameters
    ----------
    hdf5_url_or_path : str
        URL or local path of the HDF5 file. If a URL (http/https), it is
        used both as the source for reading metadata and as the target for
        chunk references. For remote files, zindi uses its built-in Remfile
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
        byte ranges, see ``zindi.chunk_index``) in place of one ref per chunk.
        None lists every chunk in refs.

    Returns
    -------
    dict
        A reference file system dict with keys "refs" and "version", and
        "indexes" mapping array paths to {"url", "index"} when any array is
        indexed, in which case "version" is 2.
    """
    refs: dict[str, Any] = {}
    indexes: dict[str, dict] = {}
    opts = {"indexes": indexes, "chunk_index_threshold": chunk_index_threshold}

    if h5f is not None:
        _process_group(h5f, "", refs, hdf5_url_or_path, h5f, **opts)
    elif local_hdf5_path is not None:
        with h5py.File(local_hdf5_path, "r") as opened:
            _process_group(opened, "", refs, hdf5_url_or_path, opened, **opts)
    elif hdf5_url_or_path.startswith("http://") or hdf5_url_or_path.startswith("https://"):
        from .remfile import ZindiRemfile

        remf = ZindiRemfile(hdf5_url_or_path)
        with h5py.File(remf, "r") as opened:
            _process_group(opened, "", refs, hdf5_url_or_path, opened, **opts)
    else:
        with h5py.File(hdf5_url_or_path, "r") as opened:
            _process_group(opened, "", refs, hdf5_url_or_path, opened, **opts)

    _add_dtype_attrs(refs)
    rfs: dict[str, Any] = {"refs": refs, "version": 2 if indexes else 1}
    if indexes:
        rfs["indexes"] = indexes
    _apply_templates(rfs)
    return rfs


def write_rfs(rfs: dict, output_path: str) -> None:
    """Write a reference file system to disk.

    A path ending in ".json" writes a single version 1 reference file in which
    every chunk of an indexed array is listed in refs, readable by any kerchunk
    reader. Any other path writes a directory holding refs.json and, for each
    chunk index, a zarr v3 array under index/<array path>. The "indexes" entry
    in refs.json gives that path, relative to refs.json, and the file is marked
    version 2 so that readers without index support refuse it. Opening the
    directory reads only refs.json; index chunks are read when needed.
    """
    if output_path.endswith(".json"):
        with open(output_path, "w") as f:
            json.dump(_expand_indexes(rfs), f, indent=2, sort_keys=True)
        return

    refs_json = os.path.join(output_path, "refs.json")
    if os.path.isdir(output_path) and os.listdir(output_path) and not os.path.exists(refs_json):
        raise FileExistsError(f"{output_path} exists and is not a zindi RFS directory")
    index_dir = os.path.join(output_path, "index")
    if os.path.isdir(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(output_path, exist_ok=True)

    indexes = rfs.get("indexes", {})
    header = {k: v for k, v in rfs.items() if k != "indexes"}
    if indexes:
        header["version"] = 2
        header["indexes"] = {p: {"url": e["url"], "index": f"index/{p}"} for p, e in indexes.items()}
    with open(refs_json, "w") as f:
        json.dump(header, f, indent=2, sort_keys=True)

    store = zarr.storage.LocalStore(index_dir)
    for path, entry in indexes.items():
        data = np.asarray(ChunkIndex(entry["url"], entry["index"]).array[...])
        arr = zarr.create_array(
            store,
            name=path,
            shape=data.shape,
            dtype="uint64",
            chunks=(*index_block_shape(data.shape[:-1]), 2),
            fill_value=int(MISSING),
            compressors=BloscCodec(cname="zstd", clevel=5, shuffle="shuffle", typesize=8),
            zarr_format=3,
        )
        arr[...] = data


def _expand_indexes(rfs: dict) -> dict:
    """Return a version 1 copy of rfs with every indexed chunk listed in refs."""
    indexes = rfs.get("indexes")
    if not indexes:
        return rfs
    templates = rfs.get("templates", {})

    def expand(url: str) -> str:
        for key, value in templates.items():
            url = url.replace("{{" + key + "}}", value)
        return url

    refs = {
        key: [expand(val[0]), val[1], val[2]] if isinstance(val, list) and len(val) == 3 else val
        for key, val in rfs["refs"].items()
    }
    for path, entry in indexes.items():
        for coords, offset, nbytes in ChunkIndex(entry["url"], entry["index"]).iter_chunks():
            refs[f"{path}/c/" + "/".join(map(str, coords))] = [entry["url"], offset, nbytes]
    out = {k: v for k, v in rfs.items() if k not in ("indexes", "templates", "refs")}
    out["version"] = 1
    out["refs"] = refs
    _apply_templates(out)
    return out


# ---------------------------------------------------------------------------
# Internal: recursive group/dataset processing
# ---------------------------------------------------------------------------


def _chunk_key(ndim: int) -> str:
    """Build the zarr v3 chunk key for the first chunk of an array.

    A zero-dimensional array has a single chunk keyed "c". An array with
    ndim dimensions keys its first chunk "c/0/.../0" with one index per
    dimension.
    """
    if ndim == 0:
        return "c"
    return "c/" + "/".join(["0"] * ndim)


def _process_group(
    item: h5py.Group,
    path: str,
    refs: dict,
    url: str,
    h5f: h5py.File,
    **opts: Any,
) -> None:
    """Process an HDF5 group, adding zarr v3 metadata to refs."""
    # Check for soft link - if so, record in parent's _LINKS and skip children
    if path:
        link = h5f.get("/" + path, getlink=True)
        if isinstance(link, h5py.SoftLink):
            # Soft links are handled via _LINKS on the parent group
            # (added by the parent's processing). Don't recurse into the target.
            return

    # Build group zarr.json
    attrs = _collect_attrs(item, h5f=h5f, label=path or "(root)")

    # hdmf-zarr stores the root .specloc as a plain path, not a reference
    specloc = attrs.get(".specloc")
    if not path and isinstance(specloc, dict) and "_REFERENCE" in specloc:
        attrs[".specloc"] = specloc["_REFERENCE"]["path"].lstrip("/")

    # Collect _LINKS for any child soft links (unified convention)
    links = _collect_child_links(item, h5f)
    if links:
        attrs["_LINKS"] = links

    group_meta = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": attrs,
    }
    meta_key = f"{path}/zarr.json" if path else "zarr.json"
    refs[meta_key] = json.dumps(group_meta, separators=(",", ":"))

    # Process children
    for name in item.keys():
        child_path = f"{path}/{name}" if path else name

        # Check if this child is a soft link
        child_link = h5f.get("/" + child_path, getlink=True)
        if isinstance(child_link, h5py.SoftLink):
            # Already recorded in parent _LINKS; skip
            continue

        child = item[name]
        if isinstance(child, h5py.Group):
            _process_group(child, child_path, refs, url, h5f, **opts)
        elif isinstance(child, h5py.Dataset):
            _process_dataset(child, child_path, refs, url, h5f, **opts)


def _process_dataset(
    ds: h5py.Dataset,
    path: str,
    refs: dict,
    url: str,
    h5f: h5py.File,
    *,
    indexes: dict,
    chunk_index_threshold: int | None,
) -> None:
    """Process an HDF5 dataset, adding zarr v3 array metadata and chunk refs."""
    attrs = _collect_attrs(ds, h5f=h5f, label=path)

    shape = list(ds.shape)
    dtype = ds.dtype
    is_scalar = ds.ndim == 0

    # Determine if this should be inlined
    inline = _should_inline(ds)

    if inline:
        _process_inline_dataset(ds, path, refs, attrs, is_scalar, h5f)
        return

    # Build codec pipeline
    codec_pipeline = h5_filters_to_codec_pipeline(ds)

    # Determine chunks
    chunks = list(ds.chunks) if ds.chunks else list(ds.shape)
    # Zarr doesn't allow zero-size chunks
    chunks = [max(c, 1) for c in chunks]

    # Zarr v3 data_type
    if dtype.kind == "V" and dtype.fields is not None:
        # Compound dtype — zarr v3's structured data_type carries field info natively
        data_type = _compound_dtype_to_zarr_v3(dtype)
        fill_value = _encode_compound_fill_value(dtype)
    else:
        data_type = _numpy_dtype_to_zarr_v3(dtype)
        fill_value = _encode_fill_value(ds.fillvalue, dtype)

    array_meta: dict[str, Any] = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": data_type,
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": chunks},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": fill_value,
        "codecs": codec_pipeline,
        "attributes": attrs,
        "storage_transformers": [],
    }

    refs[f"{path}/zarr.json"] = json.dumps(array_meta, separators=(",", ":"))

    # Add chunk references
    if np.prod(ds.shape) > 0:
        _add_chunk_refs(ds, path, refs, url, indexes, chunk_index_threshold)


def _process_inline_dataset(
    ds: h5py.Dataset,
    path: str,
    refs: dict,
    attrs: dict,
    is_scalar: bool,
    h5f: h5py.File,
) -> None:
    """Process a small dataset by inlining its data."""
    data = ds[()]

    if is_scalar:
        shape = []
        if isinstance(data, h5py.Reference):
            # Scalar object reference — store target path as plain string
            attrs["_DTYPE"] = "object_reference"
            target = h5f[data]
            array_meta = _make_string_array_meta(shape, attrs)
            refs[f"{path}/zarr.json"] = json.dumps(array_meta, separators=(",", ":"))
            chunk_bytes = _encode_vlen_utf8([target.name])
            chunk_key = _chunk_key(len(shape))
            refs[f"{path}/{chunk_key}"] = (
                "base64:" + base64.b64encode(chunk_bytes).decode("ascii")
            )
            return
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        if isinstance(data, str):
            # String scalar
            array_meta = _make_string_array_meta(shape, attrs)
            refs[f"{path}/zarr.json"] = json.dumps(array_meta, separators=(",", ":"))
            chunk_bytes = _encode_vlen_utf8([data])
            chunk_key = _chunk_key(len(shape))
            refs[f"{path}/{chunk_key}"] = (
                "base64:" + base64.b64encode(chunk_bytes).decode("ascii")
            )
            return
        else:
            data = np.asarray(data)
    else:
        if h5py.check_dtype(ref=ds.dtype) == h5py.Reference:
            # Object reference array — store target paths as plain strings
            data = ds[...]
            path_strs = []
            for item in np.nditer(data, flags=["refs_ok"]):
                val = item.item()
                if isinstance(val, h5py.Reference):
                    target = h5f[val]
                    path_strs.append(target.name)
                else:
                    path_strs.append("")

            shape = list(ds.shape)
            attrs["_DTYPE"] = "object_reference"
            array_meta = _make_string_array_meta(shape, attrs)
            refs[f"{path}/zarr.json"] = json.dumps(array_meta, separators=(",", ":"))

            chunk_bytes = _encode_vlen_utf8(path_strs)
            chunk_key = _chunk_key(ds.ndim)
            refs[f"{path}/{chunk_key}"] = (
                "base64:" + base64.b64encode(chunk_bytes).decode("ascii")
            )
            return

        if ds.dtype.kind in ("O", "U", "S"):
            # String array
            data = ds[...]
            str_data = []
            for item in np.nditer(data, flags=["refs_ok"]):
                val = item.item()
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                str_data.append(str(val) if val is not None else "")

            shape = list(ds.shape)
            array_meta = _make_string_array_meta(shape, attrs)
            refs[f"{path}/zarr.json"] = json.dumps(array_meta, separators=(",", ":"))

            chunk_bytes = _encode_vlen_utf8(str_data)
            chunk_key = _chunk_key(ds.ndim)
            refs[f"{path}/{chunk_key}"] = (
                "base64:" + base64.b64encode(chunk_bytes).decode("ascii")
            )
            return

    # Compound inline data
    if ds.dtype.kind == "V" and ds.dtype.fields is not None:
        shape = list(data.shape)
        dtype = data.dtype

        # Check for reference fields — resolve to path strings
        ref_fields = _get_reference_fields(dtype)
        if ref_fields:
            data, dtype = _resolve_compound_references(data, dtype, ref_fields, h5f)
            attrs["_REFERENCE_FIELDS"] = ref_fields

        data_type = _compound_dtype_to_zarr_v3(dtype)
        fill_value = _encode_compound_fill_value(dtype)

        codec_pipeline = [
            {"name": "bytes", "configuration": {"endian": "little"}}
        ]

        array_meta: dict[str, Any] = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": shape,
            "data_type": data_type,
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": shape},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": fill_value,
            "codecs": codec_pipeline,
            "attributes": attrs,
            "storage_transformers": [],
        }

        refs[f"{path}/zarr.json"] = json.dumps(array_meta, separators=(",", ":"))

        # Ensure little-endian byte order
        if dtype.byteorder == ">":
            data = data.astype(dtype.newbyteorder("<"))
        chunk_bytes = data.tobytes()
        chunk_key = _chunk_key(len(shape))
        _add_inline_ref(refs, f"{path}/{chunk_key}", chunk_bytes)
        return

    # Numeric inline data
    shape = list(data.shape)
    dtype = data.dtype
    data_type = _numpy_dtype_to_zarr_v3(dtype)
    fill_value = _encode_fill_value(ds.fillvalue, dtype)

    codec_pipeline = [
        {"name": "bytes", "configuration": {"endian": "little"}}
    ]

    array_meta: dict[str, Any] = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": data_type,
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": shape},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": fill_value,
        "codecs": codec_pipeline,
        "attributes": attrs,
        "storage_transformers": [],
    }

    refs[f"{path}/zarr.json"] = json.dumps(array_meta, separators=(",", ":"))

    # Inline the chunk data
    # Ensure little-endian byte order
    if dtype.byteorder == ">":
        data = data.astype(dtype.newbyteorder("<"))
    chunk_bytes = data.tobytes()
    chunk_key = _chunk_key(len(shape))
    _add_inline_ref(refs, f"{path}/{chunk_key}", chunk_bytes)


def _make_string_array_meta(shape: list[int], attrs: dict) -> dict:
    """Create zarr v3 array metadata for a string array."""
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": "string",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": shape},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": "",
        "codecs": [
            {"name": "vlen-utf8", "configuration": {}},
        ],
        "attributes": attrs,
        "storage_transformers": [],
    }


def _add_chunk_refs(
    ds: h5py.Dataset,
    path: str,
    refs: dict,
    url: str,
    indexes: dict,
    chunk_index_threshold: int | None,
) -> None:
    """Add chunk references, or a chunk index, for a non-inline dataset."""
    if ds.chunks is not None:
        # Chunked dataset
        chunk_size = ds.chunks
        num_chunks = get_max_num_chunks(shape=ds.shape, chunk_size=chunk_size)
        if chunk_index_threshold is not None and num_chunks > chunk_index_threshold:
            _add_chunk_index(ds, path, url, indexes, num_chunks)
            return
        pbar = tqdm(
            total=num_chunks,
            desc=f"Chunk refs for {path}",
            leave=True,
            delay=2,
        )

        def store_chunk_info(chunk_info: Any) -> None:
            chunk_offset = chunk_info.chunk_offset
            byte_offset = chunk_info.byte_offset
            byte_count = chunk_info.size
            # zarr v3 chunk key: c/<idx0>/<idx1>/...
            indices = [str(a // b) for a, b in zip(chunk_offset, chunk_size)]
            chunk_key = f"{path}/c/" + "/".join(indices)
            refs[chunk_key] = [url, byte_offset, byte_count]
            pbar.update()

        apply_to_all_chunk_info(ds, store_chunk_info)
        pbar.close()
    else:
        # Contiguous dataset - single chunk
        byte_offset, byte_count = get_byte_range_for_contiguous_dataset(ds)
        chunk_key = f"{path}/{_chunk_key(ds.ndim)}"
        refs[chunk_key] = [url, byte_offset, byte_count]


def _add_chunk_index(
    ds: h5py.Dataset,
    path: str,
    url: str,
    indexes: dict,
    num_chunks: int,
) -> None:
    """Record a dataset's chunk byte ranges in an index array."""
    chunk_size = ds.chunks
    grid_shape = tuple(-(-a // b) for a, b in zip(ds.shape, chunk_size))
    pbar = tqdm(total=num_chunks, desc=f"Chunk index for {path}", leave=True, delay=2)

    def for_each_chunk(set_chunk: Any) -> None:
        def store_chunk_info(chunk_info: Any) -> None:
            coords = tuple(a // b for a, b in zip(chunk_info.chunk_offset, chunk_size))
            set_chunk(coords, chunk_info.byte_offset, chunk_info.size)
            pbar.update()

        apply_to_all_chunk_info(ds, store_chunk_info)

    indexes[path] = {"url": url, "index": build_index(grid_shape, for_each_chunk)}
    pbar.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    kind = dtype.kind
    itemsize = dtype.itemsize
    mapping = {
        ("f", 2): "float16",
        ("f", 4): "float32",
        ("f", 8): "float64",
        ("i", 1): "int8",
        ("i", 2): "int16",
        ("i", 4): "int32",
        ("i", 8): "int64",
        ("u", 1): "uint8",
        ("u", 2): "uint16",
        ("u", 4): "uint32",
        ("u", 8): "uint64",
        ("b", 1): "bool",
    }
    result = mapping.get((kind, itemsize))
    if result is None:
        raise ValueError(f"Unsupported dtype for zarr v3: {dtype}")
    return result


def _compound_dtype_to_zarr_v3(dtype: np.dtype) -> dict:
    """Convert a numpy structured dtype to a zarr v3 structured data_type dict."""
    fields = []
    for field_name in dtype.names:
        field_dtype = dtype[field_name]
        if field_dtype.kind == "S":
            zarr_type = {
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
        fields.append([field_name, zarr_type])

    return {
        "name": "structured",
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


def _add_inline_ref(refs: dict, key: str, data: bytes) -> None:
    """Add inline data to refs, base64-encoding if needed."""
    if data.startswith(b"base64:"):
        refs[key] = "base64:" + base64.b64encode(data).decode("ascii")
    else:
        try:
            refs[key] = data.decode("ascii")
        except UnicodeDecodeError:
            refs[key] = "base64:" + base64.b64encode(data).decode("ascii")


def _add_dtype_attrs(refs: dict) -> None:
    """Set the _DTYPE attribute on every non-compound array, as hdmf-zarr does.

    Values follow hdmf-zarr: the numpy type name for numeric arrays and "str"
    for strings. Compound arrays carry their fields in the structured
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


def _apply_templates(rfs: dict) -> None:
    """Replace frequently-used URLs with template placeholders."""
    refs = rfs["refs"]
    url_counts: dict[str, int] = {}
    for val in refs.values():
        if isinstance(val, list) and len(val) == 3:
            url = val[0]
            url_counts[url] = url_counts.get(url, 0) + 1

    templates: dict[str, str] = {}
    template_idx = 0
    for url, count in url_counts.items():
        if count >= 5:
            template_key = f"u{template_idx}"
            templates[template_key] = url
            template_idx += 1

    if not templates:
        return

    # Build reverse lookup
    url_to_template = {url: key for key, url in templates.items()}

    # Replace URLs with template references
    for ref_key, val in refs.items():
        if isinstance(val, list) and len(val) == 3:
            url = val[0]
            if url in url_to_template:
                val[0] = "{{" + url_to_template[url] + "}}"

    rfs["templates"] = templates
