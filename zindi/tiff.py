"""Generate a zarr v3 reference file system (RFS) from a TIFF file.

tifffile knows the TIFF layouts: strips and tiles, multi-page series, OME and
other pyramids, and the compression schemes. This generator asks tifffile for
the kerchunk references of each series and records them through RfsBuilder,
so TIFF files get the same directory form, chunk indexes, evenly spaced
chunk series, and source checks as HDF5 files.

Series i is stored at path "i": an array, or for a pyramid a group of levels
"i/0", "i/1", ..., the layout bioformats2raw uses for OME-Zarr.

TIFF deflate and zstd chunks are ordinary zlib and zstd streams, so they are
written with the standard codec names every Zarr library reads. Other TIFF
compressions (LZW, JPEG, the horizontal predictor, and so on) use the Zarr
codecs from imagecodecs, which zindi registers when it opens such a file.
Other readers need imagecodecs too.
"""

from __future__ import annotations

import io
import json
import os
import posixpath
from typing import Any

from .builder import RfsBuilder
from .gen import Generator

# TIFF compressions whose chunks are standard streams, under codec names any Zarr reader knows
PORTABLE_CODECS = {
    "imagecodecs_zlib": {"name": "numcodecs.zlib", "configuration": {"level": 1}},
    "imagecodecs_deflate": {"name": "numcodecs.zlib", "configuration": {"level": 1}},
    "imagecodecs_zstd": {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
}


def generate_rfs_tiff(
    url_or_path: str,
    *,
    series: int | list[int] | None = None,
    chunk_index_threshold: int | None = 1000,
    record_sources: bool = True,
) -> dict:
    """Generate a zarr v3 reference file system from a TIFF file.

    Parameters
    ----------
    url_or_path : str
        URL or local path of the TIFF file.
    series : int, list of int, or None
        Which series to include. None includes every series.
    chunk_index_threshold : int or None
        Arrays with more chunks than this, unless they are evenly spaced, get a
        chunk index in place of one ref per chunk.
    record_sources : bool
        Record the size and, for remote files, the ETag of the file, so readers
        can detect that it changed.

    Returns
    -------
    dict
        A reference file system dict; see zindi.builder.
    """
    import tifffile

    if url_or_path.startswith(("http://", "https://")):
        from .remfile import ZindiRemfile

        handle: Any = ZindiRemfile(url_or_path)
    else:
        handle = open(url_or_path, "rb")
    try:
        with tifffile.TiffFile(handle) as tif:
            if series is None:
                selected = list(range(len(tif.series)))
            else:
                selected = [series] if isinstance(series, int) else list(series)
            builder = RfsBuilder()
            builder.add_group("")
            for i in selected:
                refs = _series_references(tif.series[i])
                _add_series(builder, refs, str(i), url_or_path, tif.filehandle.name, chunk_index_threshold)
    finally:
        handle.close()
    return builder.build(record_sources=record_sources)


def _series_references(series: Any) -> dict:
    """tifffile's version 1 kerchunk references for one series, in Zarr v3 form.

    tifffile refuses strips that do not divide the image evenly. An uncompressed
    page stored in one piece can instead be a single chunk, so those series are
    retried with one chunk per page.
    """
    from tifffile import CHUNKMODE

    def write(**kwargs: Any) -> dict:
        with series.aszarr(**kwargs) as store:
            buf = io.StringIO()
            store.write_fsspec(buf, url="", zarr_format=3, version=1)
        return json.loads(buf.getvalue())

    try:
        return write()
    except ValueError as e:
        if "incomplete chunks" not in str(e):
            raise
        pages = [page for page in series.pages if page is not None]
        if all(page.compression == 1 and page.is_contiguous for page in pages):
            return write(chunkmode=CHUNKMODE.PAGE)
        raise ValueError(
            f"{series.name or 'series'}: compressed strips or tiles that do not divide the "
            "image evenly cannot be read as Zarr chunks"
        ) from e


def _add_series(
    builder: RfsBuilder,
    refs: dict,
    prefix: str,
    url_or_path: str,
    own_name: str,
    chunk_index_threshold: int | None,
) -> None:
    """Record one series' metadata and chunks under prefix."""
    templates = refs.get("templates", {})
    entries = list(refs["refs"].items())
    for entry in refs.get("gen", []):
        entries.extend(Generator(entry, templates).items())

    arrays: dict[str, dict] = {}
    tables: dict[tuple[str, str], dict[tuple[int, ...], tuple[int, int]]] = {}
    for key, value in entries:
        full = f"{prefix}/{key}"
        if key == "zarr.json" or key.endswith("/zarr.json"):
            meta = json.loads(value)
            path = full[: -len("/zarr.json")]
            if meta.get("node_type") == "array":
                meta["codecs"] = [PORTABLE_CODECS.get(c.get("name"), c) for c in meta["codecs"]]
                arrays[path] = meta
            builder.set_metadata(path, meta)
        elif isinstance(value, list):
            path, coords = _split_chunk_key(full)
            url = _source_url(value[0], templates, own_name, url_or_path)
            tables.setdefault((path, url), {})[coords] = (int(value[1]), int(value[2]))
        else:
            builder.refs[full] = value  # data tifffile stored inline

    for (path, url), table in tables.items():
        meta = arrays[path]
        shape = meta["shape"]
        chunk_shape = meta["chunk_grid"]["configuration"]["chunk_shape"]
        grid = [-(-s // c) for s, c in zip(shape, chunk_shape)]
        builder.add_chunks(path, grid, url, table, index_threshold=chunk_index_threshold)


def _split_chunk_key(key: str) -> tuple[str, tuple[int, ...]]:
    path, sep, coords = key.rpartition("/c/")
    if sep:
        return path, tuple(int(c) for c in coords.split("/"))
    return key[: -len("/c")], ()  # zero-dimensional


def _source_url(ref_url: str, templates: dict, own_name: str, url_or_path: str) -> str:
    """Map a file name in tifffile's references to the file's URL or path."""
    for name, value in templates.items():
        ref_url = ref_url.replace("{{" + name + "}}", value)
    if ref_url in ("", own_name, os.path.basename(own_name)):
        return url_or_path
    # Another file of a multi-file series, next to this one
    if url_or_path.startswith(("http://", "https://")):
        return posixpath.join(posixpath.dirname(url_or_path), ref_url)
    return os.path.join(os.path.dirname(url_or_path), ref_url)
