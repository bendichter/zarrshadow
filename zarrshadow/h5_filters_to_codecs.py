"""Convert HDF5 compression filters to zarr v3 codec specifications.

The zarr v3 codec pipeline consists of:
  1. array-to-bytes codec (e.g., "bytes" for endian encoding)
  2. bytes-to-bytes codecs (e.g., compression like "numcodecs.zlib")

HDF5 stores raw compressed chunks. By declaring the matching codec pipeline
in the zarr v3 metadata, zarr can decompress HDF5 chunks directly without
data conversion.

Adapted from lindi (which adapted from kerchunk).
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np


def h5_filters_to_codec_pipeline(
    h5obj: h5py.Dataset,
) -> list[dict[str, Any]]:
    """Build a zarr v3 codec pipeline for an HDF5 dataset.

    Returns a list of codec dicts suitable for the "codecs" field in
    zarr.json metadata. The pipeline always starts with a "bytes" codec
    for endian encoding, followed by any compression/filter codecs.
    """
    # Start with the bytes codec (array-to-bytes, handles endianness)
    endian = _dtype_endianness(h5obj.dtype)
    pipeline: list[dict[str, Any]] = [
        {"name": "bytes", "configuration": {"endian": endian}}
    ]

    # Add filter/compression codecs (bytes-to-bytes)
    filter_codecs = _h5_filters_to_codecs(h5obj)
    pipeline.extend(filter_codecs)

    return pipeline


def _dtype_endianness(dtype: np.dtype) -> str:
    """Return 'little' or 'big' for a numpy dtype."""
    if dtype.byteorder == ">":
        return "big"
    # '<', '=', '|' all map to little (or single-byte types where it doesn't matter)
    return "little"


def _h5_filters_to_codecs(h5obj: h5py.Dataset) -> list[dict[str, Any]]:
    """Convert HDF5 filters to zarr v3 bytes-to-bytes codec dicts.

    Adapted from lindi and kerchunk.
    """
    if h5obj.scaleoffset:
        raise RuntimeError(
            f"{h5obj.name} uses HDF5 scaleoffset filter - not supported"
        )
    if h5obj.compression in ("szip", "lzf"):
        raise RuntimeError(
            f"{h5obj.name} uses szip or lzf compression - not supported"
        )

    codecs: list[dict[str, Any]] = []

    if h5obj.shuffle and h5obj.dtype.kind != "O":
        codecs.append({
            "name": "numcodecs.shuffle",
            "configuration": {"elementsize": h5obj.dtype.itemsize},
        })

    for filter_id, properties in h5obj._filters.items():
        fid = str(filter_id)
        if fid == "32001":
            # Blosc
            blosc_compressors = (
                "blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd",
            )
            _1, _2, bytes_per_num, total_bytes, clevel, shuffle, compressor = properties
            codecs.append({
                "name": "numcodecs.blosc",
                "configuration": {
                    "blocksize": total_bytes,
                    "clevel": clevel,
                    "shuffle": shuffle,
                    "cname": blosc_compressors[compressor],
                },
            })
        elif fid == "32015":
            # Zstd
            codecs.append({
                "name": "numcodecs.zstd",
                "configuration": {"level": properties[0]},
            })
        elif fid in ("gzip", "zlib"):
            # HDF5 gzip uses raw deflate (zlib), not gzip container format
            level = properties if isinstance(properties, int) else properties[0]
            codecs.append({
                "name": "numcodecs.zlib",
                "configuration": {"level": level},
            })
        elif fid == "32004":
            raise RuntimeError(
                f"{h5obj.name} uses lz4 compression (filter 32004) - not supported"
            )
        elif fid == "32008":
            raise RuntimeError(
                f"{h5obj.name} uses bitshuffle compression (filter 32008) - not supported"
            )
        elif fid == "shuffle":
            # Already handled above
            pass
        elif fid == "fletcher32":
            codecs.append({
                "name": "numcodecs.fletcher32",
                "configuration": {},
            })
        else:
            raise RuntimeError(
                f"{h5obj.name} uses filter id {filter_id} with properties "
                f"{properties} - not supported"
            )

    return codecs
