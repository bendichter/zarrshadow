"""Utilities for reading HDF5 chunk byte ranges.

Ported from lindi (which was adapted from kerchunk).
"""

from __future__ import annotations

import math
import warnings
from typing import Callable

import h5py
import numpy as np


def get_max_num_chunks(*, shape: tuple[int, ...], chunk_size: tuple[int, ...]) -> int:
    """Get the maximum number of chunks in an h5py dataset.

    Similar to h5_dataset.id.get_num_chunks() but significantly faster. Does
    not account for whether some chunks are allocated.
    """
    if np.prod(chunk_size) == 0:
        return 0
    return math.prod([math.ceil(a / b) for a, b in zip(shape, chunk_size)])


def apply_to_all_chunk_info(h5_dataset: h5py.Dataset, callback: Callable) -> None:
    """Apply callback to each chunk of an h5py dataset.

    Tries chunk_iter first (requires HDF5 1.12.3+), falls back to
    get_chunk_info which is significantly slower.
    """
    assert h5_dataset.chunks is not None
    dsid = h5_dataset.id
    try:
        dsid.chunk_iter(callback)
    except AttributeError:
        num_chunks = dsid.get_num_chunks()
        if num_chunks > 100:
            warnings.warn(
                f"Dataset {h5_dataset.name} has {num_chunks} chunks. "
                f"Using get_chunk_info is slow. Consider upgrading to HDF5 1.12.3+."
            )
        for index in range(num_chunks):
            chunk_info = dsid.get_chunk_info(index)
            callback(chunk_info)


def get_chunk_byte_range(
    h5_dataset: h5py.Dataset, chunk_coords: tuple[int, ...]
) -> tuple[int, int]:
    """Get (byte_offset, byte_count) for a chunk at the given coordinates."""
    shape = h5_dataset.shape
    chunk_shape = h5_dataset.chunks
    assert chunk_shape is not None

    chunk_coords_shape = [
        (shape[i] + chunk_shape[i] - 1) // chunk_shape[i] if chunk_shape[i] != 0 else 0
        for i in range(len(shape))
    ]
    ndim = h5_dataset.ndim
    assert len(chunk_coords) == ndim
    chunk_index = 0
    for i in range(ndim):
        chunk_index += int(chunk_coords[i] * np.prod(chunk_coords_shape[i + 1 :]))
    return _get_chunk_byte_range_for_chunk_index(h5_dataset, chunk_index)


def _get_chunk_byte_range_for_chunk_index(
    h5_dataset: h5py.Dataset, chunk_index: int
) -> tuple[int, int]:
    """Get (byte_offset, byte_count) for a chunk by linear index."""
    dsid = h5_dataset.id
    chunk_info = dsid.get_chunk_info(chunk_index)
    return chunk_info.byte_offset, chunk_info.size


def get_byte_range_for_contiguous_dataset(
    h5_dataset: h5py.Dataset,
) -> tuple[int, int]:
    """Get (byte_offset, byte_count) for a contiguous (non-chunked) dataset."""
    dsid = h5_dataset.id
    byte_offset = dsid.get_offset()
    byte_count = dsid.get_storage_size()
    return byte_offset, byte_count
