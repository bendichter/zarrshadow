"""Tests for detecting whether HDF5 reports offsets relative to the userblock."""

import h5py
import numpy as np
import pytest

from zarrshadow import generate_rfs, open_rfs
from zarrshadow import hdf5 as gr

USERBLOCK = 512


@pytest.fixture
def mat_like(tmp_path):
    """An HDF5 file with a 512-byte userblock, as MATLAB v7.3 files have."""
    path = str(tmp_path / "data.mat")
    rng = np.random.default_rng(0)
    with h5py.File(path, "w", userblock_size=USERBLOCK) as f:
        f.create_dataset("chunked", data=rng.standard_normal(5000), chunks=(500,), compression="gzip")
        f.create_dataset("indexed", data=rng.standard_normal(5000), chunks=(50,))
        f.create_dataset("contiguous", data=rng.standard_normal((1000, 7)))
    return path


def _values(path):
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in f}


def _read_all(rfs):
    root = open_rfs(rfs)
    return {k: root[k][...] for k in ["chunked", "indexed", "contiguous"]}


GENERATE = dict(chunk_index_threshold=20, contiguous_chunk_bytes=8192)


def test_no_userblock_needs_no_check(tmp_path):
    path = str(tmp_path / "plain.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.arange(5000.0), chunks=(500,))
    with h5py.File(path, "r") as f:
        assert gr._detect_offset_shift(f, read_bytes=None) == 0


def test_absolute_offsets_detected(mat_like):
    with h5py.File(mat_like, "r") as f:
        assert gr._detect_offset_shift(f, gr._raw_reader(mat_like)) == 0
    rfs = generate_rfs(mat_like, **GENERATE)
    for k, v in _read_all(rfs).items():
        np.testing.assert_array_equal(v, _values(mat_like)[k], err_msg=k)


def test_superblock_relative_offsets_are_shifted(mat_like, monkeypatch):
    """Simulate HDF5 1.10, which reports offsets 512 bytes too low for this file."""
    real_first = gr._first_stored_block
    real_iter = gr.apply_to_all_chunk_info
    real_contiguous = gr.get_byte_range_for_contiguous_dataset

    def first_block(h5f):
        offset, data = real_first(h5f)
        return offset - USERBLOCK, data

    def apply_to_all(ds, callback):
        real_iter(ds, lambda info: callback(info._replace(byte_offset=info.byte_offset - USERBLOCK)))

    def contiguous(ds):
        offset, count = real_contiguous(ds)
        return offset - USERBLOCK, count

    monkeypatch.setattr(gr, "_first_stored_block", first_block)
    monkeypatch.setattr(gr, "apply_to_all_chunk_info", apply_to_all)
    monkeypatch.setattr(gr, "get_byte_range_for_contiguous_dataset", contiguous)

    with h5py.File(mat_like, "r") as f:
        assert gr._detect_offset_shift(f, gr._raw_reader(mat_like)) == USERBLOCK
    rfs = generate_rfs(mat_like, **GENERATE)
    assert "indexed" in rfs["indexes"] and rfs["gen"]  # every kind of reference is covered
    for k, v in _read_all(rfs).items():
        np.testing.assert_array_equal(v, _values(mat_like)[k], err_msg=k)


def test_offsets_matching_neither_raise(mat_like, monkeypatch):
    monkeypatch.setattr(gr, "_first_stored_block", lambda h5f: (100, b"no such bytes here"))
    with pytest.raises(RuntimeError, match="do not match the file"):
        generate_rfs(mat_like)
