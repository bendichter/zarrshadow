"""Statistics of blocks of an array, and finding a range of a sorted array with them."""

import warnings
from datetime import datetime, timezone

import h5py
import numpy as np
import pytest
import zarr

from shadowzarr import generate_rfs, open_rfs, write_rfs
from shadowzarr.block_statistics import (
    add_block_statistics,
    compute_minmax,
    find_range,
    load_block_statistics,
    resolve_block_shape,
    write_block_statistics,
)
from shadowzarr.open_rfs import load_rfs


def _zarr(data, chunks):
    """data in an array at "a" in a new in-memory group."""
    root = zarr.open_group(zarr.storage.MemoryStore(), mode="w", zarr_format=3)
    array = root.create_array("a", shape=data.shape, dtype=data.dtype, chunks=chunks)
    array[...] = data
    return array


def _expected_minmax(data, block_shape):
    """Every block's minimum and maximum, one block at a time."""
    grid = [-(-s // b) for s, b in zip(data.shape, block_shape)]
    out = np.empty(grid + [2], dtype=data.dtype)
    for coords in np.ndindex(*grid):
        block = data[tuple(slice(i * b, (i + 1) * b) for i, b in zip(coords, block_shape))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[coords] = (np.nanmin(block), np.nanmax(block)) if data.dtype.kind == "f" else (block.min(), block.max())
    return out


# -- Block shapes --


def test_resolve_block_shape():
    assert resolve_block_shape([1000, 8], [100, 4]) == [100, 4]
    assert resolve_block_shape([1000, 8], [100, 4], granularity=3) == [300, 12]
    assert resolve_block_shape([1000, 8], [100, 4], granularity=[10, 1]) == [1000, 4]
    assert resolve_block_shape([1000, 8], [100, 4], block_shape=[None, 1]) == [1000, 1]
    assert resolve_block_shape([1000, 8], [100, 4], block_shape=[25, None]) == [25, 8]
    # A block covering the whole dimension need not fit the chunks
    assert resolve_block_shape([1000, 8], [100, 4], block_shape=[1000, 8]) == [1000, 8]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"block_shape": [30, 4]}, "neither divides"),
        ({"block_shape": [150, 4]}, "neither divides"),
        ({"block_shape": [0, 4]}, "at least 1"),
        ({"block_shape": [100]}, "dimensions"),
        ({"block_shape": [100, 4], "granularity": 2}, "not both"),
    ],
)
def test_resolve_block_shape_errors(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resolve_block_shape([1000, 8], [100, 4], **kwargs)


# -- minmax --


@pytest.mark.parametrize(
    "block_shape",
    [[100, 4], [50, 1], [25, 2], [300, 8], [1050, 1], [1, 8], [1050, 8]],
)
@pytest.mark.parametrize("read_bytes", [1, 2000, 2**26])
def test_minmax_matches_numpy(block_shape, read_bytes):
    """Blocks smaller than chunks, larger, per column, per row, and the whole array; 1050 rows leave a short last chunk.

    read_bytes reads one chunk, a few, or the whole array at a time.
    """
    data = np.random.default_rng(1).integers(-30000, 30000, (1050, 8)).astype("int16")
    array = _zarr(data, (100, 4))
    result = compute_minmax(array, block_shape, read_bytes=read_bytes)
    np.testing.assert_array_equal(result, _expected_minmax(data, block_shape))


def test_minmax_keeps_the_data_type_exactly():
    data = np.array([2**62 + 1, 2**62 + 3, 2**62 + 2, -(2**62)], dtype="int64")
    result = compute_minmax(_zarr(data, (2,)), [2])
    assert result.dtype == np.int64
    np.testing.assert_array_equal(result, [[2**62 + 1, 2**62 + 3], [-(2**62), 2**62 + 2]])


def test_minmax_ignores_nan():
    data = np.arange(40, dtype="float64").reshape(20, 2)
    data[0:5] = np.nan  # a whole block
    data[7, 1] = np.nan  # one value
    result = compute_minmax(_zarr(data, (10, 2)), [5, 1])
    np.testing.assert_array_equal(result, _expected_minmax(data, [5, 1]))
    assert np.isnan(result[0]).all()


def test_minmax_of_a_missing_chunk_is_the_fill_value():
    root = zarr.open_group(zarr.storage.MemoryStore(), mode="w", zarr_format=3)
    array = root.create_array("a", shape=(30,), dtype="int32", chunks=(10,), fill_value=7)
    array[0:10] = np.arange(10)
    np.testing.assert_array_equal(compute_minmax(array, [10]), [[0, 9], [7, 7], [7, 7]])


# -- find_range --


def _sorted_times(n=50_000, seed=2):
    rng = np.random.default_rng(seed)
    times = np.cumsum(rng.exponential(0.01, n))
    times[1000:1010] = times[1000]  # repeated values, also across a block boundary below
    return np.sort(times)


@pytest.mark.parametrize("block", [1000, 2500, 5000, 10_000, 50_000])
def test_find_range_matches_searchsorted(block):
    times = _sorted_times()
    array = _zarr(times, (5000,))
    write_block_statistics(array, block_shape=[block], sorted="ascending")
    for low, high in [(1.0, 2.5), (times[1000], times[1000]), (-5, 0.5), (times[-1] - 1, times[-1] + 1),
                      (times[-1] + 1, times[-1] + 2), (-2, -1), (3.0, 2.0), (times[999], times[1009])]:
        expected = slice(int(np.searchsorted(times, low, "left")), int(np.searchsorted(times, high, "right")))
        found = find_range(array, low, high)
        assert (found.start, found.stop) == (expected.start, max(expected.start, expected.stop)), (low, high)


def test_find_range_reads_only_the_statistics_and_two_blocks():
    times = _sorted_times()
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w", zarr_format=3)
    array = root.create_array("t", shape=times.shape, dtype=times.dtype, chunks=(1000,))
    array[...] = times
    write_block_statistics(array, sorted="ascending")

    read = []

    class Counting(zarr.storage.WrapperStore):
        async def get(self, key, prototype, byte_range=None):
            read.append(key)
            return await super().get(key, prototype, byte_range)

    counted = zarr.open_array(Counting(store), path="t", mode="r")
    result = find_range(counted, 100.0, 101.0)
    assert result == slice(int(np.searchsorted(times, 100.0)), int(np.searchsorted(times, 101.0, "right")))
    data_chunks = [k for k in read if k.startswith("t/c/")]
    assert 1 <= len(data_chunks) <= 2
    assert len(times) // 1000 == 50


def test_find_range_without_statistics_reads_the_whole_array():
    times = _sorted_times(5000)
    array = _zarr(times, (1000,))
    assert find_range(array, 10.0, 20.0) == slice(int(np.searchsorted(times, 10.0)), int(np.searchsorted(times, 20.0, "right")))
    # Statistics without "sorted" are not used for bisection
    write_block_statistics(array)
    assert find_range(array, 10.0, 20.0) == slice(int(np.searchsorted(times, 10.0)), int(np.searchsorted(times, 20.0, "right")))


# -- In a zarr store --


def test_write_block_statistics_layout(tmp_path):
    root = zarr.open_group(str(tmp_path / "a.zarr"), mode="w", zarr_format=3)
    data = np.random.default_rng(3).normal(size=(3000, 4))
    array = root.create_array("acquisition/es/data", shape=data.shape, dtype=data.dtype, chunks=(1000, 4))
    array[...] = data
    attribute = write_block_statistics(array, block_shape=[500, 1])
    assert attribute == {
        "statistics": [{"type": "minmax", "block_shape": [500, 1], "path": ".stats/data/minmax"}]
    }
    reopened = zarr.open_group(str(tmp_path / "a.zarr"), mode="r")
    stats = reopened["acquisition/es/.stats/data/minmax"]
    assert stats.shape == (6, 4, 2)
    assert stats.attrs["block_shape"] == [500, 1]
    block_shape, values = load_block_statistics(reopened["acquisition/es/data"])
    assert block_shape == [500, 1]
    np.testing.assert_array_equal(values, _expected_minmax(data, [500, 1]))
    # A copy of the parent group carries the statistics
    import shutil

    copy_root = zarr.open_group(str(tmp_path / "b.zarr"), mode="w", zarr_format=3)
    shutil.copytree(tmp_path / "a.zarr" / "acquisition" / "es", tmp_path / "b.zarr" / "moved")
    copied = zarr.open_group(str(tmp_path / "b.zarr"), mode="r")["moved/data"]
    np.testing.assert_array_equal(load_block_statistics(copied)[1], values)
    assert copy_root is not None
    # Writing again replaces the entry rather than adding a second one
    attribute = write_block_statistics(array, block_shape=[1000, 4])
    assert [e["block_shape"] for e in attribute["statistics"]] == [[1000, 4]]


# -- In a reference file system --


@pytest.fixture
def h5_times(tmp_path):
    path = tmp_path / "times.h5"
    times = _sorted_times(40_000)
    with h5py.File(path, "w") as f:
        f.create_dataset("chunked", data=times, chunks=(4000,))
        f.create_dataset("contiguous", data=times)
        f.create_dataset("signal", data=np.random.default_rng(4).integers(-500, 500, (12_000, 4)).astype("int16"),
                         chunks=(3000, 4), compression="gzip")
    return str(path), times


def _check_times(group, times):
    for name in ["chunked", "contiguous"]:
        array = group[name]
        assert array.attrs["block_statistics"]["sorted"] == "ascending"
        found = find_range(array, 50.0, 60.0)
        assert found == slice(int(np.searchsorted(times, 50.0)), int(np.searchsorted(times, 60.0, "right")))


@pytest.mark.parametrize("output", [None, "directory", "refs.json"])
def test_add_block_statistics_to_rfs(h5_times, tmp_path, output):
    path, times = h5_times
    # The contiguous dataset becomes slabs of 4000 values; its blocks are a quarter of a slab
    rfs = generate_rfs(path, contiguous_chunk_bytes=32_000)
    rfs = add_block_statistics(rfs, {
        "chunked": {"block_shape": [1000], "sorted": "ascending"},
        "contiguous": {"block_shape": [1000], "sorted": "ascending"},
        "signal": {"block_shape": [500, 1]},
    })
    if output:
        out = str(tmp_path / ("out.shadowzarr" if output == "directory" else "refs.json"))
        write_rfs(rfs, out)
        rfs = load_rfs(out)
    group = open_rfs(rfs)
    _check_times(group, times)
    _, values = load_block_statistics(group["signal"])
    np.testing.assert_array_equal(values, _expected_minmax(group["signal"][...], [500, 1]))
    stats = group[".stats/chunked/minmax"]
    assert stats.attrs["array"] == "chunked"
    assert path in str(stats.attrs["sources"])


def test_add_block_statistics_does_not_change_its_input(h5_times):
    path, _ = h5_times
    rfs = generate_rfs(path)
    before = repr(rfs)
    add_block_statistics(rfs, {"chunked": {}})
    assert repr(rfs) == before


def test_block_statistics_through_icechunk(h5_times, tmp_path):
    icechunk = pytest.importorskip("icechunk")
    pytest.importorskip("virtualizarr.parsers.icechunk")
    from shadowzarr.icechunk import rfs_to_icechunk

    path, times = h5_times
    rfs = add_block_statistics(generate_rfs(path, contiguous_chunk_bytes=32_000), {
        "chunked": {"block_shape": [1000], "sorted": "ascending"},
        "contiguous": {"block_shape": [1000], "sorted": "ascending"},
    })
    prefix = f"file://{tmp_path}/"
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(icechunk.VirtualChunkContainer(prefix, icechunk.local_filesystem_store(str(tmp_path))))
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path / "repo")),
        config=config,
        authorize_virtual_chunk_access={prefix: icechunk.credentials.LocalFileSystemAccess},
    )
    session = repo.writable_session("main")
    rfs_to_icechunk(rfs, session)
    session.commit("Add references with block statistics")
    group = zarr.open_group(repo.readonly_session("main").store, mode="r")
    _check_times(group, times)


def test_nwb_with_block_statistics_reads_with_hdmf_zarr(tmp_path):
    """NWBZarrIO reads a file with block statistics; it only warns about the .stats arrays.

    hdmf-zarr reads every group, including .stats, and warns that its arrays
    have no hdmf dtype attribute. Skipping .stats there would remove the warnings.
    """
    pynwb = pytest.importorskip("pynwb")
    hdmf_zarr_nwb = pytest.importorskip("hdmf_zarr.nwb")
    from shadowzarr.rfs_store import RfsStore

    path = str(tmp_path / "test.nwb")
    times = _sorted_times(20_000)
    nwb = pynwb.NWBFile(session_description="d", identifier="i",
                        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc))
    nwb.add_acquisition(pynwb.TimeSeries(name="ts", data=np.arange(len(times), dtype="float32"),
                                         unit="V", timestamps=times))
    with pynwb.NWBHDF5IO(path, "w") as io:
        io.write(nwb)
    rfs = add_block_statistics(generate_rfs(path), {
        "acquisition/ts/timestamps": {"block_shape": [1000], "sorted": "ascending"},
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        io = hdmf_zarr_nwb.NWBZarrIO(RfsStore(rfs), mode="r")
        read = io.read()
    unexpected = [str(w.message) for w in caught if "/.stats/" not in str(w.message)]
    assert not unexpected, unexpected
    np.testing.assert_array_equal(read.acquisition["ts"].timestamps[:], times)
    io.close()
    group = open_rfs(rfs)
    found = find_range(group["acquisition/ts/timestamps"], 10.0, 11.0)
    assert found == slice(int(np.searchsorted(times, 10.0)), int(np.searchsorted(times, 11.0, "right")))
