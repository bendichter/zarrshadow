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
    compute_block_statistics_levels,
    find_range,
    compute_missing_count,
    compute_moments,
    minmax_range,
    moments_range,
    rollup_missing_count,
    rollup_moments,
    rollup_minmax,
    searchsorted,
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


@pytest.mark.parametrize("block", [1000, 5000, 50_000])
@pytest.mark.parametrize("side", ["left", "right"])
def test_searchsorted_matches_numpy(block, side):
    times = _sorted_times()
    array = _zarr(times, (5000,))
    write_block_statistics(array, block_shape=[block], sorted="ascending")
    rng = np.random.default_rng(5)
    values = np.concatenate([
        rng.uniform(times[0] - 1, times[-1] + 1, 300),
        times[[0, 999, 1000, 1005, 1009, 1010, 4999, 5000, len(times) - 1]],  # values in the array, repeats, block edges
        [-np.inf, np.inf],
    ])
    np.testing.assert_array_equal(searchsorted(array, values, side), np.searchsorted(times, values, side))
    np.testing.assert_array_equal(
        searchsorted(array, values[:310].reshape(-1, 31), side), np.searchsorted(times, values[:310].reshape(-1, 31), side)
    )
    for value in values[:20]:
        found = searchsorted(array, value, side)
        assert isinstance(found, int) and found == np.searchsorted(times, value, side)


def test_searchsorted_reads_each_block_once():
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
    values = np.sort(np.random.default_rng(6).uniform(times[0], times[-1], 2000))
    np.testing.assert_array_equal(searchsorted(counted, values), np.searchsorted(times, values))
    data_chunks = [k for k in read if k.startswith("t/c/")]
    assert len(data_chunks) == len(set(data_chunks)) <= 50


def test_searchsorted_rejects_a_bad_side():
    array = _zarr(np.arange(10.0), (5,))
    with pytest.raises(ValueError, match="side"):
        searchsorted(array, 3.0, "middle")


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
        "statistics": [{"type": "minmax", "block_shape": [500, 1], "path": ".stats/data/minmax/500x1"}]
    }
    reopened = zarr.open_group(str(tmp_path / "a.zarr"), mode="r")
    stats = reopened["acquisition/es/.stats/data/minmax/500x1"]
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
    # The same block shape again replaces its entry; another block shape is a second level
    attribute = write_block_statistics(array, block_shape=[500, 1])
    assert [e["block_shape"] for e in attribute["statistics"]] == [[500, 1]]
    attribute = write_block_statistics(array, block_shape=[3000, 1])
    assert [e["path"] for e in attribute["statistics"]] == [".stats/data/minmax/500x1", ".stats/data/minmax/3000x1"]
    reopened = zarr.open_group(str(tmp_path / "a.zarr"), mode="r")
    assert sorted(reopened["acquisition/es/.stats/data/minmax"].array_keys()) == ["3000x1", "500x1"]
    # The smallest blocks by default, or the block shape asked for
    assert load_block_statistics(reopened["acquisition/es/data"])[0] == [500, 1]
    block_shape, coarse = load_block_statistics(reopened["acquisition/es/data"], block_shape=[3000, 1])
    np.testing.assert_array_equal(coarse, _expected_minmax(data, [3000, 1]))


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
        "signal": [{"block_shape": [500, 1]}, {"block_shape": [6000, None]}],
    })
    if output:
        out = str(tmp_path / ("out.shadowzarr" if output == "directory" else "refs.json"))
        write_rfs(rfs, out)
        rfs = load_rfs(out)
    group = open_rfs(rfs)
    _check_times(group, times)
    _, values = load_block_statistics(group["signal"])
    np.testing.assert_array_equal(values, _expected_minmax(group["signal"][...], [500, 1]))
    _, coarse = load_block_statistics(group["signal"], block_shape=[6000, 4])
    np.testing.assert_array_equal(coarse, _expected_minmax(group["signal"][...], [6000, 4]))
    assert sorted(group[".stats/signal/minmax"].array_keys()) == ["500x1", "6000x4"]
    stats = group[".stats/chunked/minmax/1000"]
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


# -- Levels --


@pytest.mark.parametrize(
    "fine, coarse",
    [([100, 1], [300, 1]), ([100, 1], [1050, 1]), ([50, 2], [200, 8]), ([25, 1], [100, 4]), ([100, 4], [100, 8])],
)
def test_rollup_equals_computing_from_the_data(fine, coarse):
    """1050 rows leave short last blocks at both levels; [.., 8] covers the whole second dimension."""
    data = np.random.default_rng(7).integers(-30000, 30000, (1050, 8)).astype("int16")
    array = _zarr(data, (100, 4))
    rolled = rollup_minmax(compute_minmax(array, fine), fine, coarse, data.shape)
    np.testing.assert_array_equal(rolled, _expected_minmax(data, coarse))


def test_rollup_with_nan():
    data = np.arange(80, dtype="float64").reshape(40, 2)
    data[0:10] = np.nan
    data[15, 0] = np.nan
    array = _zarr(data, (10, 2))
    rolled = rollup_minmax(compute_minmax(array, [5, 1]), [5, 1], [10, 1], data.shape)
    np.testing.assert_array_equal(rolled, _expected_minmax(data, [10, 1]))
    assert np.isnan(rolled[0]).all()


def test_rollup_needs_nested_blocks():
    with pytest.raises(ValueError, match="whole numbers"):
        rollup_minmax(np.zeros((10, 2)), [100], [150], [1000])


def test_levels_read_the_data_once():
    times = _sorted_times()
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w", zarr_format=3)
    array = root.create_array("t", shape=times.shape, dtype=times.dtype, chunks=(1000,))
    array[...] = times
    read = []

    class Counting(zarr.storage.WrapperStore):
        async def get(self, key, prototype, byte_range=None):
            read.append(key)
            return await super().get(key, prototype, byte_range)

    counted = zarr.open_array(Counting(store), path="t", mode="r")
    levels = compute_block_statistics_levels(counted, ["minmax"], [[10_000], [1000], [50_000], [3000]])
    assert [b for b, _ in levels] == [[10_000], [1000], [50_000], [3000]]
    for block, values in levels:
        np.testing.assert_array_equal(values["minmax"], _expected_minmax(times, block))
    data_chunks = [k for k in read if k.startswith("t/c/")]
    # 1000 is read from the data; 3000 and 10,000 are rolled up from it, and 50,000 from 10,000
    assert len(data_chunks) == len(set(data_chunks)) == 50


def test_write_levels(tmp_path):
    root = zarr.open_group(str(tmp_path / "a.zarr"), mode="w", zarr_format=3)
    data = np.random.default_rng(8).normal(size=(3000, 4))
    array = root.create_array("es/data", shape=data.shape, dtype=data.dtype, chunks=(500, 4))
    array[...] = data
    attribute = write_block_statistics(array, levels=[[100, 1], [500, 1], [None, 1]])
    assert [e["path"] for e in attribute["statistics"]] == [
        ".stats/data/minmax/100x1", ".stats/data/minmax/500x1", ".stats/data/minmax/3000x1"
    ]
    for block in ([100, 1], [500, 1], [3000, 1]):
        np.testing.assert_array_equal(load_block_statistics(array, block_shape=block)[1], _expected_minmax(data, block))
    with pytest.raises(ValueError, match="not both"):
        write_block_statistics(array, levels=[[100, 1]], block_shape=[100, 1])


@pytest.mark.parametrize("side", ["left", "right"])
def test_searchsorted_through_levels(side):
    times = _sorted_times()
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w", zarr_format=3)
    array = root.create_array("t", shape=times.shape, dtype=times.dtype, chunks=(1000,))
    array[...] = times
    # 4000 doesn't nest in 3000, so it is skipped; 1000 -> 3000 -> 9000 -> whole array
    write_block_statistics(array, levels=[[1000], [3000], [9000], [4000], [None]], sorted="ascending")
    rng = np.random.default_rng(9)
    values = np.concatenate([rng.uniform(times[0] - 1, times[-1] + 1, 500), times[[0, 999, 1000, 1009, 8999, 9000, -1]]])
    np.testing.assert_array_equal(searchsorted(array, values, side), np.searchsorted(times, values, side))

    read = []

    class Counting(zarr.storage.WrapperStore):
        async def get(self, key, prototype, byte_range=None):
            read.append(key)
            return await super().get(key, prototype, byte_range)

    counted = zarr.open_array(Counting(store), path="t", mode="r")
    assert searchsorted(counted, 123.0, side) == np.searchsorted(times, 123.0, side)
    assert len([k for k in read if k.startswith("t/c/")]) == 1


@pytest.mark.parametrize("levels", [None, [[100, 1]], [[100, 1], [500, 1], [2000, 1]], [[50, 4], [500, 4]]])
def test_minmax_range_matches_numpy(levels):
    data = np.random.default_rng(10).normal(size=(5050, 4))
    data[700:760, 2] = np.nan
    array = _zarr(data, (100, 4))
    if levels:
        write_block_statistics(array, levels=levels)
    rng = np.random.default_rng(11)
    ranges = [(0, 5050), (0, 1), (5049, 5050), (99, 101), (100, 2000), (700, 760)]
    ranges += [tuple(sorted(rng.integers(0, 5051, 2))) for _ in range(30)]
    whole = bool(levels) and levels[0][1] == 4
    for start, stop in ranges:
        if start == stop:
            continue
        low, high = minmax_range(array, start, stop)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            part = data[start:stop]
            want_low, want_high = np.nanmin(part, axis=0), np.nanmax(part, axis=0)
            if whole:
                want_low, want_high = np.nanmin(part, keepdims=True)[0], np.nanmax(part, keepdims=True)[0]
        np.testing.assert_array_equal(low, want_low, err_msg=f"{start}:{stop}")
        np.testing.assert_array_equal(high, want_high, err_msg=f"{start}:{stop}")


def test_minmax_range_reads_only_the_edges():
    data = np.random.default_rng(12).integers(-1000, 1000, (100_000, 3)).astype("int16")
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w", zarr_format=3)
    array = root.create_array("es", shape=data.shape, dtype=data.dtype, chunks=(1000, 1))
    array[...] = data
    write_block_statistics(array, levels=[[1000, 1], [10_000, 1]])
    read = []

    class Counting(zarr.storage.WrapperStore):
        async def get(self, key, prototype, byte_range=None):
            read.append(key)
            return await super().get(key, prototype, byte_range)

    counted = zarr.open_array(Counting(store), path="es", mode="r")
    low, high = minmax_range(counted, 1500, 98_700)
    np.testing.assert_array_equal(low, data[1500:98_700].min(axis=0))
    np.testing.assert_array_equal(high, data[1500:98_700].max(axis=0))
    # Only the partial first and last chunk of each channel are read from the data
    assert len([k for k in read if k.startswith("es/c/")]) == 2 * 3


def test_minmax_range_one_dimensional_and_errors():
    times = _sorted_times(10_000)
    array = _zarr(times, (1000,))
    write_block_statistics(array, levels=[[1000], [5000]])
    assert minmax_range(array, 10, 9_990) == (times[10], times[9_989])
    with pytest.raises(ValueError, match="start < stop"):
        minmax_range(array, 5, 5)
    wide = _zarr(np.zeros((100, 8)), (10, 4))
    write_block_statistics(wide, block_shape=[10, 4])
    with pytest.raises(ValueError, match="whole size"):
        minmax_range(wide, 0, 50)


def test_readers_skip_coarse_levels_when_a_finer_one_is_small():
    data = np.random.default_rng(13).integers(-1000, 1000, (100_000, 3)).astype("int16")
    times = np.sort(np.random.default_rng(14).uniform(0, 1000, 100_000))
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w", zarr_format=3)
    es = root.create_array("es", shape=data.shape, dtype=data.dtype, chunks=(1000, 1))
    es[...] = data
    ts = root.create_array("ts", shape=times.shape, dtype=times.dtype, chunks=(1000,))
    ts[...] = times
    write_block_statistics(es, levels=[[1000, 1], [10_000, 1]])
    write_block_statistics(ts, levels=[[1000], [10_000]], sorted="ascending")
    read = []

    class Counting(zarr.storage.WrapperStore):
        async def get(self, key, prototype, byte_range=None):
            read.append(key)
            return await super().get(key, prototype, byte_range)

    counted = zarr.open_group(Counting(store), mode="r")
    for max_bytes, coarse_read in [(2**20, False), (0, True)]:
        read.clear()
        low, _ = minmax_range(counted["es"], 1500, 98_700, max_bytes=max_bytes)
        np.testing.assert_array_equal(low, data[1500:98_700].min(axis=0))
        assert any("minmax/10000x1/" in k for k in read) == coarse_read
        read.clear()
        assert searchsorted(counted["ts"], 500.0, max_bytes=max_bytes) == np.searchsorted(times, 500.0)
        assert any("minmax/10000/" in k for k in read) == coarse_read


# -- missing_count and moments --


def _blocks(data, block_shape):
    grid = [-(-s // b) for s, b in zip(data.shape, block_shape)]
    for coords in np.ndindex(*grid):
        yield coords, data[tuple(slice(i * b, (i + 1) * b) for i, b in zip(coords, block_shape))]


def _expected_moments(data, block_shape, missing):
    grid = [-(-s // b) for s, b in zip(data.shape, block_shape)]
    out = np.empty(grid + [3])
    for coords, block in _blocks(data, block_shape):
        keep = np.ones(block.shape, bool)
        for m in missing:
            keep &= ~np.isnan(block) if isinstance(m, float) and np.isnan(m) else block != m
        values = block[keep].astype(np.float64)
        out[coords] = (len(values), values.mean() if len(values) else np.nan,
                       ((values - values.mean()) ** 2).sum() if len(values) else np.nan)
    return out


@pytest.mark.parametrize("block_shape", [[100, 4], [50, 1], [300, 8], [1050, 1], [1, 8]])
def test_missing_count_and_moments_of_floats(block_shape):
    rng = np.random.default_rng(15)
    data = (1e6 + rng.normal(size=(1050, 8))).astype("float64")  # a large offset tests precision
    data[rng.random(data.shape) < 0.05] = np.nan
    data[0:50, 0] = np.nan
    array = _zarr(data, (100, 4))
    count = compute_missing_count(array, block_shape)
    for coords, block in _blocks(data, block_shape):
        assert count[coords] == np.isnan(block).sum()
    moments = compute_moments(array, block_shape, read_bytes=2000)
    np.testing.assert_allclose(moments, _expected_moments(data, block_shape, [float("nan")]), rtol=1e-9, atol=1e-6)


def test_moments_of_int16_are_exact():
    data = np.random.default_rng(16).integers(-32768, 32768, (2000, 3)).astype("int16")
    data[::7, 1] = -32768
    array = _zarr(data, (500, 1))
    moments = compute_moments(array, [500, 1], missing=[-32768])
    assert moments.dtype == np.int64
    for coords, block in _blocks(data, [500, 1]):
        values = block[block != -32768].astype(np.int64)
        assert list(moments[coords]) == [len(values), values.sum(), (values * values).sum()]
    count = compute_missing_count(array, [500, 1], missing=[-32768])
    assert count[:, 1].sum() == (data[:, 1] == -32768).sum() and count[:, 0].sum() == (data[:, 0] == -32768).sum()


@pytest.mark.parametrize("fine, coarse", [([100, 1], [300, 1]), ([100, 1], [1050, 1]), ([50, 2], [200, 8])])
def test_rollups_of_missing_count_and_moments(fine, coarse):
    rng = np.random.default_rng(17)
    data = rng.normal(5.0, 2.0, (1050, 8))
    data[rng.random(data.shape) < 0.1] = np.nan
    array = _zarr(data, (100, 4))
    np.testing.assert_array_equal(
        rollup_missing_count(compute_missing_count(array, fine), fine, coarse, data.shape),
        compute_missing_count(array, coarse),
    )
    np.testing.assert_allclose(
        rollup_moments(compute_moments(array, fine), fine, coarse, data.shape), compute_moments(array, coarse),
        rtol=1e-10,
    )
    ints = rng.integers(-1000, 1000, (1050, 8)).astype("int16")
    iarray = _zarr(ints, (100, 4))
    np.testing.assert_array_equal(
        rollup_moments(compute_moments(iarray, fine), fine, coarse, ints.shape), compute_moments(iarray, coarse)
    )


@pytest.mark.parametrize("dtype", ["float32", "int16"])
@pytest.mark.parametrize("levels", [None, [[100, 1]], [[100, 1], [500, 1], [2000, 1]]])
def test_moments_range_matches_numpy(dtype, levels):
    rng = np.random.default_rng(18)
    data = (rng.normal(100, 30, (5050, 4))).astype(dtype)
    missing = None
    if dtype == "float32":
        data[700:760, 2] = np.nan
    else:
        data[::11, 3] = -32768
        missing = [-32768]
    array = _zarr(data, (100, 4))
    if levels:
        write_block_statistics(array, ["moments"], levels=levels, missing=missing)
    ranges = [(0, 5050), (0, 1), (99, 101), (100, 2000), (700, 760)]
    ranges += [tuple(sorted(rng.integers(0, 5051, 2))) for _ in range(20)]
    for start, stop in ranges:
        if start == stop:
            continue
        result = moments_range(array, start, stop, ddof=1, missing=missing)
        part = data[start:stop].astype(np.float64)
        if dtype == "int16":
            part[data[start:stop] == -32768] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            np.testing.assert_array_equal(result["count"], (~np.isnan(part)).sum(axis=0))
            np.testing.assert_allclose(result["mean"], np.nanmean(part, axis=0), rtol=1e-9, err_msg=f"{start}:{stop}")
            np.testing.assert_allclose(result["variance"], np.nanvar(part, axis=0, ddof=1), rtol=1e-7,
                                       err_msg=f"{start}:{stop}")


def test_moments_range_one_dimensional():
    times = _sorted_times(10_000)
    array = _zarr(times, (1000,))
    write_block_statistics(array, ["moments", "missing_count"], levels=[[1000], [5000]])
    result = moments_range(array, 10, 9_990)
    assert result["count"] == 9_980
    assert np.isclose(result["mean"], times[10:9_990].mean())
    assert np.isclose(result["variance"], times[10:9_990].var())
    entries = array.attrs["block_statistics"]["statistics"]
    assert {e["type"] for e in entries} == {"moments", "missing_count"}
    assert all(e["missing"] == ["NaN"] for e in entries)


def test_all_statistics_in_one_pass_over_the_data():
    data = np.random.default_rng(19).normal(size=(20_000, 2))
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w", zarr_format=3)
    array = root.create_array("a", shape=data.shape, dtype=data.dtype, chunks=(1000, 1))
    array[...] = data
    read = []

    class Counting(zarr.storage.WrapperStore):
        async def get(self, key, prototype, byte_range=None):
            read.append(key)
            return await super().get(key, prototype, byte_range)

    counted = zarr.open_array(Counting(store), path="a", mode="r")
    levels = compute_block_statistics_levels(counted, ["minmax", "missing_count", "moments"], [[1000, 1], [5000, 1]])
    chunks = [k for k in read if k.startswith("a/c/")]
    assert len(chunks) == len(set(chunks)) == 40
    np.testing.assert_array_equal(levels[1][1]["minmax"], _expected_minmax(data, [5000, 1]))


def test_moments_through_references(h5_times, tmp_path):
    path, times = h5_times
    rfs = add_block_statistics(generate_rfs(path), {
        "signal": [{"statistics": ["moments", "missing_count"], "block_shape": [500, 1], "missing": [-32768]},
                   {"statistics": ["moments", "missing_count"], "block_shape": [3000, 1], "missing": [-32768]}],
    })
    out = str(tmp_path / "refs.json")
    write_rfs(rfs, out)
    group = open_rfs(load_rfs(out))
    signal = group["signal"][...]
    result = moments_range(group["signal"], 100, 11_000)
    np.testing.assert_array_equal(result["count"], np.full(4, 10_900))
    np.testing.assert_allclose(result["mean"], signal[100:11_000].mean(axis=0))
    np.testing.assert_allclose(result["variance"], signal[100:11_000].var(axis=0))
    entries = group["signal"].attrs["block_statistics"]["statistics"]
    assert {(e["type"], tuple(e["block_shape"])) for e in entries} == {
        ("moments", (500, 1)), ("missing_count", (500, 1)), ("moments", (3000, 1)), ("missing_count", (3000, 1))
    }
    assert all(e["missing"] == [-32768] for e in entries)
