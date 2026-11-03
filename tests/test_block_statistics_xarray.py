"""Xarray DataArrays whose time selection and reductions use block statistics."""

import warnings
from datetime import datetime, timezone

import numpy as np
import pytest
import zarr

xr = pytest.importorskip("xarray")

from shadowzarr.block_statistics import write_block_statistics  # noqa: E402
from shadowzarr.block_statistics_xarray import BlockStatisticsIndex, StatsArray, open_timeseries  # noqa: E402


class Counting(zarr.storage.WrapperStore):
    """Records the keys read."""

    def __init__(self, store, read=None):
        super().__init__(store)
        self.read = read if read is not None else []

    def _with_store(self, store):
        return type(self)(store, self.read)

    async def get(self, key, prototype, byte_range=None):
        self.read.append(key)
        return await super().get(key, prototype, byte_range)


def _chunks(read, array):
    return [k for k in read if k.startswith(f"ts/{array}/c/")]


@pytest.fixture
def series():
    """A TimeSeries group with irregular timestamps and four channels, read through a counting store."""
    rng = np.random.default_rng(0)
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w", zarr_format=3)
    group = root.create_group("ts")
    data = rng.normal(10.0, 3.0, size=(20_000, 4)).astype("float32")
    data[5000:5100, 2] = np.nan
    times = np.cumsum(rng.exponential(0.001, 20_000))
    d = group.create_array("data", shape=data.shape, dtype=data.dtype, chunks=(1000, 1))
    d[...] = data
    t = group.create_array("timestamps", shape=times.shape, dtype=times.dtype, chunks=(1000,))
    t[...] = times
    write_block_statistics(d, ["minmax", "moments"], levels=[[1000, 1], [5000, 1]])
    write_block_statistics(t, levels=[[1000]], sorted="ascending")
    counting = Counting(store)
    return open_timeseries(zarr.open_group(counting, mode="r")["ts"]), data, times, counting.read


def test_selection_by_time_reads_only_statistics_and_edges(series):
    da, data, times, read = series
    read.clear()
    window = da.sel(time=slice(2.0, 15.0))
    start, stop = np.searchsorted(times, 2.0), np.searchsorted(times, 15.0, "right")
    assert window.sizes["time"] == stop - start
    assert isinstance(window.variable._data, StatsArray)
    assert len(_chunks(read, "timestamps")) <= 2
    assert not _chunks(read, "data")
    np.testing.assert_array_equal(window.time.values, times[start:stop])
    np.testing.assert_array_equal(window.values, data[start:stop])


@pytest.mark.parametrize("method", ["max", "min", "mean", "var", "std"])
@pytest.mark.parametrize("skipna", [None, False])
def test_reductions_over_time_use_statistics(series, method, skipna):
    da, data, times, read = series
    window = da.sel(time=slice(2.0, 15.0))
    start, stop = np.searchsorted(times, 2.0), np.searchsorted(times, 15.0, "right")
    read.clear()
    result = getattr(window, method)("time", skipna=skipna)
    part = data[start:stop].astype(np.float64) if method in ("mean", "var", "std") else data[start:stop]
    numpy = getattr(np, method if skipna is False else "nan" + method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        np.testing.assert_allclose(result.values, numpy(part, axis=0), rtol=1e-5)
    assert result.dtype == np.float32
    # Each channel reads at most its two partial edge chunks
    assert len(_chunks(read, "data")) <= 2 * 4
    if skipna is False:
        assert np.isnan(result.values[2]) and not np.isnan(result.values[[0, 1, 3]]).any()


def test_other_operations_read_the_data(series):
    da, data, times, _ = series
    window = da.isel(time=slice(100, 900))
    np.testing.assert_allclose(window.median("time").values, np.median(data[100:900], axis=0))
    np.testing.assert_allclose((window * 2).values, data[100:900] * 2)
    np.testing.assert_allclose(window.isel(dim_1=1).values, data[100:900, 1])
    np.testing.assert_allclose(window.isel(time=[0, 5, 7]).values, data[[100, 105, 107]])
    np.testing.assert_allclose(window.max("dim_1").values, data[100:900].max(axis=1))


def test_label_selection(series):
    da, data, times, _ = series
    np.testing.assert_array_equal(da.sel(time=times[123]).values, data[123])
    assert da.sel(time=times[123] + 1e-7, method="nearest").time.values == times[123]
    picked = da.sel(time=times[[5, 17, 900]] + 1e-8, method="nearest")
    np.testing.assert_array_equal(picked.time.values, times[[5, 17, 900]])
    with pytest.raises(KeyError):
        da.sel(time=times[123] + 1e-7)
    with pytest.raises(KeyError):
        da.sel(time=times[123] + 1.0, method="nearest", tolerance=1e-6)
    # Selection within a selection
    inner = da.sel(time=slice(2.0, 15.0)).sel(time=slice(5.0, 6.0))
    start, stop = np.searchsorted(times, 5.0), np.searchsorted(times, 6.0, "right")
    np.testing.assert_array_equal(inner.time.values, times[start:stop])
    assert isinstance(inner.xindexes["time"], BlockStatisticsIndex)


def test_regularly_sampled_series_and_integers():
    rng = np.random.default_rng(1)
    root = zarr.open_group(zarr.storage.MemoryStore(), mode="w", zarr_format=3)
    group = root.create_group("es")
    data = rng.integers(-2000, 2000, (30_000, 3)).astype("int16")
    d = group.create_array("data", shape=data.shape, dtype=data.dtype, chunks=(3000, 1))
    d[...] = data
    s = group.create_array("starting_time", shape=(), dtype="float64")
    s[...] = 10.0
    s.attrs["rate"] = 1000.0
    write_block_statistics(d, ["minmax", "moments"], levels=[[3000, 1], [15_000, 1]])
    da = open_timeseries(group, dims=("time", "channel"))
    assert "time" in da.xindexes
    window = da.sel(time=slice(12.0, 35.0))
    start, stop = 2000, 25_001
    assert window.sizes["time"] == stop - start
    np.testing.assert_array_equal(window.max("time").values, data[start:stop].max(axis=0))
    np.testing.assert_allclose(window.mean("time").values, data[start:stop].mean(axis=0))
    np.testing.assert_allclose(window.var("time").values, data[start:stop].astype(np.float64).var(axis=0))
    assert window.mean("time").dtype == np.float64


def test_nwb_file_through_references(tmp_path):
    pynwb = pytest.importorskip("pynwb")
    from shadowzarr import generate_rfs, open_rfs
    from shadowzarr.block_statistics import add_block_statistics

    rng = np.random.default_rng(2)
    times = np.cumsum(rng.exponential(0.01, 50_000))
    data = rng.normal(size=(50_000, 3)).astype("float32")
    path = str(tmp_path / "test.nwb")
    nwb = pynwb.NWBFile(session_description="d", identifier="i",
                        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc))
    nwb.add_acquisition(pynwb.TimeSeries(name="ts", data=data, unit="V", timestamps=times))
    with pynwb.NWBHDF5IO(path, "w") as io:
        io.write(nwb)
    rfs = generate_rfs(path, contiguous_chunk_bytes=40_000)
    start, stop = np.searchsorted(times, 100.0), np.searchsorted(times, 400.0, "right")
    want_mean = data[start:stop].astype(np.float64).mean(axis=0)
    # The data's chunks span all channels, so its default blocks do too: those can't answer a
    # per-channel reduction, which then reads the data; [n, 1] blocks can
    chunk = open_rfs(rfs)["acquisition/ts/data"].chunks[0]
    for block_shape in (None, [chunk, 1]):
        with_stats = add_block_statistics(rfs, {
            "acquisition/ts/timestamps": {"sorted": "ascending"},
            "acquisition/ts/data": {"statistics": ["minmax", "moments"], "block_shape": block_shape},
        })
        da = open_timeseries(open_rfs(with_stats)["acquisition/ts"], dims=("time", "channel"))
        assert da.attrs["unit"] == "V"
        window = da.sel(time=slice(100.0, 400.0))
        np.testing.assert_allclose(window.mean("time").values, want_mean, rtol=1e-5)
        np.testing.assert_array_equal(window.max("time").values, data[start:stop].max(axis=0))
        assert window.variable._data._has("minmax") == (block_shape is not None)
