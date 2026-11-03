"""The same Xarray code on a remote NWB recording, with and without block statistics.

Opens one TimeSeries twice over the same references:

- plain: a DataArray over the Zarr arrays as Xarray would normally hold
  them, with the timestamps loaded into a pandas index
- with block statistics: shadowzarr.block_statistics_xarray.open_timeseries,
  after adding minmax statistics to the timestamps and minmax and moments
  per channel to the data

then runs identical selections and reductions on both and checks they agree.

    python scripts/block_statistics_xarray_demo.py \\
        https://api.dandiarchive.org/api/assets/86db184c-6fad-4005-85cc-98d9a400d39c/download/ \\
        acquisition/probe_729445652_lfp/probe_729445652_lfp_data
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import xarray as xr
import zarr

from shadowzarr import generate_rfs, open_rfs
from shadowzarr.block_statistics import add_block_statistics
from shadowzarr.block_statistics_xarray import open_timeseries


def _timed(fn):
    t = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("group", help="path of the TimeSeries group")
    parser.add_argument("--windows", default="1200:1202,1000:1600,2000:3800", help="start:stop pairs in seconds")
    args = parser.parse_args()

    rfs = generate_rfs(args.url)
    group = open_rfs(rfs)[args.group]
    chunk = group["data"].chunks[0]
    t = time.perf_counter()
    with_stats = add_block_statistics(rfs, {
        f"{args.group}/timestamps": {"sorted": "ascending"},
        f"{args.group}/data": [
            {"statistics": ["minmax", "moments"], "block_shape": [f * chunk, 1]} for f in (1, 10, 100)
        ],
    })
    print(f"statistics computed in {time.perf_counter() - t:.0f} s (one read of the data and timestamps)\n")

    def plain() -> xr.DataArray:
        g = open_rfs(rfs)[args.group]
        data = xr.Variable(("time", "channel"), xr.core.indexing.LazilyIndexedArray(_Backend(g["data"])))
        return xr.DataArray(data, coords={"time": np.asarray(g["timestamps"][...])})

    def fast() -> xr.DataArray:
        return open_timeseries(open_rfs(with_stats)[args.group], dims=("time", "channel"))

    rows = []
    for label, opener in [("plain xarray", plain), ("block statistics", fast)]:
        da, opened = _timed(opener)
        rows.append((label, "open", opened))
        for window in args.windows.split(","):
            a, b = (float(x) for x in window.split(":"))
            for name in ("max", "mean", "std"):
                result, elapsed = _timed(lambda: getattr(da.sel(time=slice(a, b)), name)("time").values)
                rows.append((label, f"{name} over {a:g}-{b:g} s", elapsed, result))
    print(f"{'':<18}{'operation':<28}{'time':>8}")
    for row in rows:
        print(f"{row[0]:<18}{row[1]:<28}{row[2]:>7.2f}s")
    # Check both against float64 numpy on the window's data, to within a millionth of its spread:
    # means near zero differ relatively more, since plain Xarray sums float32 data in float32
    plain_results = [r[3] for r in rows if r[0] == "plain xarray" and len(r) == 4]
    fast_results = [r[3] for r in rows if r[0] == "block statistics" and len(r) == 4]
    g = open_rfs(rfs)[args.group]
    times = np.asarray(g["timestamps"][...])
    i = 0
    worst = {"plain xarray": 0.0, "block statistics": 0.0}
    for window in args.windows.split(","):
        a, b = (float(x) for x in window.split(":"))
        lo, hi = np.searchsorted(times, a), np.searchsorted(times, b, "right")
        data = np.asarray(g["data"][lo:hi]).astype(np.float64)
        spread = data.std(axis=0)
        for name in ("max", "mean", "std"):
            reference = getattr(np, name)(data, axis=0)
            for label, result in (("plain xarray", plain_results[i]), ("block statistics", fast_results[i])):
                worst[label] = max(worst[label], float(np.max(np.abs(result - reference) / spread)))
            i += 1
    for label, error in worst.items():
        print(f"{label}: largest difference from float64 numpy, relative to each channel's spread: {error:.1e}")
    assert worst["block statistics"] < 1e-6


class _Backend(xr.backends.BackendArray):
    """A Zarr array as an Xarray lazily indexed backend array."""

    def __init__(self, array: zarr.Array):
        self.array, self.shape, self.dtype = array, array.shape, array.dtype

    def __getitem__(self, key):
        return xr.core.indexing.explicit_indexing_adapter(
            key, self.shape, xr.core.indexing.IndexingSupport.BASIC, lambda k: np.asarray(self.array[k])
        )


if __name__ == "__main__":
    main()
