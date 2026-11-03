"""Benchmark per-channel minimum, maximum, mean and variance over windows of a remote recording.

The statistics are a minmax per channel at several levels: one block per
chunk along time, and coarser levels rolled up from it. For each window
length this compares reading the window's data with
shadowzarr.block_statistics.minmax_range, which uses whole blocks of the
coarsest level that fit, finer levels toward the edges, and the data only for
the partial blocks at the two ends. It also runs minmax_range with only the
finest level, to show what the coarser levels add.

Bytes and requests are counted at the Zarr store.

    python scripts/block_statistics_minmax_benchmark.py \\
        https://api.dandiarchive.org/api/assets/86db184c-6fad-4005-85cc-98d9a400d39c/download/ \\
        acquisition/probe_729445652_lfp/probe_729445652_lfp_data/data --rate 1250
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import zarr

from shadowzarr import generate_rfs
from shadowzarr.block_statistics import add_block_statistics, minmax_range, moments_range
from shadowzarr.rfs_store import RfsStore


class CountingStore(zarr.storage.WrapperStore):
    """Counts the gets and the bytes the store returns."""

    def __init__(self, store):
        super().__init__(store)
        self.reset()

    def reset(self):
        self.requests = 0
        self.bytes = 0

    async def get(self, key, prototype, byte_range=None):
        result = await super().get(key, prototype, byte_range)
        if not key.endswith("zarr.json"):
            self.requests += 1
            self.bytes += len(result) if result is not None else 0
        return result


def _measure(counting, fn):
    counting.reset()
    t = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t, counting.requests, counting.bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("dataset")
    parser.add_argument("--rate", type=float, required=True, help="samples per second")
    parser.add_argument("--factors", default="1,10,100", help="chunks per block of each level (default 1,10,100)")
    parser.add_argument("--seconds", default="2,60,1800", help="window lengths in seconds (default 2,60,1800)")
    parser.add_argument("--skip-whole", type=float, default=600.0,
                        help="don't read windows longer than this many seconds whole (default 600)")
    args = parser.parse_args()
    factors = [int(f) for f in args.factors.split(",")]

    t = time.perf_counter()
    rfs = generate_rfs(args.url)
    print(f"references: {time.perf_counter() - t:.1f} s")
    probe = zarr.open_group(RfsStore(rfs), mode="r", zarr_format=3)[args.dataset]
    chunk = probe.chunks[0]
    levels = [{"block_shape": [f * chunk] + [1] * (probe.ndim - 1), "statistics": ["minmax", "moments"]}
              for f in factors]

    t = time.perf_counter()
    rfs = add_block_statistics(rfs, {args.dataset: levels})
    stats_time = time.perf_counter() - t

    counting = CountingStore(RfsStore(rfs))
    root = zarr.open_group(counting, mode="r", zarr_format=3)
    array = root[args.dataset]
    parent = root[args.dataset.rpartition("/")[0]]
    entries = array.attrs["block_statistics"]["statistics"]
    stats_bytes = sum(parent[e["path"]].nbytes for e in entries)
    print(f"array: {array.shape} {array.dtype}, {array.nbytes / 1e9:.2f} GB, chunks of {array.chunks}")
    shapes = sorted({e["block_shape"][0] for e in entries})
    print(f"statistics: minmax and moments, levels of {', '.join(map(str, shapes))} samples per channel, "
          f"{stats_bytes / 1e6:.2f} MB, computed in {stats_time:.0f} s (one read of the whole array)")

    # The same array, seeing only the finest level
    finest = dict(array.attrs["block_statistics"])
    finest["statistics"] = [e for e in entries if e["block_shape"][0] == min(x["block_shape"][0] for x in entries)]
    one_level = _OneLevel(array, finest)

    rng = np.random.default_rng(0)
    print(f"\n{'window':>8}  {'approach':<24}{'time':>9}{'requests':>10}{'MB read':>10}")
    for seconds in [float(x) for x in args.seconds.split(",")]:
        length = int(seconds * args.rate)
        start = int(rng.integers(0, array.shape[0] - length))
        stop = start + length
        rows = []
        if seconds <= args.skip_whole:
            whole, *cost = _measure(counting, lambda: np.asarray(array[start:stop]))
            expected = (np.nanmin(whole, axis=0), np.nanmax(whole, axis=0))
            rows.append(("read the window", *cost))
        else:
            expected = None
        result, *cost = _measure(counting, lambda: minmax_range(array, start, stop))
        if expected is not None:
            np.testing.assert_array_equal(result[0], expected[0])
            np.testing.assert_array_equal(result[1], expected[1])
        rows.append((f"minmax_range, {len(shapes)} levels", *cost))
        result_m, *cost = _measure(counting, lambda: moments_range(array, start, stop))
        if expected is not None:
            np.testing.assert_allclose(result_m["mean"], whole.astype(np.float64).mean(axis=0), rtol=1e-6)
            np.testing.assert_allclose(result_m["variance"], whole.astype(np.float64).var(axis=0), rtol=1e-6)
        rows.append((f"moments_range, {len(shapes)} levels", *cost))
        single, *cost = _measure(counting, lambda: minmax_range(one_level, start, stop))
        np.testing.assert_array_equal(single[0], result[0])
        np.testing.assert_array_equal(single[1], result[1])
        rows.append(("minmax_range, 1 level", *cost))
        for name, elapsed, requests, read in rows:
            print(f"{seconds:>7.0f}s  {name:<24}{elapsed:>8.2f}s{requests:>10}{read / 1e6:>10.2f}")


class _OneLevel:
    """The array with its block_statistics attribute narrowed to one level."""

    def __init__(self, array, attribute):
        self._array = array
        self.attrs = {"block_statistics": attribute}

    def __getattr__(self, name):
        return getattr(self._array, name)

    def __getitem__(self, key):
        return self._array[key]


if __name__ == "__main__":
    main()
