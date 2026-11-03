"""Benchmark finding a time window in a sorted timestamps array of a remote NWB file.

Compares three ways to find the indices of a window of time:

- whole: read the whole timestamps array, then np.searchsorted
- 1 s index: the temporaldata approach (torch_brain), an index of the position
  of every whole second, built once by reading the whole array, then one read
  of the slice it points to
- block statistics: shadowzarr.block_statistics.find_range, which reads the
  array's minmax statistics and one or two blocks

Bytes and requests are counted at the Zarr store, so they are what each
approach asks the store for; the 1 s index's own bytes are counted as if read
from the store too.

    python scripts/block_statistics_benchmark.py \\
        https://api.dandiarchive.org/api/assets/86db184c-6fad-4005-85cc-98d9a400d39c/download/ \\
        acquisition/probe_729445652_lfp/probe_729445652_lfp_data/timestamps
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import zarr

from shadowzarr import generate_rfs
from shadowzarr.block_statistics import add_block_statistics, find_range
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


def _windows(times: np.ndarray, n: int, length: float, seed: int) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    starts = rng.uniform(times[0], times[-1] - length, n)
    return [(float(s), float(s + length)) for s in starts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("dataset")
    parser.add_argument("--granularity", type=int, default=1, help="chunks per block (default 1)")
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--length", type=float, default=2.0, help="window length in seconds")
    args = parser.parse_args()

    t = time.perf_counter()
    rfs = generate_rfs(args.url)
    print(f"references: {time.perf_counter() - t:.1f} s")

    t = time.perf_counter()
    rfs = add_block_statistics(rfs, {args.dataset: {"granularity": args.granularity, "sorted": "ascending"}})
    stats_time = time.perf_counter() - t

    counting = CountingStore(RfsStore(rfs))
    root = zarr.open_group(counting, mode="r", zarr_format=3)
    array = root[args.dataset]
    entry = array.attrs["block_statistics"]["statistics"][0]
    stats_bytes = root[entry["path"]].nbytes
    print(f"array: {array.shape[0]:,} values, {array.nbytes / 1e6:.1f} MB, chunks of {array.chunks[0]:,}")
    print(f"statistics: blocks of {entry['block_shape'][0]:,}, {stats_bytes / 1e3:.1f} KB, "
          f"computed in {stats_time:.1f} s (reads the whole array once)")

    # Whole array
    counting.reset()
    t = time.perf_counter()
    times = np.asarray(array[...])
    whole = (time.perf_counter() - t, counting.requests, counting.bytes)
    windows = _windows(times, args.windows, args.length, seed=0)
    expected = [slice(int(np.searchsorted(times, a)), int(np.searchsorted(times, b, "right"))) for a, b in windows]

    # The 1 s index: position of every whole second, built from the whole array
    seconds = np.arange(np.floor(times[0]), np.ceil(times[-1]) + 1)
    index = np.searchsorted(times, seconds).astype("int64")

    rows = []
    for name in ["1 s index", "block statistics"]:
        counting.reset()
        t = time.perf_counter()
        for (a, b), want in zip(windows, expected):
            if name == "1 s index":
                lo = index[max(0, int(np.floor(a - seconds[0])))]
                hi = index[min(len(index) - 1, int(np.ceil(b - seconds[0])))]
                part = np.asarray(array[lo:hi])
                found = slice(int(lo + np.searchsorted(part, a)), int(lo + np.searchsorted(part, b, "right")))
            else:
                found = find_range(array, a, b, root=root)
            assert found == want, (name, a, b, found, want)
        elapsed = (time.perf_counter() - t) / len(windows)
        per_requests = counting.requests / len(windows)
        per_bytes = counting.bytes / len(windows)
        if name == "1 s index":
            per_bytes += index.nbytes  # the index has to be read too
        rows.append((name, elapsed, per_requests, per_bytes))

    print(f"\n{args.windows} windows of {args.length} s; per lookup:")
    print(f"{'approach':<18}{'time':>10}{'requests':>10}{'MB read':>12}")
    print(f"{'whole array':<18}{whole[0]:>9.2f}s{whole[1]:>10}{whole[2] / 1e6:>12.2f}")
    for name, elapsed, requests, read in rows:
        print(f"{name:<18}{elapsed:>9.2f}s{requests:>10.1f}{read / 1e6:>12.3f}")
    print(f"\n1 s index size: {index.nbytes / 1e3:.1f} KB ({len(index):,} seconds)")


if __name__ == "__main__":
    main()
