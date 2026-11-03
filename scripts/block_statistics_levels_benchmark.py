"""Benchmark what levels of minmax statistics add, on a rechunked copy of a recording.

The source's chunks may be too large to split into small blocks (compressed
chunks can't be split by reference). This copies part of a remote recording
into a local sharded Zarr store with small chunks per channel, then times
minmax_range over windows with only the finest level and with every level.
A store wrapper adds a fixed delay per request and a transfer time per byte,
to stand in for object storage; requests run concurrently, as they would
remotely.

    python scripts/block_statistics_levels_benchmark.py \\
        https://api.dandiarchive.org/api/assets/86db184c-6fad-4005-85cc-98d9a400d39c/download/ \\
        acquisition/probe_729445652_lfp/probe_729445652_lfp_data/data \\
        --rate 1250 --hours 1 --copy /tmp/lfp_copy.zarr
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

import numpy as np
import zarr

from shadowzarr import generate_rfs
from shadowzarr.block_statistics import minmax_range, write_block_statistics
from shadowzarr.rfs_store import RfsStore


class SlowStore(zarr.storage.WrapperStore):
    """Adds latency per request and transfer time per byte, and counts both.

    Requests wait their latency concurrently, but share one link: each
    request's bytes are sent after the bytes already queued.
    """

    def __init__(self, store, latency: float, bandwidth: float, counts: dict | None = None):
        super().__init__(store)
        self.latency, self.bandwidth = latency, bandwidth
        # Shared with the copies zarr makes, such as a read-only one
        self.counts = counts if counts is not None else {"requests": 0, "bytes": 0, "link_free": 0.0}

    @property
    def requests(self):
        return self.counts["requests"]

    @property
    def bytes(self):
        return self.counts["bytes"]

    def reset(self):
        self.counts.update(requests=0, bytes=0, link_free=0.0)

    def _with_store(self, store):
        return type(self)(store, self.latency, self.bandwidth, self.counts)

    async def get(self, key, prototype, byte_range=None):
        result = await super().get(key, prototype, byte_range)
        size = len(result) if result is not None else 0
        if not key.endswith("zarr.json"):
            self.counts["requests"] += 1
            self.counts["bytes"] += size
        await asyncio.sleep(self.latency)
        loop = asyncio.get_running_loop()
        now = loop.time()
        done = max(now, self.counts["link_free"]) + size / self.bandwidth
        self.counts["link_free"] = done
        await asyncio.sleep(done - now)
        return result


class _Levels:
    """The array with its block_statistics attribute narrowed to some levels."""

    def __init__(self, array, entries):
        self._array = array
        attribute = dict(array.attrs["block_statistics"])
        attribute["statistics"] = entries
        self.attrs = {"block_statistics": attribute}

    def __getattr__(self, name):
        return getattr(self._array, name)

    def __getitem__(self, key):
        return self._array[key]


def make_copy(url: str, dataset: str, path: str, samples: int, chunk: int, shard: int) -> None:
    rfs = generate_rfs(url)
    source = zarr.open_group(RfsStore(rfs), mode="r", zarr_format=3)[dataset]
    channels = source.shape[1]
    root = zarr.open_group(path, mode="w", zarr_format=3)
    copy = root.create_array(
        "data", shape=(samples, channels), dtype=source.dtype,
        chunks=(chunk, 1), shards=(shard, 1), compressors=[zarr.codecs.ZstdCodec(level=3)],
    )
    step = 20 * source.chunks[0]
    t = time.perf_counter()
    for start in range(0, samples, step):
        stop = min(start + step, samples)
        copy[start:stop] = source[start:stop]
        print(f"  copied {stop / samples:.0%} in {time.perf_counter() - t:.0f} s", end="\r", flush=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("dataset")
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--copy", required=True, help="where to keep the local copy; reused if it exists")
    parser.add_argument("--chunk-seconds", type=float, default=0.1)
    parser.add_argument("--latency", type=float, default=0.02, help="seconds per request (default 0.02)")
    parser.add_argument("--bandwidth", type=float, default=50e6, help="bytes per second (default 50e6)")
    parser.add_argument("--seconds", default="10,60,600,1800,3000", help="window lengths in seconds")
    args = parser.parse_args()

    chunk = int(round(args.chunk_seconds * args.rate))
    shard = chunk * 1000
    samples = int(args.hours * 3600 * args.rate) // shard * shard
    if not os.path.exists(args.copy):
        print(f"copying {samples:,} samples into chunks of {chunk} per channel")
        make_copy(args.url, args.dataset, args.copy, samples, chunk, shard)
    array = zarr.open_group(args.copy, mode="r+", zarr_format=3)["data"]

    levels = [[chunk * 10**k, 1] for k in range(5)]
    t = time.perf_counter()
    attribute = array.attrs.get("block_statistics")
    if not attribute or len(attribute["statistics"]) != len(levels):
        attribute = write_block_statistics(array, levels=levels)
    print(f"array: {array.shape} {array.dtype}, chunks of {array.chunks}")
    blocks = ", ".join(f"{e['block_shape'][0] / args.rate:g} s" for e in attribute["statistics"])
    print(f"statistics: {blocks} per channel, ready in {time.perf_counter() - t:.0f} s")

    slow = SlowStore(zarr.storage.LocalStore(args.copy), args.latency, args.bandwidth)
    remote = zarr.open_group(slow, mode="r", zarr_format=3)["data"]
    entries = sorted(remote.attrs["block_statistics"]["statistics"], key=lambda e: e["block_shape"][0])
    parent = zarr.open_group(args.copy, mode="r", zarr_format=3)
    sizes = {e["path"]: parent[e["path"]].nbytes for e in entries}
    print("level sizes: " + ", ".join(f"{e['block_shape'][0] / args.rate:g} s = {sizes[e['path']] / 1e6:.2f} MB"
                                      for e in entries))
    variants = [("finest level only", _Levels(remote, entries[:1])), (f"all {len(entries)} levels", remote)]

    print(f"\nsimulated storage: {args.latency * 1e3:.0f} ms per request, {args.bandwidth / 1e6:.0f} MB/s")
    print(f"{'window':>8}  {'approach':<20}{'time':>9}{'requests':>10}{'MB read':>10}")
    rng = np.random.default_rng(0)
    for seconds in [float(x) for x in args.seconds.split(",")]:
        length = int(seconds * args.rate)
        start = int(rng.integers(1, array.shape[0] - length - 1))
        stop = start + length
        expected = None
        for name, target in variants:
            slow.reset()
            t = time.perf_counter()
            low, high = minmax_range(target, start, stop)
            elapsed = time.perf_counter() - t
            if expected is None:
                expected = (low, high)
            np.testing.assert_array_equal(low, expected[0])
            np.testing.assert_array_equal(high, expected[1])
            print(f"{seconds:>7.0f}s  {name:<20}{elapsed:>8.2f}s{slow.requests:>10}{slow.bytes / 1e6:>10.2f}")
        # Check against the data for a few channels; all of them is millions of small chunks
        data = np.asarray(array[start:stop, :4])
        np.testing.assert_array_equal(expected[0][:4], data.min(axis=0))
        np.testing.assert_array_equal(expected[1][:4], data.max(axis=0))


if __name__ == "__main__":
    main()
