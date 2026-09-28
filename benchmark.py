"""Benchmark: remfile vs zindi for cold reads of remote NWB data.

Reads the first minute of acquisition/ElectricalSeries from a DANDI asset.
Single cold run for each approach — no caching, no warm-up.
"""

import time

import numpy as np

URL = "https://api.dandiarchive.org/api/assets/5a9cc6f1-aeaf-46cc-aae7-ea27960236ea/download/"


def _get_n_samples(es):
    rate = es.rate
    n_samples = int(rate * 60)
    n_channels = es.data.shape[1]
    print(f"  {n_samples:,} samples x {n_channels} channels @ {rate} Hz")
    return n_samples


def read_with_remfile():
    """Read via NWBHDF5IO + remfile."""
    import h5py
    import remfile
    from pynwb import NWBHDF5IO

    t0 = time.perf_counter()
    rf = remfile.File(URL)
    h5f = h5py.File(rf, "r")
    io = NWBHDF5IO(file=h5f, load_namespaces=True)
    nwbfile = io.read()
    t_open = time.perf_counter() - t0

    n_samples = _get_n_samples(nwbfile.acquisition["ElectricalSeries"])

    t1 = time.perf_counter()
    data = nwbfile.acquisition["ElectricalSeries"].data[:n_samples, :]
    t_read = time.perf_counter() - t1

    io.close()
    return data, t_open, t_read


def read_with_zindi(rfs):
    """Read via NWBZarrIO + zindi RfsStore (no cache)."""
    from hdmf_zarr import NWBZarrIO
    from zindi import RfsStore

    store = RfsStore(rfs)

    t0 = time.perf_counter()
    io = NWBZarrIO(path=store, mode="r")
    io.open()
    nwbfile = io.read()
    t_open = time.perf_counter() - t0

    n_samples = _get_n_samples(nwbfile.acquisition["ElectricalSeries"])

    t1 = time.perf_counter()
    data = nwbfile.acquisition["ElectricalSeries"].data[:n_samples, :]
    t_read = time.perf_counter() - t1

    io.close()
    return data, t_open, t_read


if __name__ == "__main__":
    import warnings

    from zindi import generate_rfs, write_rfs

    warnings.filterwarnings("ignore")

    # Pre-generate RFS (one-time cost, not part of the benchmark)
    rfs_path = "/tmp/bench_rfs.zindi.json"
    try:
        import json
        with open(rfs_path) as f:
            rfs = json.load(f)
        print(f"Loaded pre-generated RFS from {rfs_path} ({len(rfs['refs']):,} refs)")
    except FileNotFoundError:
        print("Generating RFS (one-time cost)...")
        rfs = generate_rfs(URL)
        write_rfs(rfs, rfs_path, format="json")
        print(f"Saved RFS to {rfs_path} ({len(rfs['refs']):,} refs)")
    print()

    # --- remfile ---
    print("=" * 60)
    print("NWBHDF5IO + remfile (cold)")
    print("=" * 60)
    try:
        rf_data, rf_open, rf_read = read_with_remfile()
        print(f"  Open:  {rf_open:.1f}s")
        print(f"  Read:  {rf_read:.1f}s")
        print(f"  Total: {rf_open + rf_read:.1f}s")
    except Exception as e:
        print(f"  ERROR: {e}")
        rf_data = None
    print()

    # --- zindi ---
    print("=" * 60)
    print("NWBZarrIO + zindi (cold, pre-generated RFS, no cache)")
    print("=" * 60)
    try:
        z_data, z_open, z_read = read_with_zindi(rfs)
        print(f"  Open:  {z_open:.1f}s")
        print(f"  Read:  {z_read:.1f}s")
        print(f"  Total: {z_open + z_read:.1f}s")
    except Exception as e:
        import traceback
        traceback.print_exc()
        z_data = None
    print()

    # --- validation ---
    if rf_data is not None and z_data is not None:
        print("=" * 60)
        print("Validation")
        print("=" * 60)
        print(f"  Data match: {np.array_equal(rf_data, z_data)}")
        print(f"  Shape: {rf_data.shape}")
