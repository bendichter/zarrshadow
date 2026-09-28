# Zindi

Represent remote HDF5 NWB files as Zarr v3 via JSON reference file systems.

## What it does

Zindi reads the metadata and chunk layout of an HDF5 file (local or remote) and produces a small JSON file that describes the same data as a Zarr v3 store. The JSON contains:

- **Zarr v3 metadata** (`zarr.json` entries for every group and array)
- **Chunk references** pointing to byte ranges in the original HDF5 file (`[url, offset, size]`)
- **Inline data** for small datasets (base64-encoded)

When you open this JSON, Zindi provides a zarr v3 `Store` that fetches chunks on demand from the remote HDF5 file using HTTP Range requests. No data is copied — the original file is the source of truth.

## How it relates to Lindi

[Lindi](https://github.com/NeurodataWithoutBorders/lindi) does something similar but targets Zarr v2 and creates an h5py-like shim object for use with `pynwb.NWBHDF5IO`.

Zindi instead produces a proper Zarr v3 store, following the [unified convention](https://github.com/NeurodataWithoutBorders/lindi/issues/125) that aligns Lindi and hdmf-zarr. hdmf-zarr adopted this convention in its Zarr v3 release (0.14.0), so a Zindi store can be read with `hdmf_zarr.NWBZarrIO` directly, without an h5py shim layer.

## Installation

```bash
pip install -e .
```

## Quick start

### Generate a reference file system from a remote NWB file

```python
from zindi import generate_rfs, write_rfs

url = "https://api.dandiarchive.org/api/assets/6e7e9b91-0d66-45af-b646-dfb11e4d9967/download/"

rfs = generate_rfs(url)
write_rfs(rfs, "example.zindi.json")
```

Just pass the URL — Zindi handles remote file access internally.

### Load the JSON and read data as Zarr v3

```python
from zindi import open_rfs

root = open_rfs("example.zindi.json")

# Browse the hierarchy
print(root.attrs["neurodata_type"])  # 'NWBFile'

# Read data (fetched from remote HDF5 via byte-range requests)
spike_times = root["units/spike_times"][:]
print(spike_times.shape)  # (359781,)
```

### Generate from a local HDF5 file

If you have a local copy but want chunk references to point to a remote URL:

```python
from zindi import generate_rfs, write_rfs

rfs = generate_rfs(
    "https://example.com/data.nwb",
    local_hdf5_path="/path/to/local/data.nwb",
)
write_rfs(rfs, "data.zindi.json")
```

### Read as an NWB file with hdmf-zarr

With hdmf-zarr 0.14.0 or later, pass the store to `NWBZarrIO`:

```python
from hdmf_zarr import NWBZarrIO
from zindi import RfsStore, load_rfs

with NWBZarrIO(RfsStore(load_rfs("example.zindi.json")), mode="r") as io:
    nwbfile = io.read()
    print(nwbfile.acquisition)
```

## Files with many chunks

A long electrophysiology recording can have millions of chunks. Listing each one in the JSON makes a file of hundreds of megabytes that has to be downloaded and parsed before anything can be read. For this reason, `generate_rfs` gives any array with more than `chunk_index_threshold` chunks (default 1000) a chunk index in place of individual refs. The index is a `uint64` array shaped like the chunk grid plus a last axis of length 2, holding `(offset, nbytes)` for each chunk, the same layout as the Zarr v3 sharding index. Chunks that were never written hold `2**64 - 1`.

Write to a path that does not end in `.json` to get a directory:

```python
rfs = generate_rfs(url)
write_rfs(rfs, "example.zindi")
root = open_rfs("example.zindi")  # also accepts a URL to the directory
```

```
example.zindi/
├── refs.json                       # metadata, small arrays, and chunk refs for arrays under the threshold
└── index/                          # one zarr v3 array per indexed array
    └── acquisition/ElectricalSeries/data/
        ├── zarr.json
        └── c/...
```

Opening the directory reads only `refs.json`. The index arrays are split into blocks of about 65,536 chunks and compressed with Blosc (zstd with byte shuffle), and a block is read the first time a chunk it covers is requested. They are ordinary Zarr arrays, so any Zarr library, including zarrita.js, can read them. For a synthetic NWB file with 2 million chunks, the directory is 0.8 MB against 176 MB for the single JSON, and `NWBZarrIO.read()` takes 0.04 s against 1.5 s.

Writing to a path ending in `.json` still produces a single file, with every chunk listed in `refs`. Pass `chunk_index_threshold=None` to `generate_rfs` to list every chunk in memory as well.

## DANDI support

Zindi handles DANDI API URLs automatically. The DANDI URL (which returns a 302 redirect to a presigned S3 URL) is resolved transparently, with the presigned URL cached for 10 minutes.

For embargoed datasets, set the appropriate environment variable:

```bash
export DANDI_API_KEY=your_token_here
# or for sandbox:
export DANDI_SANDBOX_API_KEY=your_token_here
```

## Unified convention

The generated JSON follows the [unified Zarr v3 convention](https://github.com/hdmf-dev/hdmf-zarr/issues/335) for representing HDF5/NWB concepts in Zarr:

| Feature | Convention |
|---------|-----------|
| Groups | `zarr.json` with `node_type: "group"` |
| Arrays | `zarr.json` with `node_type: "array"`, codecs pipeline |
| Scalars | Zero-dimensional array: `shape: []`, `chunk_shape: []`, single chunk keyed `c` |
| Dataset dtype | `_DTYPE` attribute on non-compound arrays: the numpy type name (e.g. `"float64"`), `"str"` for strings, `"object_reference"` for references |
| Soft links | `_LINKS` list on parent group: `[{"name", "source", "path"}]` |
| References in datasets | `_DTYPE: "object_reference"` with target paths as strings; compound reference fields listed in `_REFERENCE_FIELDS` |
| Spec location | Root `.specloc` attribute holds the plain path `"specifications"` |
| References in attrs | `{"_REFERENCE": {"source": ".", "path": "/target"}}` |
| NaN/Inf in attrs | Written as the float tokens `NaN`, `Infinity`, `-Infinity`, as zarr-python does |
| Strings | `data_type: "string"` with `vlen-utf8` codec |

## Local chunk caching

By default, every array slice triggers an HTTP Range request. For repeated access to the same data (common in interactive analysis), you can enable a persistent local cache backed by SQLite:

```python
from zindi import LocalCache, open_rfs

cache = LocalCache()  # persists to ~/.zindi/cache
root = open_rfs("example.zindi.json", local_cache=cache)

# First read fetches from remote; subsequent reads are served from disk
data = root["units/spike_times"][:]
```

To prevent unbounded cache growth, set a size limit. Least-recently-accessed chunks are evicted when the limit is exceeded:

```python
cache = LocalCache(max_size_bytes=500_000_000)  # 500 MB cap
```

## Request merging

When reading a multi-chunk slice, zindi automatically merges nearby HTTP Range requests into fewer, larger fetches. For example, reading 10 contiguous chunks from a remote file may result in a single HTTP request instead of 10.

Two parameters control the merging behavior:

- `merge_gap`: maximum gap in bytes between two ranges to merge (default 256 KB)
- `max_merge_size`: maximum size of a single merged request (default 50 MB)

```python
root = open_rfs("example.zindi.json", merge_gap=1_000_000, max_merge_size=100_000_000)
```

Set `merge_gap=0` to disable merging and fetch every chunk individually.

## Architecture

```
zindi/
├── generate_rfs.py          # HDF5 → Zarr v3 reference file system
├── open_rfs.py              # Open RFS as zarr.Group
├── rfs_store.py             # Zarr v3 Store backed by reference file system
├── chunk_index.py           # Byte-range indexes for arrays with many chunks
├── http_store.py            # Read-only zarr Store over HTTP, for remote index arrays
├── remfile.py               # File-like HTTP reader optimized for h5py
├── h5_filters_to_codecs.py  # HDF5 filters → Zarr v3 codec pipeline
├── h5_chunk_utils.py        # HDF5 chunk byte range utilities
├── attr_conversion.py       # HDF5 attrs → JSON-serializable values
└── url_resolver.py          # DANDI URL resolution with caching
```

**Data flow:**

```
Remote HDF5 file
    ↓ (h5py + Remfile: read metadata and chunk layout)
Reference file system (.zindi.json, or .zindi/ directory with chunk indexes)
    ↓ (RfsStore: zarr v3 Store implementation)
zarr.Group (read-only, chunks fetched on demand)
    ↓ (hdmf_zarr.NWBZarrIO)
pynwb NWBFile
```

## Current limitations

This is v0.1 — the following are not yet implemented:

- Nested compound dtypes (structs within structs)
- Cross-file object references (same-file references are supported)
- External array links (`_EXTERNAL_ARRAY_LINK`)
