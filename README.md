# zarrshadow

Represent remote HDF5 NWB files as Zarr v3 via JSON reference file systems.

## What it does

zarrshadow reads the metadata and chunk layout of an HDF5 file (local or remote) and produces a small JSON file that describes the same data as a Zarr v3 store. The JSON contains:

- **Zarr v3 metadata** (`zarr.json` entries for every group and array)
- **Chunk references** pointing to byte ranges in the original HDF5 file (`[url, offset, size]`)
- **Inline data** for small datasets (base64-encoded)

When you open this JSON, zarrshadow provides a zarr v3 `Store` that fetches chunks on demand from the remote HDF5 file using HTTP Range requests. No data is copied — the original file is the source of truth.

![A zarrshadow reference file system copies Zarr metadata and small datasets out of the HDF5 file and stores each chunk as a pointer into it](docs/images/store-contents.svg)

The metadata and small datasets are copied into the JSON when it is generated. Each chunk is a `[url, offset, size]` pointer into the original file and is fetched with an HTTP range request when it is read. The strip on the right enlarges the first 0.8 MB of the 103 GB file, where `c/0/0` begins right after 10 KB of HDF5 headers. The rest of the file holds more chunks, with more headers and heaps spread through it.

## How the Packages Fit Together

zarrshadow sits between the libraries that understand source formats and the libraries that read Zarr.

![zarrshadow reads the headers of source files and writes kerchunk references, which zarr-python reads through zarrshadow's store while fetching byte ranges from the source files; materialize turns the references into an ordinary Zarr store; in MATLAB, matzarr indexes MAT files for zarr-matlab](docs/images/ecosystem.svg)

On the writing side, zarrshadow reads only headers, through h5py, tifffile, and NEO, and writes references in the [kerchunk](https://fsspec.github.io/kerchunk/spec.html) JSON format with Zarr v3 keys. For an NWB file, hdmf-zarr and PyNWB write the file's structure, and NeuroConv can supply the file to write, which is experimental. On the reading side, `RfsStore` is a zarr-python store, so anything built on zarr-python reads the references, including hdmf-zarr's `NWBZarrIO`. A version 1 file is plain kerchunk, which fsspec reads too. A version 2 file, which uses `gen`, `indexes`, or `selections`, needs zarrshadow's store.

MATLAB has a parallel path for MAT files. [matzarr](https://github.com/catalystneuro/matzarr) indexes a `.mat` file into Zarr v3 metadata and a `manifest.json`, which the `ManifestStore` of [zarr-matlab](https://github.com/catalystneuro/zarr-matlab) reads, and a script in matzarr translates that index into kerchunk references. zarr-matlab does not read kerchunk references, so the references zarrshadow writes cannot be read from MATLAB as they are.

`materialize` removes the difference. Its output is an ordinary Zarr v3 store with no references, which zarr-python, zarr-matlab, and zarrita.js all read.

## How it relates to Lindi

[Lindi](https://github.com/NeurodataWithoutBorders/lindi) does something similar but targets Zarr v2 and creates an h5py-like shim object for use with `pynwb.NWBHDF5IO`.

zarrshadow instead produces a proper Zarr v3 store, following the [unified convention](https://github.com/NeurodataWithoutBorders/lindi/issues/125) that aligns Lindi and hdmf-zarr. hdmf-zarr adopted this convention in its Zarr v3 release (0.14.0), so a zarrshadow store can be read with `hdmf_zarr.NWBZarrIO` directly, without an h5py shim layer.

## Installation

```bash
pip install -e .
```

## Quick start

### Generate a reference file system from a remote NWB file

```python
from zarrshadow import generate_rfs, write_rfs

url = "https://api.dandiarchive.org/api/assets/6e7e9b91-0d66-45af-b646-dfb11e4d9967/download/"

rfs = generate_rfs(url)
write_rfs(rfs, "example.zarrshadow.json")
```

Just pass the URL — zarrshadow handles remote file access internally.

### Load the JSON and read data as Zarr v3

```python
from zarrshadow import open_rfs

root = open_rfs("example.zarrshadow.json")

# Browse the hierarchy
print(root.attrs["neurodata_type"])  # 'NWBFile'

# Read data (fetched from remote HDF5 via byte-range requests)
spike_times = root["units/spike_times"][:]
print(spike_times.shape)  # (359781,)
```

### Generate from a local HDF5 file

If you have a local copy but want chunk references to point to a remote URL:

```python
from zarrshadow import generate_rfs, write_rfs

rfs = generate_rfs(
    "https://example.com/data.nwb",
    local_hdf5_path="/path/to/local/data.nwb",
)
write_rfs(rfs, "data.zarrshadow.json")
```

### Read as an NWB file with hdmf-zarr

With hdmf-zarr 0.14.0 or later, pass the store to `NWBZarrIO`:

```python
from hdmf_zarr import NWBZarrIO
from zarrshadow import RfsStore, load_rfs

with NWBZarrIO(RfsStore(load_rfs("example.zarrshadow.json")), mode="r") as io:
    nwbfile = io.read()
    print(nwbfile.acquisition)
```

## Files with many chunks

A long electrophysiology recording can have millions of chunks. Listing each one in the JSON makes a file of hundreds of megabytes that has to be downloaded and parsed before anything can be read. For this reason, `generate_rfs` gives any array with more than `chunk_index_threshold` chunks (default 1000) a chunk index in place of individual refs. The index is a `uint64` array shaped like the chunk grid plus a last axis of length 2, holding `(offset, nbytes)` for each chunk, the same layout as the Zarr v3 sharding index. Chunks that were never written hold `2**64 - 1`.

Write to a path that does not end in `.json` to get a directory:

```python
rfs = generate_rfs(url)
write_rfs(rfs, "example.zarrshadow")
root = open_rfs("example.zarrshadow")  # also accepts a URL to the directory
```

![The same file as a single JSON and as a directory, with the chunk refs of large arrays moved into index arrays](docs/images/json-vs-directory.svg)

This is the DANDI file from the example below in both forms. The 327,680 chunk refs of the `ElectricalSeries` are two thirds of the single JSON, and the whole 48.4 MB is read when it is opened. In the directory, the four arrays with more than 1,000 chunks become index arrays, and opening reads only `refs.json`.

Opening the directory reads only `refs.json`. Each index is itself an ordinary Zarr array, chunked so that one index chunk holds the entries for about 65,536 data chunks and compressed with Blosc (zstd with byte shuffle). An index chunk is read the first time any data chunk it covers is requested, and recent index chunks are kept in memory. Because they are plain Zarr arrays, any Zarr library, including zarrita.js, can read them. For a synthetic NWB file with 2 million chunks, the directory is 0.8 MB against 176 MB for the single JSON, and `NWBZarrIO.read()` takes 0.04 s against 1.5 s.

### The refs.json Format

`refs.json` is a [kerchunk reference file](https://fsspec.github.io/kerchunk/spec.html) with Zarr v3 keys and a few additions. For the file in the example below it looks like this, shortened:

```json
{
  "version": 2,
  "templates": {"u0": "https://api.dandiarchive.org/api/assets/5a9cc6f1-aeaf-46cc-aae7-ea27960236ea/download/"},
  "refs": {
    "zarr.json": "{\"zarr_format\":3,\"node_type\":\"group\",...}",
    "acquisition/ElectricalSeries/data/zarr.json": "{\"shape\":[495184000,160],...}",
    "acquisition/Video: Rat08-20130708-02-run/timestamps/c/0": ["{{u0}}", 73293974410, 3559],
    "session_description/c": "base64:AQAAAK4DAABUaGUgY29uc29saWRhdGlvbi..."
  },
  "indexes": {
    "acquisition/ElectricalSeries/data": {
      "url": "https://api.dandiarchive.org/api/assets/5a9cc6f1-aeaf-46cc-aae7-ea27960236ea/download/",
      "index": "index/acquisition/ElectricalSeries/data"
    }
  },
  "sources": {
    "https://api.dandiarchive.org/api/assets/5a9cc6f1-aeaf-46cc-aae7-ea27960236ea/download/": {
      "size": 102986180753,
      "etag": "\"b885aa8ec05afd3337ae440e35249431-1535\""
    }
  }
}
```

`refs` holds the Zarr metadata, small datasets, and a `[url, offset, size]` entry for each chunk of an array with at most 1,000 chunks. For each larger array, `indexes` gives the file its chunks are in and the path of its index array, relative to `refs.json`. There is nothing in between: a reader looks up the array in `indexes`, opens that Zarr array, and reads the index chunk it needs. `gen`, `selections`, and `sources` are described below.

A directory that uses `indexes`, `gen`, or `selections` is marked `"version": 2`, so readers that only know version 1 of the kerchunk format refuse it instead of returning fill values for the chunks they cannot find. Writing to a path ending in `.json` produces a version 1 file with every chunk listed in `refs`, which any kerchunk reader can open. A file with `selections` stays version 2, because version 1 cannot express them.

### Contiguous Datasets

HDF5 stores a dataset that was written without chunking as one contiguous block. As a single Zarr chunk, reading any part of it would fetch all of it. `generate_rfs` presents a contiguous dataset larger than `contiguous_chunk_bytes` (default 4 MiB) as slabs along its first axis, described by one kerchunk `gen` entry such as this one:

```json
{"key": "acquisition/timestamps/c/{{i}}", "url": "{{u0}}", "offset": "{{2048 + i * 4194304}}", "length": "4194304", "dimensions": {"i": {"stop": 58}}}
```

zarrshadow computes a slab's offset when that chunk is requested. The slab height divides the first axis when a divisor is close to the target, so every slab has the same length. When none does, the last slab is read at full length, and zarr discards the part past the end of the array.

### Arrays Stored with Other Bytes

Some files store an array together with bytes that do not belong to it. A SpikeGLX recording holds 384 neural channels and one sync channel, interleaved sample by sample, so no byte range contains the neural channels alone. Other formats put a header in front of every sample. For these, `selections` says which bytes of the file belong to the array:

```json
"selections": {"imec0.ap": {"record_size": 770, "keep": [[0, 768]]}}
```

Every reference of the array is read as consecutive records of `record_size` bytes, here one sample of all 385 int16 channels. From each record the byte ranges in `keep` are taken and joined in the order listed, and the rest is dropped. The Zarr metadata describes only the selected data, a 384-column array with the plain `bytes` codec, so nothing in the codec chain is specific to zarrshadow. Listing several ranges keeps columns that are not next to each other, and listing them in another order reorders the columns. A selection applies to uncompressed data.

```python
from zarrshadow import RfsBuilder
from zarrshadow.builder import columns_selection

record_size, keep = columns_selection(n_columns=385, itemsize=2, columns=slice(0, 384))
builder = RfsBuilder()
builder.add_group("")
builder.add_array("imec0.ap", shape=[n_samples, 384], data_type="int16", chunk_shape=[30_000, 384])
builder.add_selection("imec0.ap", record_size, keep)
builder.add_contiguous_chunks(
    "imec0.ap", url=url, start=0, shape=[n_samples, 384], chunk_shape=[30_000, 384], itemsize=2, row_bytes=record_size
)
```

A request for part of a chunk reads only the records that hold it, with or without a selection, so reading a few samples from a large chunk does not fetch the whole chunk.

### Detecting Changed Files

A reference is a URL and a byte range, so it would return wrong data without any error if the file it points into were replaced. `generate_rfs` records each file's size and, for remote files, its ETag under `sources`. For DANDI assets these come from the asset metadata, whose `dandi:dandi-etag` is the ETag S3 reports. When reading, zarrshadow sends `If-Match` with every range request so that the server refuses it if the file has changed, compares the total size the server reports, and checks the size of local files. Any mismatch raises `SourceChangedError`. Pass `validate_sources=False` to `open_rfs` to turn the checks off.

### MATLAB Files

MATLAB `.mat` files saved with `-v7.3` are HDF5 files with a 512-byte userblock in front, and zarrshadow reads them like any other HDF5 file. HDF5 1.14 and later report chunk offsets from the start of the file, but HDF5 1.10 reports them from the end of the userblock, so `generate_rfs` checks one stored block against the file and corrects the offsets if needed. zarrshadow presents the data as HDF5 stores it: arrays are transposed relative to MATLAB, `char` arrays are UTF-16 codes, and cell arrays are references into `#refs#`. [matzarr](https://github.com/catalystneuro/matzarr) reads the same files from MATLAB with MATLAB semantics.

### Example

This file from DANDI has an `ElectricalSeries` of 495,184,000 samples by 160 channels (int16 at 20 kHz, about 6.9 hours). The HDF5 file stores it in chunks of 241,790 samples by 1 channel, about 12 seconds of one channel each, so its chunk grid is 2,048 by 160, or 327,680 data chunks.

```python
rfs = generate_rfs("https://api.dandiarchive.org/api/assets/5a9cc6f1-aeaf-46cc-aae7-ea27960236ea/download/")
write_rfs(rfs, "example.zarrshadow")
```

In the single JSON, those 327,680 data chunks are 327,680 entries in `refs`. In the directory, they are one index array at `index/acquisition/ElectricalSeries/data` with shape (2,048, 160, 2), chunked as (409, 160, 2), so it has 6 index chunks of about 320 KB each after compression. Each index chunk covers 409 rows of the data chunk grid for all 160 channels, which is the first 82 minutes of the recording for index chunk 0, the next 82 minutes for index chunk 1, and so on.

To read one second starting one hour in on channel 17:

```python
root = open_rfs("example.zarrshadow")
data = root["acquisition/ElectricalSeries/data"][72_000_000:72_020_000, 17]
```

![Reading data chunk c/297/17: row 297 falls in index chunk 0, entry [297, 17] gives the byte range, and one range request fetches it](docs/images/index-lookup.svg)

The circled numbers match the steps below. The drawing is not to scale: each of the 2,048 rows of the chunk grid would be a fraction of a pixel, and the last index chunk, which covers only 3 rows, is drawn larger than it is.

1. Samples 72,000,000 to 72,019,999 fall in row 297 of the data chunk grid (297 × 241,790 = 71,811,630), so zarr asks the store for data chunk `acquisition/ElectricalSeries/data/c/297/17`.
2. The chunk is not in `refs`, but the array is in `indexes`, which points to the index array at `index/acquisition/ElectricalSeries/data`. `(297, 17)` is in index chunk `(0, 0, 0)`, since 297 // 409 = 0, so the store reads `index/acquisition/ElectricalSeries/data/c/0/0/0`. This is the only index read, and later reads in the first 82 minutes reuse it from memory.
3. Entry `[297, 17]` of that index chunk gives the byte offset and size of the data chunk in the HDF5 file, and the store fetches those bytes with one HTTP range request.

Reading the same second on all 160 channels needs data chunks `c/297/0` through `c/297/159`, which all sit in the same index chunk, so it still reads one index chunk and then fetches 160 data chunks.

Pass `chunk_index_threshold=None` to `generate_rfs` to list every chunk in `refs` in memory as well.

## TIFF Files

`generate_rfs_tiff` builds references for a TIFF file with [tifffile](https://github.com/cgohlke/tifffile), which handles strips and tiles, multi-page series, and OME and other pyramids. Install it with `pip install zarrshadow[tiff]`.

```python
from zarrshadow import generate_rfs_tiff, open_rfs

root = open_rfs(generate_rfs_tiff("stack.ome.tif"))
frame = root["0"][100]      # series 0; a pyramid has its levels at "0/0", "0/1", ...
```

Series `i` is stored at path `"i"`, the layout bioformats2raw uses for OME-Zarr. The pages of an uncompressed stack are usually evenly spaced in the file, and then the whole series is one `gen` entry. TIFF deflate and zstd chunks are ordinary zlib and zstd streams, so they get the standard `numcodecs.zlib` and `zstd` codecs, which any Zarr library can decode. Other compressions, such as LZW, JPEG, and the horizontal predictor, use the Zarr codecs from imagecodecs, which zarrshadow registers when it opens such a file. Other readers need imagecodecs too, and browser readers do not have them. Strips that do not divide the image evenly are read one page at a time when the pages are uncompressed and stored in one piece.

## Electrophysiology Formats Read by NEO

`generate_rfs_neo` builds references from a [NEO](https://neo.readthedocs.io) raw reader, for the formats whose NEO readers describe where their signals are stored: SpikeGLX, Open Ephys binary, Axon, BrainVision, Elan, Micromed, NeuroNexus, Neuroscope, Multi Channel Systems raw, raw binary, WinEDR, WinWCP, and Maxwell. Install it with `pip install zarrshadow[neo]`.

```python
from neo.rawio import SpikeGLXRawIO
from zarrshadow import generate_rfs_neo, write_rfs

reader = SpikeGLXRawIO(dirname="Noise4Sam_g0")
rfs = generate_rfs_neo(reader, url_for=lambda path: "https://my-bucket/" + path)
write_rfs(rfs, "Noise4Sam_g0.zarrshadow")
```

Each of NEO's signal buffers becomes one array, time by channel, at `"<buffer id>"` for a recording with one segment and at `"block<b>/segment<s>/<buffer id>"` otherwise. The samples of a raw buffer are evenly spaced in the file, so the whole buffer is one `gen` entry however long the recording is. The array's `neo` attribute lists the streams stored in the buffer, with the columns that belong to each, the sampling rate, `t_start`, and each channel's id, name, units, gain, and offset. `url_for` maps the local paths NEO reads to where the files are hosted.

Continuous integration checks the arrays against NEO's own reads on the recordings NEO tests these readers with, which are hosted on [GIN](https://gin.g-node.org/NeuralEnsemble/ephy_testing_data): 59 recordings across 12 of these formats, with every block, segment, and stream equal. Run the same tests locally with `pytest -m gin`, which downloads about 300 MB with datalad. Maxwell recordings are stored in HDF5 with MaxWell's own compression filter, for which there is no Zarr codec, so only uncompressed Maxwell files can be referenced. When a recording ends partway through its last chunk at the end of the file, that chunk is shorter than the others; zarrshadow pads it, but other Zarr readers will not read it.

## Virtual Arrays

A `VirtualArray` describes an array stored in other files: its shape, data type, and chunking, and where its chunks are. It holds no data. It can be sliced and stacked like an array, and the result is another `VirtualArray` that points at the same bytes.

```python
from zarrshadow import RfsBuilder, VirtualArray, stack

raw = VirtualArray.contiguous("run_g0_t0.imec0.ap.bin", shape=[n_samples, 385], dtype="int16")
neural = raw[:, :384]             # drop the sync channel
sync = raw[:, 384]
first_minute = neural[: 60 * 30_000]

# One file per channel becomes a time by channel array
channels = [VirtualArray.contiguous(path, shape=[n_samples], dtype="int16", offset=16384) for path in paths]
signal = stack(channels, axis=1)

builder = RfsBuilder()
builder.add_group("")
neural.add_to(builder, "neural", dimension_names=["time", "channel"])
rfs = builder.build()
```

A slice along the first axis of a contiguous array moves the byte range. A slice or a list of indices along a later axis becomes a selection. Any other array can be indexed by whole chunks: any slice or list along an axis with one element per chunk, such as the pages of a TIFF stack (`pages[::2]`), and slices that start and stop on chunk boundaries elsewhere. `VirtualArray.from_chunks` makes an uncompressed array from chunks given one by one, each a file, an offset, and a length. `stack` joins arrays of the same shape, data type, and chunking along a new axis: each keeps its chunks, which get one more coordinate. `transpose` puts the axes in another order, as for a TIFF stack stored (page, row, column) that an NWB `TwoPhotonSeries` wants as (frame, x, y): `pages.transpose(0, 2, 1)`. The chunks stay as the file has them, and the array gets the Zarr `transpose` codec when a chunk's bytes depend on the order, which they do not when the chunk extends along one axis only. zarr-python reads whole chunks from an array with that codec, so it suits small chunks such as one frame. `VirtualArray.from_rfs(rfs, path)` takes an array from a reference file system that a generator built, which can be placed in another file or stacked. Only contiguous arrays can be sliced inside their chunks, and before they are stacked or transposed, because slicing inside the chunks of any other array would need its data. Joining arrays end to end along an existing axis is not supported.

Some formats store a packet for every sample, with a header before the values. `VirtualArray.records` describes those: each row of the array is in one record of fixed size, after `skip` bytes. `VirtualArray.from_memmap` takes a `numpy.memmap`, or a view of one, and returns the part of the file it shows, which is a short way to reference what a reader already maps into memory.

```python
# 13 bytes of header, then 64 int16 samples, in every packet
signal = VirtualArray.records(path, shape=[n_samples, 64], dtype="int16", record_size=141, skip=13, offset=header_size)

packets = np.memmap(path, dtype=[("header", "u1", 13), ("samples", "<i2", 64)], mode="r", offset=header_size)
signal = VirtualArray.from_memmap(packets["samples"])
```

`virtual_arrays_neo(reader)` returns one `VirtualArray` for each signal stream of a NEO reader, holding only that stream's channels, with the stream's sampling rate and channel information in `attributes`. `generate_rfs_neo` describes each buffer as the file stores it. `virtual_arrays_neo` also covers Blackrock and SpikeGadgets, whose NEO readers do not describe their buffers but read them through memory maps.

## Virtual NWB Files

`zarrshadow.nwb` writes an NWB file whose large datasets are references to the acquisition files, so that no signal data is read or copied. Build the `NWBFile` with [pynwb](https://pynwb.readthedocs.io) as usual and give each large dataset a `VirtualArray`'s placeholder as its data. Install with `pip install zarrshadow[nwb]`.

```python
from hdmf_zarr import NWBZarrIO
from neo.rawio import SpikeGLXRawIO
from pynwb.ecephys import ElectricalSeries
from zarrshadow import RfsStore, load_rfs, virtual_arrays_neo
from zarrshadow.nwb import write_virtual_nwb

arrays = virtual_arrays_neo(SpikeGLXRawIO(dirname="Noise4Sam_g0"))
ap = arrays["imec0.ap"]           # 384 neural channels of the 385 in the file

nwbfile = ...                     # an NWBFile with a device, an electrode group, and 384 electrodes
nwbfile.add_acquisition(
    ElectricalSeries(
        name="ElectricalSeries",
        data=ap.placeholder(),
        electrodes=nwbfile.create_electrode_table_region(list(range(384)), "all electrodes"),
        rate=ap.attributes["sampling_rate"],
        conversion=1e-6,
        channel_conversion=ap.attributes["gain"],
    )
)
write_virtual_nwb(nwbfile, "session.nwb.zarrshadow")

with NWBZarrIO(RfsStore(load_rfs("session.nwb.zarrshadow")), mode="r") as io:
    nwbfile = io.read()
```

hdmf-zarr writes the file's structure: the groups, attributes, object ids, references, the cached specification, and every dataset that holds real data. For a placeholder it writes only the array's metadata, and `write_virtual_nwb` adds the chunk locations. One file can draw on any number of source files. The result reads through `NWBZarrIO`, passes `pynwb.validate`, and can be exported to an ordinary NWB file with `NWBZarrIO.export` or `NWBHDF5IO.export`, passing `write_args={"link_data": False}` so that the data is copied.

The scaling of a series stays in NWB's `conversion`, `offset`, and `channel_conversion`, so the file's integers are referenced as they are. Continuous integration writes a SpikeGLX recording from GIN this way and compares both series with NEO's reads.

### From NeuroConv

`zarrshadow.neuroconv_bridge.virtualize` takes an NWB file that [NeuroConv](https://neuroconv.readthedocs.io) built in memory and swaps its data iterators for references, so NeuroConv supplies the metadata and the tables, and the signals stay in the source files.

```python
from neuroconv.datainterfaces import SpikeGLXRecordingInterface
from zarrshadow.neuroconv_bridge import virtualize
from zarrshadow.nwb import write_virtual_nwb

interface = SpikeGLXRecordingInterface(folder_path="Noise4Sam_g0", stream_id="imec0.ap")
nwbfile = interface.create_nwbfile(metadata=interface.get_metadata())
virtualize(nwbfile)
write_virtual_nwb(nwbfile, "session.nwb.zarrshadow")
```

For electrophysiology it covers recordings that SpikeInterface reads through a NEO reader that `virtual_arrays_neo` supports. For imaging it covers roiextractors' TIFF extractors (plain TIFF stacks, ScanImage, Thor, Micro-Manager, and Bruker with one file per frame), which keep a table of the page that holds each frame, and its HDF5 extractor. Each TIFF page becomes one chunk, and the frames are transposed into the (frame, width, height) order NeuroConv writes. Anything else raises `NotVirtualizable`: compressed TIFF pages, frames cropped out of a page, Bruker volumes, and readers that do not say where their data is. With `strict=False` those are left for NeuroConv to copy.

Continuous integration compares the result with NeuroConv's own conversion on GIN data, for SpikeGLX (AP band and NIDQ), Open Ephys binary, Neuroscope, MCS raw, Blackrock, SpikeGadgets, a TIFF stack, ScanImage (single and two channels, planes, and volumes), an HDF5 movie, Bruker, Thor, and Micro-Manager. Every dataset is equal except the file's creation time.

This is experimental. As of October 2026, SpikeInterface requires `zarr<3` and zarrshadow needs Zarr v3, so NeuroConv and zarrshadow install together only with that requirement overridden (`uv pip install --override`), and NeuroConv 0.10 reads `zarr.codec_registry` at import, which Zarr v3 removed. `tests/test_neuroconv_bridge.py` shows the environment and the one-line workaround.

## Formats That VirtualiZarr Parses

[VirtualiZarr](https://virtualizarr.readthedocs.io) parses NetCDF, HDF5, GRIB, FITS, Zarr, and kerchunk references into a `ManifestStore`: Zarr metadata and, for each array, a manifest of where its chunks are. `zarrshadow.virtualizarr` takes those manifests. Install with `pip install zarrshadow[virtualizarr]`.

```python
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtualizarr.parsers import HDFParser
from zarrshadow import write_rfs
from zarrshadow.virtualizarr import manifest_store_to_rfs, virtual_array

store = HDFParser()(url="file:///data/air.nc", registry=ObjectStoreRegistry({"file://": LocalStore()}))
write_rfs(manifest_store_to_rfs(store), "air.zarrshadow")
```

`manifest_store_to_rfs` writes the whole store as a reference file system, with a chunk index for each array that has many chunks. `virtual_array` takes one `ManifestArray` as a `VirtualArray`, to stack, transpose, or put in an NWB file. An uncompressed array stored in one piece of one file becomes a contiguous `VirtualArray`, so it can also be sliced along any axis, which a `ManifestArray` cannot do by column.

VirtualiZarr's HDF5 parser does not read NWB files, which hold variable-length strings and object references. Use `generate_rfs` for those.

## Materializing

`materialize` reads the bytes a reference file system points at and writes them into an ordinary Zarr store, which then no longer depends on the source files.

```python
from zarrshadow import materialize

report = materialize("session.nwb.zarrshadow", "session.nwb.zarr")
```

Groups, attributes, and datasets stored in the references file are copied as they are. Each array whose chunks are references is rewritten: by default an array of numbers is cut into chunks of about 4 MiB along its first axis and compressed with zarr's default compressor, and any other array is copied as stored. A `layout` function chooses per array, returning arguments for `zarr.create_array` or `None` to leave the array's chunks as they are:

```python
from zarr.codecs import BloscCodec

def layout(path, array):
    if path.endswith("ElectricalSeries/data"):
        return {"chunks": (30_000, 64), "compressors": BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")}
    return {}

materialize("session.nwb.zarrshadow", "session.nwb.zarr", layout=layout)
```

A virtual NWB file materializes into an NWB Zarr file that `NWBZarrIO` opens from its directory. The reading goes through `RfsStore`, so materializing needs none of the libraries that read the source formats, and it can run wherever the references and the source files can be reached. The returned report gives each rewritten array's size and stored size. Pass `verify=True` to read back what was written and compare.

## Other File Formats

`generate_rfs` is the generator for HDF5. Everything after it (the store, the directory format, chunk indexes, `gen`, and source checks) works for any format, and a generator for another format builds the same references with `RfsBuilder`. For a raw binary recording with 16 interleaved `int16` channels after a 12-byte header:

```python
from zarrshadow import RfsBuilder, open_rfs, write_rfs

builder = RfsBuilder()
builder.add_group("")
builder.add_array("data", shape=[10_000, 16], data_type="int16", chunk_shape=[1000, 16])
builder.add_strided_chunks("data", ndim=2, url="raw.bin", start=12, stride=32_000, length=32_000, count=10)
rfs = builder.build()
write_rfs(rfs, "raw.zarrshadow")
```

`add_chunk` adds one chunk at a time, `add_index` adds all the chunks of a large array as an index array, `add_strided_chunks` adds evenly spaced chunks as a `gen` entry, and `add_inline_chunk` stores small data in the references themselves.

## DANDI support

zarrshadow handles DANDI API URLs automatically. The DANDI URL (which returns a 302 redirect to a presigned S3 URL) is resolved transparently, with the presigned URL cached for 10 minutes.

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
from zarrshadow import LocalCache, open_rfs

cache = LocalCache()  # persists to ~/.zarrshadow/cache
root = open_rfs("example.zarrshadow.json", local_cache=cache)

# First read fetches from remote; subsequent reads are served from disk
data = root["units/spike_times"][:]
```

To prevent unbounded cache growth, set a size limit. Least-recently-accessed chunks are evicted when the limit is exceeded:

```python
cache = LocalCache(max_size_bytes=500_000_000)  # 500 MB cap
```

## Request merging

When reading a multi-chunk slice, zarrshadow automatically merges nearby HTTP Range requests into fewer, larger fetches. For example, reading 10 contiguous chunks from a remote file may result in a single HTTP request instead of 10.

Two parameters control the merging behavior:

- `merge_gap`: maximum gap in bytes between two ranges to merge (default 256 KB)
- `max_merge_size`: maximum size of a single merged request (default 50 MB)

```python
root = open_rfs("example.zarrshadow.json", merge_gap=1_000_000, max_merge_size=100_000_000)
```

Set `merge_gap=0` to disable merging and fetch every chunk individually.

## Architecture

```
zarrshadow/
├── builder.py               # RfsBuilder and write_rfs, independent of the source format
├── hdf5.py                  # HDF5 → reference file system, through RfsBuilder
├── tiff.py                  # TIFF → reference file system, through tifffile and RfsBuilder
├── neo_rawio.py             # NEO raw readers → reference file system, through RfsBuilder
├── virtual.py               # VirtualArray: slicing and stacking arrays stored in other files
├── nwb.py                   # Virtual NWB files, written through hdmf-zarr
├── neuroconv_bridge.py      # NeuroConv's in-memory NWB files → virtual NWB files
├── virtualizarr.py          # VirtualiZarr manifests → reference file system and VirtualArray
├── materialize.py           # Reference file system → ordinary Zarr store
├── open_rfs.py              # Open RFS as zarr.Group
├── rfs_store.py             # Zarr v3 Store backed by reference file system
├── chunk_index.py           # Byte-range indexes for arrays with many chunks
├── gen.py                   # Lazy evaluation of kerchunk gen entries
├── sources.py               # Recording source files and detecting changes
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
Reference file system (.zarrshadow.json, or .zarrshadow/ directory with chunk indexes)
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
