# Plan: Virtual NWB Files from Multiple Sources

Written 2026-10-05. The per-format evidence, with the reader code behind each layout claim, is in `virtual-nwb-format-survey.md`.

## Goal

Write an NWB file whose large datasets are references to bytes in the original acquisition files, so that a conversion copies no signal data. One NWB file can draw on several source files (a SpikeGLX `.bin`, a TIFF stack, an existing HDF5 file). The result must be valid NWB: `NWBZarrIO` reads it, `pynwb.validate` passes, and every dataset equals what an ordinary NeuroConv conversion writes.

## What Exists

- zarrshadow generators for HDF5, TIFF, and the 13 NEO readers with the buffer description API, tested in CI on 59 GIN recordings.
- A refs format that records several source files with size and ETag, plus `gen`, `indexes`, and (in https://github.com/bendichter/zarrshadow/pull/18) `selections`.
- `RfsStore`, which `NWBZarrIO` reads, and which reads byte ranges of a chunk as ranges.
- Partial reads of uncompressed chunks in zarr-python (https://github.com/zarr-developers/zarr-python/pull/4458, in review), zarrita.js (branch), and zarr-matlab (merged).
- An experiment showing that hdmf-zarr 0.14.0 supports skeleton and graft with no change: an empty data iterator wrapped in `ZarrDataIO` writes an array's metadata and no chunks, and the grafted file reads back, validates, and stays lazy. Two source files in one NWB file worked.

## Design

### 1. VirtualArray (zarrshadow)

One object that describes an array stored in other files: shape, dtype, chunk shape, codecs, fill value, where its chunks are (refs, a `gen` entry, or an index), its selection if any, and its sources. Generators return these instead of writing straight into a builder.

It supports basic slicing, which returns another `VirtualArray`:

- a slice along the first axis becomes a different byte range (free);
- a slice or list along a later axis of uncompressed data becomes a selection (`virtual[:, :384]`);
- anything else raises.

`stack` joins arrays along a new axis, which is how one file per channel, frame, or plane becomes one array. Joining arrays end to end along an existing axis is left out for now: Zarr's regular chunk grid allows it only when the pieces are whole numbers of chunks long.

`VirtualArray.placeholder()` returns the `ZarrDataIO` that hdmf-zarr writes as metadata only.

### 2. Sources

Each generator gains a function that returns `VirtualArray`s. For NEO that is one per stream per segment, with the stream's columns selected from its buffer, so the sync channel of a SpikeGLX file is its own array and the neural channels are another.

### 3. write_virtual_nwb (zarrshadow[nwb])

1. Find every placeholder in the in-memory `NWBFile`.
2. Write the file with `NWBZarrIO` to a `MemoryStore`. hdmf-zarr produces the specs, object IDs, references, and attributes.
3. Graft: inline every key hdmf-zarr wrote, and for each placeholder add its chunk references, `gen` entries, index, and selection under the array's path. Merge `sources`.
4. Check that each array's written `shape`, `data_type`, `chunk_grid`, and `codecs` equal the source's. Set the layout at write time through `ZarrDataIO` so the copy of the metadata in the root stays correct.
5. Write the refs file or directory.

A dataset that cannot be virtual (compressed with an unsupported filter, or needing a transform no codec expresses) is written inline as real data, with a warning that names it.

### 4. NeuroConv Bridge

NeuroConv keeps building the `NWBFile` and its metadata. The bridge replaces each interface's data iterator with a `VirtualArray` when the interface's extractor exposes its byte layout:

- SpikeInterface recordings read through NEO expose the raw reader and stream, which is what the NEO generator needs.
- `BinaryRecordingExtractor` exposes file paths, dtype, and offset directly.
- roiextractors TIFF and HDF5 imaging extractors expose the file path and series or dataset.

Channel subsets and orderings that the interface applies become slices of the `VirtualArray`. This starts as a module in zarrshadow that depends on NeuroConv, and moves into NeuroConv as a backend once the interface is stable.

## Axis Order

A reference describes bytes in the order the file has them, and NWB fixes the order of a dataset's axes. Two mechanisms cover the difference, and neither is specific to zarrshadow.

1. Chunk layout. When each chunk is a run along a single axis, the chunk grid expresses the layout and nothing is reordered. Channel-major data (all of channel 0, then all of channel 1) becomes a time-by-channel array with one-column chunks, where chunk `j` along the channel axis points at channel `j`'s bytes. ROI-by-time traces work the same way. Partial reads keep working.
2. The Zarr v3 `transpose` codec, when a chunk spans two axes in the wrong order, as with image frames stored (y, x) for a dataset NWB wants as (x, y). It composes with a selection: the store selects bytes, then the codecs decode and transpose. hdmf-zarr writes a `TransposeCodec` given through `ZarrDataIO(filters=...)`; reading data back through one is still to test. zarr-python turns off partial reads for an array with this codec, so use it where chunks are small (one frame) and prefer the chunk layout elsewhere.

Axis reordering stays out of the selection rule, which would duplicate the codec in a form every reader would have to implement.

The survey found that axis order comes up more often than extra bytes. Every imaging format needs the codec, because NeuroConv writes (frames, width, height) and the files hold (frames, rows, columns). Intan header-attached blocks and EDF records store each channel's samples together, as do Suite2p and CaImAn traces; those fit the chunk layout, with one chunk per record per channel. Referencing such a block as (time, channel) without either mechanism scrambles the data and raises no error, so each generator needs a test against the reader's own output.

Work items:

- Done: `VirtualArray.transpose`, which uses mechanism 1 when the chunks allow it and the codec otherwise, and a test that writes a TIFF stack as a `TwoPhotonSeries` through hdmf-zarr and reads it back through the codec.
- Lift the NEO generator's restriction to C-ordered, time-first buffers. None of the 13 readers on GIN needs it, so there is no real file to test it on yet.

## Values and Gaps

A reference holds the bytes the file has, which is sometimes not what NeuroConv writes. The survey found these cases, each of which would give wrong data with no error:

- Sign. SpikeInterface negates the samples when every NEO gain is negative (Neuralynx with `InputInverted`), and the Scanbox and Biocam readers invert values. The virtual series keeps the file's values and carries the sign and offset in `conversion` and `offset`. Tests compare scaled values, not stored ones.
- Unsigned samples around an offset (Intan header-attached). NeuroConv already writes `offset`, so the stored values match.
- Gaps and partial records. Neuralynx files can hold several sections with a short last record, Open Ephys legacy files have lost packets that NEO fills with zeros, and Blackrock starts a new block at each pause. A generator must check that the records are full and continuous, and otherwise write one array per segment or refuse.
- Different rates in one file (EDF signals, Intan auxiliary channels, TDT stores). Each needs its own array.
- Small records. Neuralynx, Open Ephys legacy, Intan, EDF, and TDT give chunks of 0.1 to 16 kB per channel, so remote reads depend on `RfsStore` merging neighboring requests.
- Sample sizes Zarr has no data type for (24-bit BDF and WAV). These cannot be referenced.

A column permutation needs no selection: the `electrodes` region of an `ElectricalSeries` can list the electrodes in file order. A column subset does need one.

## Validation

- For each GIN recording: convert with NeuroConv in the ordinary way and virtually, and compare every dataset and attribute.
- `pynwb.validate(io=...)` on every virtual file.
- NWB Inspector, once its two compression checks handle Zarr v3 (they raise on any Zarr v3 file today).
- Export the virtual file to HDF5 and compare, as the test of "materialize".

## Phases

0. Done: generators, selections, partial reads, hdmf-zarr feasibility.
1. Done: `VirtualArray` with slicing and stacking, NEO arrays per stream (`virtual_arrays_neo`), and `write_virtual_nwb`, tested on the GIN recordings and on a SpikeGLX recording written as NWB. Generators still write into a builder; `VirtualArray.from_rfs` takes an array from what they build.
   Also done: `materialize`, which writes a virtual file into an ordinary Zarr store with a chosen chunking and compression. It reads through `RfsStore` only, so the same function can run in an upload client or on the archive.
2. Prototype done: `zarrshadow.neuroconv_bridge.virtualize` swaps the data iterators in an NWB file that NeuroConv built for references. It finds each iterator's SpikeInterface recording, walks through channel slices to the recording that holds the NEO reader, and takes that stream's array from `virtual_arrays_neo`. Compared with NeuroConv's own conversion on GIN data, every dataset is equal except the file's creation time, for SpikeGLX (AP band and NIDQ), Open Ephys binary with and without a sync channel, Neuroscope, and MCS raw. Recordings whose samples SpikeInterface negates, and iterators that return scaled values, are refused.

   Two things stand between the prototype and something users can install:
   - SpikeInterface requires `zarr<3` (0.105.1), and zarrshadow needs Zarr v3. Zarr v3 support is in progress at https://github.com/SpikeInterface/spikeinterface/pull/4260. Until then the two install together only with the requirement overridden.
   - NeuroConv 0.10.2 reads `zarr.codec_registry` at import, which Zarr v3 removed. Building an NWB file in memory works once that one attribute is put back, and the bridge needs nothing else from NeuroConv. NeuroConv's own Zarr backend was not tested under Zarr v3. NeuroConv pins `zarr<3` on its main branch. Zarr v3 support is tracked in https://github.com/catalystneuro/neuroconv/issues/2076, and https://github.com/catalystneuro/neuroconv/pull/1749 is a draft port of its Zarr backend.
3. Imaging. Done: a TIFF stack as an NWB series, through `VirtualArray.from_rfs`, `stack` for one file per frame, and `transpose`; and the NeuroConv side, where `virtualize` reads roiextractors' table of the page that holds each frame and makes each page a chunk (`VirtualArray.from_chunks`). Compared with NeuroConv's own conversion on GIN data, every dataset is equal for a TIFF stack, ScanImage (single and two channels, planes, and volumes), an HDF5 movie, Bruker with one file per frame, Thor, and Micro-Manager. Refused for now: compressed pages (the Thor LZW test set), frames cropped out of a page (ScanImage's field-of-view window), and Bruker volumes, which roiextractors assembles from several extractors.
4. New ephys generators. Done: Blackrock nsX (specs 2.1 to 3.0, with one block per pause, and the files with a packet per sample) and SpikeGadgets. NEO's readers for these do not have the buffer description API, but both read through memory maps, and a memory map says where its data is: `VirtualArray.from_memmap` turns one, or a view of one, into a virtual array, and `VirtualArray.records` describes rows stored in packets of fixed size. `virtual_arrays_neo` uses these for the two readers, so `virtualize` covers them with no change to the bridge. Checked against NEO on the five Blackrock and four SpikeGadgets recordings on GIN, and against NeuroConv's own conversion for both. `generate_rfs_neo` does not cover them.

   Also done: Intan, in its three layouts. One file per signal is a plain binary, and one file per channel is a stack of them. A file with blocks after the header holds each channel's samples together in a block, so `VirtualArray.blocks` makes each block one chunk, described as channels by samples and then transposed. One `gen` entry covers every block of a stream. The digital channels and the stimulation current have no array, because NEO computes them from the stored words. Checked against NEO on the twelve Intan recordings on GIN and against NeuroConv's own conversion for the three layouts. The blocks are small (60 or 128 samples), so a remote read of such a file depends on merging neighboring requests, and EDF, Neuralynx, and Open Ephys legacy will be the same.

   NEO maps Intan's files by their resolved paths, so for a recording reached through links, as in a datalad dataset, a memory map names the link's target. The references use the recording's own paths, which the tests check for all three readers.

   Also done: Neuralynx, Open Ephys legacy, and EDF, with `VirtualArray.blocks`. Neuralynx and Open Ephys keep a file for each channel, in records of 512 and 1024 samples, so a stream is a stack of per-channel arrays with one chunk per record. NEO reads EDF through pyedflib, which has no memory map to look at, so the layout is read from the EDF header: signals of one rate that are next to one another in a data record become one transposed chunk per record, and others one chunk per record each. Checked against NEO on ten Neuralynx, three Open Ephys, and eight EDF recordings on GIN, and against NeuroConv for Open Ephys and EDF. Refused: an Open Ephys recording with gaps (NEO fills them with zeros at positions that follow the timestamps), discontinuous EDF+, and BDF.

   Neuralynx through NeuroConv is refused in the usual case. With inverted input NEO reports a negative gain, and SpikeInterface negates the samples and keeps the gain positive, so NeuroConv writes values the file does not hold. A virtual file could reference the stored values and negate `conversion`, which gives the same voltages in a file that differs from NeuroConv's. That is a decision about the bridge, not about the format.

   Also done: WhiteMatter, CellExplorer, and 16-bit WAV, in the bridge. These need no generator, because `VirtualArray.contiguous` already describes a plain binary file. SpikeInterface reads WhiteMatter and CellExplorer with its `BinaryRecordingExtractor`, whose segments give the file, offset, data type, and time axis, and NeuroConv maps a WAV file into memory, which `VirtualArray.from_memmap` reads. Compared with NeuroConv's own conversion for all three. A 24-bit WAV file is not mapped and has no Zarr data type.
5. Readers and hosting. Done: a JavaScript store for zarrita.js in `js/`, which reads everything the Python store reads, selections included, and is tested against files the Python package writes. It read two NWB files from DANDI in Node and in Chrome with the same values as Python. It fetches reads of one file that are close together in one request of up to 1 MiB, a limit chosen from timings in Chrome and Node (js/README.md): one request for everything was the slowest setting, about three times slower than the best. The Python store merges up to 50 MB by default, which those timings suggest is worth measuring too. Remaining: the hdmf-zarr conventions (links, object references, compound types) for JavaScript, which belong in a package of their own; selections in the MATLAB reader; and where source files live.
6. Propose the bridge as a NeuroConv backend.

## Source Formats

Classes: A, an existing zarrshadow generator covers it. B, one contiguous raw block. C, fixed-size records, one chunk per record. D, needs a selection. E, not feasible. F, nothing to gain. "+T" needs the transpose codec, and "+S" needs a sign or offset carried in `conversion` and `offset`. The evidence column says whether the layout was checked against NEO's read on a GIN file or read in the reader's source only.

As of October 2026 the table's rows are implemented down to EDF, except Axon ABF as one series per sweep per channel, MEArec, and Biocam, which have not been tried through the bridge. The imaging rows for TIFF and HDF5 are implemented as phase 3 describes. TDT, Axona, Scanbox, the segmentation outputs, and Minian are not.

| Format | Layout | Class | Evidence |
|---|---|---|---|
| SpikeGLX | raw int16 (time, channel); sync is the last column | A + D | GIN file |
| Open Ephys binary | raw int16 (time, channel); ADC columns may share the file | A, + D if mixed | GIN file |
| Neuroscope, MCS raw | raw (time, channel) | A | GIN file |
| Axon ABF | interleaved (time, channel) per sweep; NeuroConv writes one series per sweep per channel | A, D per series | GIN file, source |
| MEArec | HDF5 (time, channel) float32, contiguous | A | GIN file |
| Biocam | HDF5 flat 1-D uint16, time-major; a sparse variant exists | A or B, +S; sparse E | GIN file |
| WhiteMatter, CellExplorer | int16 (time, channel), 8-byte header for WhiteMatter | B | GIN file, source |
| Blackrock 2.1 to 3.0 | headers, then one int16 block per pause | B per segment | GIN file |
| Blackrock 3.0 PTP | per-sample packets, 13 bytes plus samples | D | GIN file |
| SpikeGadgets | per-sample packets; ephys channels are the last bytes | D | GIN file |
| Intan, split files | headerless int16, one file per signal or per channel | B | GIN file |
| Intan `.rhd` and `.rhs` | blocks of 60 or 128 samples, channel-major in a block, uint16 | C, +S | GIN file |
| Neuralynx `.ncs` | per-channel file, 16 kB header, 1044-byte records of 512 int16 | C, +S | GIN file |
| Open Ephys legacy | per-channel file, 1024-byte header, 2070-byte records of 1024 big-endian int16 | C | GIN file |
| EDF | records, channel-major in a record | C | GIN file |
| TDT `.tev` | per-channel chunks at irregular offsets listed in `.tsq` | C with an index | GIN file |
| TDT `.sev` | per-channel file, addressed through `.tsq` | C, maybe B | source; contiguity not verified |
| Axona `.bin` | 432-byte packets of 3 samples by 64 channels, permuted, only active tetrodes exposed | D with several ranges, else E | GIN file |
| BDF | 24-bit samples | E | source |
| Maxwell, Plexon, Plexon2, Spike2, AlphaOmega | proprietary filter, variable blocks, or a closed reader | E | GIN file, source |
| TIFF, ScanImage, Bruker, Micro-Manager, Thor | pages, chosen per channel and plane by the extractor | A +T | source |
| HDF5 imaging, Femtonics | one dataset | A +T | source |
| Scanbox | no header, uint16 (frame, row, column); the reader returns 65535 minus the value | B +T +S | source |
| Inscopix | read through the vendor library | unknown | not verified |
| Miniscope `.avi` | video codec | E | source |
| Suite2p, CaImAn, CNMF-E, EXTRACT traces | (ROI, time) in `.npy` or HDF5 | B or A; masks F | source |
| Minian | Zarr v2 folders | A-like | source |
| WAV | RIFF data chunk, interleaved PCM | B; 24-bit E | source |
| External video, pose estimation, sorters, events, TDT photometry | already external, or small and regrouped | F | source |

NEO's buffer description API covers six of NeuroConv's ephys formats (SpikeGLX, Open Ephys binary, Neuroscope, MCS raw, Axon, and Maxwell, which is unusable). The other formats need generators of their own. NeuroConv has no recording interface for MDA, only a sorting interface.

The survey read NeuroConv at a development checkout (v0.10.1-18), NEO 0.14.4, SpikeInterface 0.103.1, and roiextractors 0.5.12 with the 0.10.0 source for extractors the installed version lacks.

## Relation to VirtualiZarr

VirtualiZarr's `ManifestArray` and `ManifestStore` cover much of what `VirtualArray` and `RfsStore` do, so in October 2026 we tried building on them. `zarrshadow.virtualizarr` came out of that: it writes a `ManifestStore` in this format and takes a `ManifestArray` as a `VirtualArray`. The rest stays separate for now, for these reasons.

- VirtualiZarr's HDF5 parser raises on object references and cannot read variable-length strings, so it does not open an NWB file (https://github.com/zarr-developers/VirtualiZarr/issues/1104).
- A `ManifestArray` cannot select columns, can slice rows inside a chunk only when the array is a single chunk, and has no transpose. Ephys with a sync channel and imaging depend on those.
- Its kerchunk writer emits Zarr v2 metadata from an xarray Dataset. https://github.com/zarr-developers/VirtualiZarr/pull/1118 adds Zarr v3 metadata as an option.
- It has no chunk index format. Its answer for large reference sets is kerchunk Parquet, which holds Zarr v2 metadata only, or Icechunk.

VirtualiZarr adds concatenation, which `VirtualArray` lacks, and parsers for NetCDF, GRIB, FITS, Zarr, and DMR++. Kerchunk itself went into maintenance mode in September 2026 and points new projects to VirtualiZarr, so proposals about the format belong there.

## Open Questions

- Hosting. A virtual file is useful only while its sources stay at stable URLs. Where do raw acquisition files live, and does DANDI accept them alongside the refs file?
- Publishing on DANDI. DANDI accepts and encourages Zarr v3, so a materialized NWB Zarr file can be uploaded. As of October 2026 it has no way to publish a Zarr-based dataset, that is, to create a persistent version of one.
- Several files in one series. Zarr's regular chunk grid joins files along an axis only when their lengths line up. The alternative is one series per file.
- Local use. HDF5 external storage would give an NWB (or NIX) HDF5 file that existing readers open unchanged, for local files only. Is that worth a second output format?
- NIX. No NIX reader reads Zarr, so a virtual NIX file would use HDF5 external storage, and NEO's NIX reader expects one array per channel.
