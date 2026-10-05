# Plan: Virtual NWB Files from Multiple Sources

Written 2026-10-05. The per-format evidence, with the reader code behind each layout claim, is in `virtual-nwb-format-survey.md`.

## Goal

Write an NWB file whose large datasets are references to bytes in the original acquisition files, so that a conversion copies no signal data. One NWB file can draw on several source files (a SpikeGLX `.bin`, a TIFF stack, an existing HDF5 file). The result must be valid NWB: `NWBZarrIO` reads it, `pynwb.validate` passes, and every dataset equals what an ordinary NeuroConv conversion writes.

## What Exists

- zindi generators for HDF5, TIFF, and the 13 NEO readers with the buffer description API, tested in CI on 59 GIN recordings.
- A refs format that records several source files with size and ETag, plus `gen`, `indexes`, and (in https://github.com/bendichter/zindi/pull/18) `selections`.
- `RfsStore`, which `NWBZarrIO` reads, and which reads byte ranges of a chunk as ranges.
- Partial reads of uncompressed chunks in zarr-python (https://github.com/zarr-developers/zarr-python/pull/4458, in review), zarrita.js (branch), and zarr-matlab (merged).
- An experiment showing that hdmf-zarr 0.14.0 supports skeleton and graft with no change: an empty data iterator wrapped in `ZarrDataIO` writes an array's metadata and no chunks, and the grafted file reads back, validates, and stays lazy. Two source files in one NWB file worked.

## Design

### 1. VirtualArray (zindi)

One object that describes an array stored in other files: shape, dtype, chunk shape, codecs, fill value, where its chunks are (refs, a `gen` entry, or an index), its selection if any, and its sources. Generators return these instead of writing straight into a builder.

It supports basic slicing, which returns another `VirtualArray`:

- a slice along the first axis becomes a different byte range (free);
- a slice or list along a later axis of uncompressed data becomes a selection (`virtual[:, :384]`);
- anything else raises.

`stack` joins arrays along a new axis, which is how one file per channel, frame, or plane becomes one array. Joining arrays end to end along an existing axis is left out for now: Zarr's regular chunk grid allows it only when the pieces are whole numbers of chunks long.

`VirtualArray.placeholder()` returns the `ZarrDataIO` that hdmf-zarr writes as metadata only.

### 2. Sources

Each generator gains a function that returns `VirtualArray`s. For NEO that is one per stream per segment, with the stream's columns selected from its buffer, so the sync channel of a SpikeGLX file is its own array and the neural channels are another.

### 3. write_virtual_nwb (zindi[nwb])

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

Channel subsets and orderings that the interface applies become slices of the `VirtualArray`. This starts as a module in zindi that depends on NeuroConv, and moves into NeuroConv as a backend once the interface is stable.

## Axis Order

A reference describes bytes in the order the file has them, and NWB fixes the order of a dataset's axes. Two mechanisms cover the difference, and neither is specific to zindi.

1. Chunk layout. When each chunk is a run along a single axis, the chunk grid expresses the layout and nothing is reordered. Channel-major data (all of channel 0, then all of channel 1) becomes a time-by-channel array with one-column chunks, where chunk `j` along the channel axis points at channel `j`'s bytes. ROI-by-time traces work the same way. Partial reads keep working.
2. The Zarr v3 `transpose` codec, when a chunk spans two axes in the wrong order, as with image frames stored (y, x) for a dataset NWB wants as (x, y). It composes with a selection: the store selects bytes, then the codecs decode and transpose. hdmf-zarr writes a `TransposeCodec` given through `ZarrDataIO(filters=...)`; reading data back through one is still to test. zarr-python turns off partial reads for an array with this codec, so use it where chunks are small (one frame) and prefer the chunk layout elsewhere.

Axis reordering stays out of the selection rule, which would duplicate the codec in a form every reader would have to implement.

The survey found that axis order comes up more often than extra bytes. Every imaging format needs the codec, because NeuroConv writes (frames, width, height) and the files hold (frames, rows, columns). Intan header-attached blocks and EDF records store each channel's samples together, as do Suite2p and CaImAn traces; those fit the chunk layout, with one chunk per record per channel. Referencing such a block as (time, channel) without either mechanism scrambles the data and raises no error, so each generator needs a test against the reader's own output.

Work items:

- `VirtualArray.transpose(order)`, which picks mechanism 1 when the chunks allow it and the codec otherwise.
- Lift the NEO generator's restriction to C-ordered, time-first buffers.
- A test that reads data through `TransposeCodec` in an hdmf-zarr file.

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
2. Prototype done: `zindi.neuroconv_bridge.virtualize` swaps the data iterators in an NWB file that NeuroConv built for references. It finds each iterator's SpikeInterface recording, walks through channel slices to the recording that holds the NEO reader, and takes that stream's array from `virtual_arrays_neo`. Compared with NeuroConv's own conversion on GIN data, every dataset is equal except the file's creation time, for SpikeGLX (AP band and NIDQ), Open Ephys binary with and without a sync channel, Neuroscope, and MCS raw. Recordings whose samples SpikeInterface negates, and iterators that return scaled values, are refused.

   Two things stand between the prototype and something users can install:
   - SpikeInterface requires `zarr<3` (0.105.1), and zindi needs Zarr v3. Zarr v3 support is in progress at https://github.com/SpikeInterface/spikeinterface/pull/4260. Until then the two install together only with the requirement overridden.
   - NeuroConv 0.10.2 reads `zarr.codec_registry` at import, which Zarr v3 removed. Building an NWB file in memory works once that one attribute is put back, and the bridge needs nothing else from NeuroConv. NeuroConv's own Zarr backend was not tested under Zarr v3. NeuroConv pins `zarr<3` on its main branch. Zarr v3 support is tracked in https://github.com/catalystneuro/neuroconv/issues/2076, and https://github.com/catalystneuro/neuroconv/pull/1749 is a draft port of its Zarr backend.
3. Imaging: the TIFF family (ScanImage, Bruker, Micro-Manager, Thor) through the extractors' page tables, and HDF5 imaging. This is the largest data volume NeuroConv handles, and it brings in the transpose work that every other imaging format reuses.
4. New ephys generators, in this order:
   - Blackrock nsX, specs 2.1 to 3.0: one contiguous block per segment.
   - SpikeGadgets: per-sample packets, the direct use of selections. The same rule covers Blackrock PTP files.
   - Intan: the split-file modes are plain binaries; the header-attached mode is the first format with one chunk per record per channel, which EDF, Neuralynx, and Open Ephys legacy reuse.
   - A raw-binary generator taking offset, dtype, and channel count covers WhiteMatter, CellExplorer, and 16-bit WAV in a few lines each.
5. Readers and hosting: selections in the JavaScript and MATLAB readers, and where source files live.
6. Propose the bridge as a NeuroConv backend.

## Source Formats

Classes: A, an existing zindi generator covers it. B, one contiguous raw block. C, fixed-size records, one chunk per record. D, needs a selection. E, not feasible. F, nothing to gain. "+T" needs the transpose codec, and "+S" needs a sign or offset carried in `conversion` and `offset`. The evidence column says whether the layout was checked against NEO's read on a GIN file or read in the reader's source only.

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

## Open Questions

- Hosting. A virtual file is useful only while its sources stay at stable URLs. Where do raw acquisition files live, and does DANDI accept them alongside the refs file?
- Several files in one series. Zarr's regular chunk grid joins files along an axis only when their lengths line up. The alternative is one series per file.
- Local use. HDF5 external storage would give an NWB (or NIX) HDF5 file that existing readers open unchanged, for local files only. Is that worth a second output format?
- NIX. No NIX reader reads Zarr, so a virtual NIX file would use HDF5 external storage, and NEO's NIX reader expects one array per channel.
