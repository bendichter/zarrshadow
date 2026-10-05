# Byte layout of NeuroConv source formats

Research survey for virtual (reference-based) NWB datasets. Nothing here is implemented.

## What was read

| Package | Version read | Where |
|---|---|---|
| neuroconv | dev checkout `v0.10.1-18-g9daf94e46` (pip metadata still says 0.9.4) | `src/neuroconv` |
| neo | 0.14.4 (neuroconv dev asks for >=0.14.5; the buffer-API reader set is the same 13 in 0.14.5, per the zindi CI run) | `neo/rawio` |
| spikeinterface | 0.103.1 (neuroconv dev asks for >=0.104.7) | `spikeinterface` |
| roiextractors | 0.5.12 installed; 0.10.0 sdist unpacked for the extractors missing from 0.5.12 (Femtonics, Minian, MultiTIFF, OME) | `roiextractors` |
| GIN ephy_testing_data | commit `d7796f594` | https://gin.g-node.org/NeuralEnsemble/ephy_testing_data |

Layouts marked as verified on a GIN file were checked with scripts that apply the layout to the raw bytes and compare the result with NEO's own read. The scripts are not part of this repository.

Verification marks: **[F]** verified on a GIN file, **[S]** verified in reader source, **[N]** not verified.

## Classes

- **A** covered by an existing zindi generator (HDF5, TIFF, NEO buffer API)
- **B** one contiguous raw block; a trivial new generator
- **C** fixed-size records, one chunk per record (strided chunks, which `gen` entries express)
- **D** needs the byte-selection rule (keep some bytes of every N)
- **E** needs a custom codec or is not feasible
- **F** nothing to gain (already external, or small and restructured)
- **+T** also needs the Zarr v3 `transpose` codec, which zindi does not emit yet and which turns off the bytes-only partial-read path
- **+S** needs a scale/offset or sign change carried in NWB `conversion`/`offset`, so the stored values differ from what NeuroConv writes

## Table

### Extracellular electrophysiology

NeuroConv writes `ElectricalSeries.data` as (time, channel) in the source integer dtype, with `conversion` or `channel_conversion` and one scalar `offset` (`tools/spikeinterface/spikeinterface.py:688-708`) [S].

| Format | NeuroConv -> extractor -> reader | On-disk layout of the signal | Mismatch with what NeuroConv writes | NEO buffer API | Class | Mark |
|---|---|---|---|---|---|---|
| SpikeGLX | `SpikeGLXRecordingInterface` -> `SpikeGLXRecordingExtractor` -> `SpikeGLXRawIO` | no header, int16 LE, (time, channel), one block per .bin | the sync word is the last column of the same block and is a separate stream; the AP/LF stream must drop it | yes | A + D | [F] (zindi GIN tests) |
| Open Ephys binary | `OpenEphysBinaryRecordingInterface` -> `OpenEphysBinaryRecordingExtractor` -> `OpenEphysBinaryRawIO` | `continuous.dat`, no header, int16 LE, (time, channel) | neural and ADC/non-neural channels share one file and are split into streams by column | yes | A, + D when the file mixes channel kinds | [F] (zindi GIN tests) |
| Open Ephys legacy | `OpenEphysLegacyRecordingInterface` -> `OpenEphysLegacyRecordingExtractor` -> `OpenEphysRawIO` | one `.continuous` file per channel: 1024-byte header, then 2070-byte records = int64 timestamp, uint16 n, uint16 rec_num, 1024 samples **big-endian** int16, 10 marker bytes | one file per channel, so the array is (time, 1)-column chunks over many files | no | C | [F] `101_CH0.continuous` |
| Neuroscope | `NeuroScopeRecordingInterface`/`LFPInterface` -> `NeuroScopeRecordingExtractor` -> `NeuroScopeRawIO` | no header, int16 (or int32) LE, (time, channel) | none | yes | A | [F] (zindi GIN tests) |
| CellExplorer recording | `CellExplorerRecordingInterface` -> `BinaryRecordingExtractor` | raw binary described by `session.mat` | none | n/a (hook: `recording._kwargs`: `file_paths`, `dtype`, `file_offset`, `time_axis`, `num_channels`) | B | [S] `cellexplorerdatainterface.py:298-313` |
| WhiteMatter | `WhiteMatterRecordingInterface` -> `WhiteMatterRecordingExtractor` (a `BinaryRecordingExtractor`) | 8-byte header, int16 LE, (time, channel) | none | n/a (same `_kwargs` hook) | B | [F] stub file, 25000 x 64 |
| MCS raw | `MCSRawRecordingInterface` -> `MCSRawRecordingExtractor` -> `RawMCSRawIO` | text header, then one (time, channel) block | none | yes | A | [F] (zindi GIN tests) |
| Blackrock nsX, spec 2.1 | `BlackrockRecordingInterface` -> `BlackrockRecordingExtractor` -> `BlackrockRawIO` | header of 32 + 4*n_ch bytes, then one int16 LE (time, channel) block | NEO drops the last sample row | no | B | [F] `l101210-001.ns5` |
| Blackrock nsX, spec 2.2, 2.3, 3.0 | same | headers, then data blocks: a 9-byte (2.x) or 13-byte (3.0) block header (flag, timestamp, n points) followed by n x n_ch int16 | one block per pause; each block is a NEO segment | no | B per block (one array per segment) | [F] `FileSpec2.3001.ns5`, `file_spec_3_0.ns6`, `pause_correct.ns2` (2 blocks) |
| Blackrock nsX, 3.0 PTP | same | per-sample packets: 13 bytes (reserved, uint64 timestamp, uint32 n=1) + n_ch int16 | 13 extra bytes on every sample | no | D | [F] `20231027-125608-001.ns6`, packet 143 B |
| Neuralynx `.ncs` | `NeuralynxRecordingInterface` -> `NeuralynxRecordingExtractor` -> `NeuralynxRawIO` | one file per channel: 16384-byte text header, then 1044-byte records = uint64 timestamp, uint32 channel, uint32 rate, uint32 nb_valid, 512 int16 LE | per-channel files; see hazards (partial records, gaps, sign) | no | C (+S) | [F] five GIN folders |
| Intan, one file per signal | `IntanRecordingInterface` -> `IntanRecordingExtractor` -> `IntanRawIO` | `amplifier.dat`: no header, int16 LE, (time, channel) | none | no | B | [F] |
| Intan, one file per channel | same (or `IntanSplitFilesRecordingExtractor`) | `amp-*.dat`: one headerless int16 file per channel | per-channel files | no | B (many sources) | [F] 64 files |
| Intan, header-attached `.rhd`/`.rhs` | same | variable header, then fixed-size blocks of 60 (RHD USB board) or 128 samples. Inside a block: timestamps, then for each amplifier channel all of its samples in a row (uint16), then aux (1/4 rate), supply, temperature, ADC, digital | the amplifier region is contiguous within a block but **channel-major**; uint16 with an offset of 32768 | no | C +T | [F] `intan_rhd_test_1.rhd` (block 24372 B, 60 samples, 192 ch), `intan_rhs_test_1.rhs` (block 18432 B, 128 samples, 32 ch) |
| SpikeGadgets `.rec` | `SpikeGadgetsRecordingInterface` -> `SpikeGadgetsRecordingExtractor` -> `SpikeGadgetsRawIO` | XML header, then per-sample packets: sync byte, device bytes, uint32 timestamp, then n_ch int16 LE | every sample has a prefix (14 to 152 bytes before the ephys channels in the GIN files); auxiliary streams such as ECU sit in the middle of the packet | no | D | [F] three files; channels are the last 2*n_ch bytes, ascending |
| TDT `.tev` | `TdtRecordingInterface` -> `TdtRecordingExtractor` -> `TdtRawIO` | chunks of `NumPoints` samples for one channel each, located through the `.tsq` index; interleaved with other stores and events, so offsets are irregular | chunks are tiny (64 samples = 128 B in the GIN file) and per channel | no | C, with an explicit chunk index instead of `gen` | [F] `aep_05`: 16 ch, 10277 chunks per channel |
| TDT `.sev` | same | one file per channel, addressed through the same `.tsq` offsets | per-channel files | no | C (B if contiguous) | [S] `tdtrawio.py:314`; contiguity [N] (GIN `.sev` files are 80-byte stubs) |
| EDF | `EDFRecordingInterface` -> `EDFRecordingExtractor` -> `EDFRawIO` (pyedflib) | 256*(1+ns)-byte header, then data records; inside a record each signal's samples are contiguous (channel-major), int16 LE; an annotations signal may sit in every record | channel-major per record; signals may differ in samples per record (NEO makes one stream per rate) | no | C with (samples, 1) chunks per (record, signal), or C +T when a stream's signals are adjacent | [F] `edf+C.edf` |
| BDF | same | as EDF with 24-bit samples | Zarr has no 24-bit integer | no | E | [S] |
| Axona `.bin` | `AxonaRecordingInterface` -> `AxonaRecordingExtractor` -> `AxonaRawIO` | 432-byte packets: 32-byte head, 3 samples x 64 channels int16, 16-byte tail | fixed column permutation, and NEO exposes only the active tetrodes (16 of 64 in the GIN file), which are not one contiguous span | no | D with several spans per packet, plus a column permutation; otherwise E | [F] `axona_raw.bin` |
| MEArec | `MEArecRecordingInterface` -> `MEArecRecordingExtractor` -> `MEArecRawIO` | HDF5 dataset `recordings`, (time, channel) float32, contiguous, uncompressed in the GIN file | none | no | A (HDF5) | [F] `mearec_test_10s.h5` |
| Biocam `.brw` | `BiocamRecordingInterface` -> `BiocamRecordingExtractor` -> `BiocamRawIO` | HDF5. `3BData/Raw` or `Well_*/Raw`: a **flat 1-D** uint16 dataset of frames*channels in time-major order (format 100 is 2-D). Contiguous and uncompressed in the GIN files | needs a 2-D view of a 1-D dataset; when `SignalInversion` is -1 the reader returns `4096 - raw` | no | A/B (take the dataset offset and treat it as a raw block) +S | [F] two files; inversion [S] `biocamrawio.py:332-342` |
| Biocam, event-based sparse | same | `EventsBasedSparseRaw`, a custom sparse encoding | must be decoded | no | E | [F] `BioCAM_BrainWave5_HW_3.0_FW_1.7.brw` |
| Maxwell | `MaxOneRecordingInterface` -> `MaxwellRecordingExtractor` -> `MaxwellRawIO` | HDF5 with compression filter 401 | no Zarr codec for the filter | yes | E | [F] (zindi GIN tests) |
| Plexon `.plx` | `PlexonRecordingInterface` -> `PlexonRecordingExtractor` -> `PlexonRawIO` | data blocks with a 16-byte header and n1*n2 int16 words; continuous, spike and event blocks of all channels are interleaved, and block sizes vary | variable-length blocks cannot be a regular chunk grid | no | E | [S] `plexonrawio.py:219-225, 473-479` |
| Plexon2 `.pl2` | `Plexon2RecordingInterface` -> `Plexon2RawIO` | read through the vendor DLL (Wine off Windows) | layout not available | no | E | [S] `plexon2rawio.py:4-6` |
| Spike2 `.smr`/`.smrx` | `Spike2RecordingInterface` -> `CedRecordingExtractor` -> `CedRawIO` (closed-source `sonpy`) | linked lists of per-channel blocks | layout not available from the reader NeuroConv uses | no | E | [S] `cedrawio.py:70-72` |
| AlphaOmega `.mpx` | `AlphaOmegaRecordingInterface` -> `AlphaOmegaRawIO` | typed blocks with a length field, channels interleaved | variable-length blocks | no | E | [S] `alphaomegarawio.py:143-230` |
| MDA | `MdaSortingInterface` only | `firings.mda` | NeuroConv has no MDA recording interface | n/a | F | [S] `ecephys/mda/mdadatainterface.py:7` |
| Kilosort, Phy, other sorters | sorting interfaces | small `.npy`/text | regrouped into the ragged Units table | n/a | F | [S] |

### Intracellular electrophysiology

| Format | Chain | Layout | Mismatch | NEO buffer API | Class | Mark |
|---|---|---|---|---|---|---|
| Axon ABF | `AbfInterface` -> `neo.AxonIO` (legacy icephys path) | header, then int16 or float32 (time, channel) interleaved; one region per sweep | NeuroConv writes one `PatchClampSeries` per sweep **per channel** (`tools/neo/neo.py:305-332`), so every series is one column; stimulus series are synthesized from the protocol and are not in the file | yes | A for the source; D per series when there is more than one ADC channel; stimulus F | layout [F] (zindi GIN tests), NWB mapping [S] |

### Optical physiology, imaging

NeuroConv writes `TwoPhotonSeries`/`OnePhotonSeries` as (frames, **width, height**), transposing the extractor's (frames, height, width); volumetric data is (frames, width, height, planes) (`tools/roiextractors/imagingextractordatachunkiterator.py:192-194`, `roiextractors.py:1037-1040`) [S]. Every imaging source therefore needs +T to match NeuroConv.

| Format | Chain | Layout | Mismatch | Class | Mark |
|---|---|---|---|---|---|
| TIFF, multi-page or multi-file | `TiffImagingInterface` -> `MultiTIFFMultiPageExtractor` | TIFF pages via tifffile; `dimension_order` says how channels and planes interleave across pages | page selection per channel/plane | A +T, one chunk per page | [S] roiextractors 0.10.0 `multitiffmultipageextractor.py:241` (`_frames_to_ifd_table`) |
| ScanImage | `ScanImage*Interface` -> `ScanImageImagingExtractor` | BigTIFF pages over one or more files; channels and planes interleaved; flyback frames present | pages chosen through `_frames_to_ifd_table`; flyback pages skipped | A +T, one chunk per page | [S] `scanimagetiffimagingextractor.py:291-324, 614-618` |
| Bruker | `BrukerTiff*Interface` -> `BrukerTiffImagingExtractor` | one `.ome.tif` per frame, plane and channel | thousands of source files in one array | A +T, one chunk per file | [S] `brukertiffimagingextractor.py:157, 928-936` |
| Micro-Manager, Thor | -> `MicroManagerTiffImagingExtractor`, `ThorTiffImagingExtractor` (OME-TIFF) | multi-page OME-TIFF | page selection | A +T | [S] |
| HDF5 imaging | `Hdf5ImagingInterface` -> `Hdf5ImagingExtractor` | one HDF5 dataset (`mov`), read through a lazy transpose | axis order | A +T | [S] `hdf5imagingextractor.py:67-69, 95` |
| Femtonics `.mesc` | `FemtonicsImagingInterface` -> `FemtonicsImagingExtractor` | HDF5 dataset `MSession_n/MUnit_n/Channel_n`, read as stored | axis order | A +T | [S] roiextractors 0.10.0 `femtonicsimagingextractor.py:155-168, 557`; chunking and compression [N] |
| Scanbox `.sbx` | `SbxImagingInterface` -> `SbxImagingExtractor` | no header (metadata in a `.mat`), uint16; the memmap is Fortran-order (channel, col, row, plane, frame), which is (frame, plane, row, col, channel) in C order | the extractor returns `65535 - x`; it raises for more than one channel or plane | B +T +S | [S] roiextractors 0.5.12 `sbximagingextractor.py:155-165, 191, 219-220` |
| Inscopix `.isxd` | `InscopixImagingInterface` -> `InscopixImagingExtractor` (`isx` library) | read through the vendor library | layout not visible in the reader | B or C, unknown | [N] |
| Miniscope `.avi` | `MiniscopeImagingInterface` -> `MiniscopeImagingExtractor` (OpenCV) | video container with a video codec | frames are decoded | E | [S] `miniscopeimagingextractor.py:122-132` |

### Optical physiology, segmentation

NeuroConv writes traces as (time, ROI) and rebuilds the masks into the plane segmentation table. Traces are small next to the raw movie.

| Format | Chain | Layout | Mismatch | Class | Mark |
|---|---|---|---|---|---|
| Suite2p | `Suite2pSegmentationInterface` -> `Suite2pSegmentationExtractor` | `F.npy`, `Fneu.npy`, `spks.npy`: NPY header then (ROI, time) | read with `.T`; chunks can only split along ROIs | B +T for traces, F for masks | [S] `suite2psegmentationextractor.py:139-145, 242-244` |
| CaImAn | -> `CaimanSegmentationExtractor` | HDF5 | traces transposed; sparse masks rebuilt | A +T for traces, F for masks | [S] roiextractors 0.10.0 `caimansegmentationextractor.py:139, 323, 371` (h5py, `estimates/*`, lazy transpose); chunking and compression [N] |
| CNMF-E, EXTRACT | -> `CnmfeSegmentationExtractor`, `ExtractSegmentationExtractor` | MATLAB v7.3 (HDF5) | same | A +T for traces, F for masks | [S] reader is h5py with a lazy transpose (`cnmfesegmentationextractor.py:86, 99`); dataset layout [N] |
| Minian | -> `MinianSegmentationExtractor` | Zarr v2 folders (`C.zarr`, `S.zarr`, ...) | already Zarr; needs a v2-to-v3 reference generator | A-like (new, simple) | [S] roiextractors 0.10.0 `miniansegmentationextractor.py:34-42` |
| SIMA | -> `SimaSegmentationExtractor` | pickles | not addressable | F | [S] |

### Behavior and fiber photometry

| Format | Chain | Layout | Mismatch | Class | Mark |
|---|---|---|---|---|---|
| WAV audio | `AudioInterface` -> `scipy.io.wavfile.read(mmap=True)` | RIFF chunks; the `data` chunk holds interleaved PCM (time, channel) | 24-bit files are not memory-mapped and have no Zarr dtype | B (E for 24-bit) | [S] `behavior/audio/audiointerface.py:294-296` |
| Video, external | `ExternalVideoInterface` | NWB `external_file` | already a reference | F | [S] `behavior/video/externalvideointerface.py:26` |
| Video, internal | `InternalVideoInterface` | decoded with OpenCV | video codec | E | [S] `internalvideointerface.py:422` |
| SLEAP, DeepLabCut, LightningPose | pose interfaces | HDF5 or CSV tables | regrouped into one series per body part; small | F (DLC `.h5` would be D in principle) | [S] |
| TDT fiber photometry | `TDTFiberPhotometryInterface` -> `tdt.read_block` | same `.tev`/`.tsq` chunks as above | several stores stacked into the columns of one series; small | F | [S] `fiber_photometry/tdt/tdtfiberphotometrydatainterface.py:534-536` |
| Doric | `DoricFiberPhotometryInterface` -> h5py | HDF5 1-D datasets, read whole | small; columns may be stacked | A when one dataset maps to one series, else F | [S] `doricfiberphotometrydatainterface.py:289-290` |

## Things that would silently give wrong data

1. **SpikeInterface negates the samples when every NEO gain is negative** (`neoextractors/neobaseextractor.py:251-255, 375-376`) [S]. NEO gives Neuralynx channels a negative gain when `InputInverted` is true (`neuralynxrawio.py:289-291`), so NeuroConv stores `-raw`. A reference to the file bytes holds `+raw`; the series needs a negative `conversion`, and a test that compares against a normal NeuroConv conversion must compare scaled values, not stored ones.
2. **Neuralynx partial records and gaps.** The last record of a section can have `nb_valid` < 512 and the file can hold several sections. In `Cheetah_v5.7.4/original_data/CSC1.ncs` NEO reports 1,395,036 samples in 4 segments while the records hold 1,396,224 [F]. One chunk per record is only right for a file with one section and all records full; otherwise one array per segment and a trimmed last chunk.
3. **Open Ephys legacy gaps.** NEO fills lost USB packets with zeros (`openephysrawio.py:140-147`) [S]; the file has no bytes for them.
4. **Blackrock pauses and PTP.** Each pause starts a new data block and a new segment [F]. PTP files have one packet per sample and can have gaps in the timestamps. Spec 2.1 files lose their last sample row in NEO [F].
5. **Channel-major data inside records.** Intan header-attached blocks and EDF records store each channel's samples together [F]. Referencing a block as (time, channel) without the transpose codec scrambles the data and raises no error.
6. **Value inversion.** Scanbox (`65535 - x`) and Biocam with `SignalInversion = -1` (`4096 - x`) are inverted by the reader [S].
7. **Unsigned samples with an offset.** Intan header-attached amplifier data is uint16 around 32768 [F]; NeuroConv writes the offset in `ElectricalSeries.offset`.
8. **Different rates in one file.** EDF signals, Intan aux (1/4 rate) and supply (1 per block) channels, and TDT stores have their own lengths; each needs its own array.
9. **Big-endian samples** in Open Ephys legacy [F].
10. **Many source files in one array** (Bruker, Neuralynx, Intan per-channel, Open Ephys legacy): the `sources` map and its change checks grow with the file count.
11. **Icephys series are single columns** of an interleaved block (see ABF above).
12. **Tiny records.** TDT (64 to a few thousand samples), EDF (often 1 s), Intan (60 or 128 samples), Neuralynx (512) and Open Ephys legacy (1024) give chunks of 0.1 to 16 kB per channel, so remote reads depend on request merging.

## Corrections to the earlier classification

| Earlier statement | What the source and files show |
|---|---|
| MDA is a "header followed by one raw array" target | NeuroConv only has `MdaSortingInterface`; there is no MDA recording interface. F. |
| Scanbox is a "header followed by one raw array" | It has no header, needs the transpose codec, and the reader inverts the values. Only single-channel, single-plane files are supported by the extractor. |
| Inscopix is a raw array | Not verifiable from the reader, which calls the vendor library. Unknown. |
| Intan's traditional block format needs the byte-selection rule | It needs one chunk per block plus the transpose codec, because the amplifier region is contiguous within a block and channel-major. Byte selection cannot produce time-major chunks larger than a block. |
| Blackrock is contiguous when the recording has no pauses | True for specs 2.1 to 3.0. Spec 3.0 PTP files are per-sample packets and need the byte-selection rule. |
| EDF works when all channels share a sampling rate | It works per NEO stream (one per rate) with a (samples, 1) chunk per record and signal; the layout is channel-major inside a record. BDF is not feasible (24-bit). |
| TDT `.sev` files are one raw stream per channel | NEO addresses `.sev` files through the `.tsq` index like `.tev`; contiguity is not verified because the GIN `.sev` files are stubs. `.tev` data is feasible only as many tiny per-channel chunks with an explicit index. |
| Axona is not worth attempting because of remapped channel order | The permutation is fixed and could be absorbed by the order of the electrodes region. The real obstacle is that only the active tetrodes are exposed, which is several byte spans per packet. |
| SLEAP is an HDF5 target | It is regrouped into pose series; F. |
| Missing from the earlier list | Open Ephys legacy (C, big-endian, one file per channel); Biocam needs a 2-D view of a 1-D dataset; ABF needs one column per series; every imaging format needs the transpose codec to match NeuroConv's (frames, width, height). |

## Cross-cutting findings

- **The transpose codec is needed more often than the byte-selection rule.** All imaging, Intan header-attached, EDF, Suite2p and CaImAn traces need it. zindi does not emit it, and zarr-python's uncompressed partial read does not apply when it is present.
- **Of NeuroConv's ephys formats, NEO's buffer API covers six**: SpikeGLX, Open Ephys binary, Neuroscope, MCS raw, Axon and Maxwell (not usable). The other seven buffer-API readers (BrainVision, Elan, Micromed, NeuroNexus, raw binary, WinEDR, WinWCP) have no NeuroConv interface.
- **Hooks from an interface to the layout.** `interface.recording_extractor.neo_reader`, `.stream_id`, `.stream_index`, `.block_index`, `.inverted_gain` (`neobaseextractor.py:27, 224-225, 254`); for binary extractors `recording._kwargs`; for TIFF `extractor._frames_to_ifd_table`. NeuroConv may wrap the extractor (channel slices, segment concatenation), so the hook has to walk to the parent.
- **A column permutation never needs a codec**: the `electrodes` region of an `ElectricalSeries` can list electrodes in file order. A column subset does.
- **Byte-selection users**: SpikeGLX sync column, Open Ephys binary mixed channels, SpikeGadgets, Blackrock PTP, multi-channel ABF, Axona (several spans).

## Best next targets

1. **Blackrock nsX (specs 2.1 to 3.0)**: widely used, one contiguous int16 block per segment, and the offsets come straight from the parsed headers. A small generator; PTP files follow once byte selection exists.
2. **TIFF imaging through the extractors' page tables (ScanImage, Bruker, multi-page TIFF)**: the largest data volumes NeuroConv handles, and the TIFF generator already exists. It needs one chunk per page chosen through `_frames_to_ifd_table`, and the transpose codec, which then serves every other imaging format.
3. **SpikeGadgets**: the cleanest real test of the byte-selection rule (verified on three GIN files), and the same rule finishes SpikeGLX, Open Ephys binary and Blackrock PTP.
4. **Intan**: the two split-file modes are plain binaries verified on GIN files; the header-attached mode is the first user of strided chunks with the transpose codec, which EDF reuses.

WhiteMatter, CellExplorer and WAV are a few lines each once a raw-binary generator takes `file_offset`, `dtype` and `num_channels`.
